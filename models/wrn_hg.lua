--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The full pre-activation ResNet variation from the technical report
-- "Identity Mappings in Deep Residual Networks" (http://arxiv.org/abs/1603.05027)
--

require 'nn'
require 'cunn'
require 'nngraph'

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iChannels

   -- Typically shareGradInput uses the same gradInput storage for all modules
   -- of the same type. This is incorrect for some SpatialBatchNormalization
   -- modules in this network b/c of the in-place CAddTable. This marks the
   -- module so that it's shared only with other modules with the same key
   local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
   end

   local function wide_basic(nInputPlane, nOutputPlane, stride)
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane

      local block = nn.Sequential()
      local convs = nn.Sequential()

      for i,v in ipairs(conv_params) do
         if i == 1 then
            local module = nInputPlane == nOutputPlane and convs or block
            module:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
            module:add(ReLU(true))
            convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
         else
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(true))
            if opt.dropout > 0 then
               convs:add(nn.Dropout(opt and opt.dropout or 0,nil,true))
            end
            convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
         end
      end

      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)

      return block
         :add(nn.ConcatTable()
            :add(convs)
            :add(shortcut))
         :add(nn.CAddTable(true))
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()
      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   -- Three 1x1 convolutions
   local function conv1x1s(nInputPlane, nOutputPlane, count)
      count = count or 3
      local s = nn.Sequential()
      for i=1,count do
         if i == 1 then
            s:add(ShareGradInput(SBatchNorm(nInputPlane), 'conv1x1'))
            s:add(ReLU(true))
            s:add(Convolution(nInputPlane,nOutputPlane,1,1))
         else
            s:add(SBatchNorm(nOutputPlane)):add(ReLU(true))
            s:add(Convolution(nOutputPlane,nOutputPlane,1,1))
         end
      end
      return s
   end

   local function score(nInputPlane, nInputSpatialDim, nOutputPlane)
      local s = nn.Sequential()
      s:add(ShareGradInput(SBatchNorm(nInputPlane), 'last'))
      s:add(ReLU(true))
      s:add(Avg(nInputSpatialDim, nInputSpatialDim, 1, 1))
      s:add(nn.View(nInputPlane):setNumInputDims(3))
      s:add(nn.Linear(nInputPlane, nOutputPlane))
      return s
   end

   local function upadd(top, bottom, nTopPlane, nBottomPlane)
      -- Up sampling the top
      local a = ShareGradInput(nn.SpatialUpSamplingNearest(2), 'upadd')(top)
      -- Transform the bottom with three 1x1 conv
      local b = conv1x1s(nBottomPlane, nTopPlane)(bottom)
      -- Add
      return nn.CAddTable(){a, b}
   end


   local input = nn.Identity()()
   local outputs
   if opt.dataset == 'cifar10' then
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6
      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      -- First hourglass
      local conv1 = Convolution(3,nStages[1],3,3,1,1,1,1)(input) -- one conv at the beginning (spatial size: 32x32)
      local res1 = layer(wide_basic, nStages[1], nStages[2], n, 1)(conv1) -- Stage 1 (spatial size: 32x32)
      local res2 = layer(wide_basic, nStages[2], nStages[3], n, 2)(res1) -- Stage 2 (spatial size: 16x16)
      local res3 = layer(wide_basic, nStages[3], nStages[4], n, 2)(res2) -- Stage 3 (spatial size: 8x8)
      local score1 = score(nStages[4], 8, 10)(res3)
      local up1 = upadd(res3, res2, nStages[4], nStages[3])
      local up2 = upadd(up1, res1, nStages[4], nStages[2])
      -- Second half-hourglass
      local second_input = conv1x1s(nStages[4], nStages[1])(up2)
      res1 = layer(wide_basic, nStages[1], nStages[2], n, 1)(second_input) -- Stage 1 (spatial size: 32x32)
      res2 = layer(wide_basic, nStages[2], nStages[3], n, 2)(res1) -- Stage 2 (spatial size: 16x16)
      res3 = layer(wide_basic, nStages[3], nStages[4], n, 2)(res2) -- Stage 3 (spatial size: 8x8)
      local score2 = score(nStages[4], 8, 10)(res3)
      -- Combine scores
      local final = nn.CAddTable(){nn.CMul(10)(score1), score2}
      outputs = {score1, final}
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local model = nn.gModule({input}, outputs)

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         if cudnn.version >= 4000 then
            v.bias = nil
            v.gradBias = nil
         else
            v.bias:zero()
         end
      end
   end
   local function BNInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.weight:fill(1)
         v.bias:zero()
      end
   end

   ConvInit('cudnn.SpatialConvolution')
   ConvInit('nn.SpatialConvolution')
   BNInit('fbnn.SpatialBatchNormalization')
   BNInit('cudnn.SpatialBatchNormalization')
   BNInit('nn.SpatialBatchNormalization')
   for k,v in pairs(model:findModules('nn.Linear')) do
      v.bias:zero()
   end
   model:cuda()

   if opt.cudnn == 'deterministic' then
      model:apply(function(m)
         if m.setMode then m:setMode(1,1,1) end
      end)
   end

   model:get(1).gradInput = nil

   return model
end

return createModel