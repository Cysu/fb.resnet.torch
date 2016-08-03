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

local nn = require 'nn'
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

   local function wide_basic(nInputPlane, nOutputPlane, stride, input)
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane

      if nInputPlane ~= nOutputPlane then
         local block = nn.Sequential()
         block:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
         block:add(ReLU(true))
         input = block(input)
      end

      local function getConvs()
        local convs = nn.Sequential()
        for i,v in ipairs(conv_params) do
           if i == 1 then
              if nInputPlane == nOutputPlane then
                 convs:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
                 convs:add(ReLU(true))
              end
              convs:add(Convolution(nInputPlane,nBottleneckPlane,table.unpack(v)))
           else
              convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(true))
              if opt.dropout > 0 then
                 convs:add(nn.Dropout(opt and opt.dropout or 0,nil,true))
              end
              convs:add(Convolution(nBottleneckPlane,nBottleneckPlane,table.unpack(v)))
           end
        end
        return convs
      end

      local function getMul(numPoly)
         local s = nn.Mul()
         s.weight:fill(1. / numPoly)
         return s
      end

      local numPoly = nInputPlane == nOutputPlane and opt.nPoly or 1

      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)
      local convs = getConvs()

      local mid = getMul(numPoly)(convs(input))

      local branches = {shortcut(input), mid}
      for i = 2, numPoly do
         local newConvs = getConvs()
         newConvs:share(convs, 'weight', 'bias', 'gradWeight', 'gradBias')
         mid = getMul(numPoly)(newConvs(mid))
         table.insert(branches, mid)
      end

      return nn.CAddTable(true)(branches)
   end

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride, input)
      local output = block(nInputPlane, nOutputPlane, stride, input)
      for i=2,count do
         output = block(nOutputPlane, nOutputPlane, 1, output)
      end
      return output
   end

   local input = nn.Identity()()
   local outputs
   if opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6

      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      local out = Convolution(3,nStages[1],3,3,1,1,1,1)(input) -- one conv at the beginning (spatial size: 32x32)
      out = layer(wide_basic, nStages[1], nStages[2], n, 1, out) -- Stage 1 (spatial size: 32x32)
      out = layer(wide_basic, nStages[2], nStages[3], n, 2, out) -- Stage 2 (spatial size: 16x16)
      out = layer(wide_basic, nStages[3], nStages[4], n, 2, out) -- Stage 3 (spatial size: 8x8)

      local last = nn.Sequential()
      last:add(ShareGradInput(SBatchNorm(nStages[4]), 'last'))
      last:add(ReLU(true))
      last:add(Avg(8, 8, 1, 1))
      last:add(nn.View(nStages[4]):setNumInputDims(3))
      local nClasses = opt.dataset == 'cifar10' and 10 or 100
      last:add(nn.Linear(nStages[4], nClasses))

      outputs = {last(out)}
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
