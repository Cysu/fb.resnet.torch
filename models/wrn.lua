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

local Convolution = cudnn.SpatialConvolution
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local Max = nn.SpatialMaxPooling
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt)
   local depth = opt.depth
   local shortcutType = opt.shortcutType or 'B'
   local iSize

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

      iSize = iSize / stride
      local attention = nn.Sequential()
         :add(Convolution(nInputPlane,1,1,1,stride,stride,0,0))
         :add(nn.View(-1):setNumInputDims(3))
         :add(nn.SoftMax())
         :add(nn.View(1, iSize, iSize):setNumInputDims(1))
         :add(nn.Replicate(nOutputPlane, 1, 3))

      local residual = nn.Sequential()
         :add(nn.ConcatTable():add(convs):add(attention))
         :add(nn.CMulTable())

      local shortcut = nInputPlane == nOutputPlane and
         nn.Identity() or
         Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0)

      return block
         :add(nn.ConcatTable()
            :add(residual)
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

   local model = nn.Sequential()
   if opt.dataset == 'imagenet' then
      local cfg = {
         [18]  = {2, 2, 2, 2},
         [34]  = {3, 4, 6, 3}
      }

      assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
      local n = cfg[depth]
      local k = opt.widen_factor
      local nStages = torch.Tensor{64, 64*k, 128*k, 256*k, 512*k}

      -- The ResNet ImageNet model
      model:add(Convolution(3,64,7,7,2,2,3,3))
      model:add(SBatchNorm(64))
      model:add(ReLU(true))
      model:add(Max(3,3,2,2,1,1))
      model:add(layer(wide_basic, nStages[1], nStages[2], n[1], 1))
      model:add(layer(wide_basic, nStages[2], nStages[3], n[2], 2))
      model:add(layer(wide_basic, nStages[3], nStages[4], n[3], 2))
      model:add(layer(wide_basic, nStages[4], nStages[5], n[4], 2))
      model:add(ShareGradInput(SBatchNorm(nStages[5]), 'last'))
      model:add(ReLU(true))
      model:add(Avg(7, 7, 1, 1))
      model:add(nn.View(nStages[5]):setNumInputDims(3))
      model:add(nn.Linear(nStages[5], 1000))
   elseif opt.dataset == 'cifar10' or opt.dataset == 'cifar100' then
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6

      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      iSize = 32

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(wide_basic, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(wide_basic, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(ShareGradInput(SBatchNorm(nStages[4]), 'last'))
      model:add(ReLU(true))
      model:add(Avg(8, 8, 1, 1))
      model:add(nn.View(nStages[4]):setNumInputDims(3))
      local nClasses = opt.dataset == 'cifar10' and 10 or 100
      model:add(nn.Linear(nStages[4], nClasses))
   else
      error('invalid dataset: ' .. opt.dataset)
   end

   local function ConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         if v.nOutputPlane == 1 then
            -- Attention branch
            v.weight:normal(0, 0.001)
            v.bias:fill(1)
         else
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