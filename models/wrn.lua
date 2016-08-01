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
local BatchNorm = nn.BatchNormalization
local DilatedConv = nn.SpatialDilatedConvolution

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

   -- wide_basic_dilated
   local function wide_basic_dilated(nInputPlane, nOutputPlane, stride)
      local conv_params = {
         {3,3,stride,stride,1,1},
         {3,3,1,1,1,1},
      }
      local nBottleneckPlane = nOutputPlane
      local nHalfBottleneckPlane = nBottleneckPlane / 2 

      local block = nn.Sequential()
      local convs = nn.Sequential()

      for i,v in ipairs(conv_params) do
         if i == 1 then
            local module = nInputPlane == nOutputPlane and convs or block
            module:add(ShareGradInput(SBatchNorm(nInputPlane), 'preact'))
            module:add(ReLU(true))
            local concat1 = nn.Concat(2)
            concat1:add(Convolution(nInputPlane,nHalfBottleneckPlane,table.unpack(v))) 
            concat1:add(DilatedConv(nInputPlane,nHalfBottleneckPlane,3,3,stride,stride,opt.nDilation,opt.nDilation,opt.nDilation,opt.nDilation))

            convs:add(concat1)
         else
            convs:add(SBatchNorm(nBottleneckPlane)):add(ReLU(true))
            if opt.dropout > 0 then
               convs:add(nn.Dropout(opt and opt.dropout or 0,nil,true))
            end
            local concat2 = nn.Concat(2)
            concat2:add(Convolution(nBottleneckPlane,nHalfBottleneckPlane,table.unpack(v)))
            concat2:add(DilatedConv(nBottleneckPlane,nHalfBottleneckPlane,3,3,1,1,opt.nDilation,opt.nDilation,opt.nDilation,opt.nDilation))
            convs:add(concat2)
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
   --

   -- Stacking Residual Units on the same stage
   local function layer(block, nInputPlane, nOutputPlane, count, stride)
      local s = nn.Sequential()
      s:add(block(nInputPlane, nOutputPlane, stride))
      for i=2,count do
         s:add(block(nOutputPlane, nOutputPlane, 1))
      end
      return s
   end

   -- Multi Pose-subclass Layer
   local function multiclass(nInputPlane, nOutputPlane)
      local s = nn.Sequential()
      s:add(nn.Dropout(0.1))
      local tableOut = nn.Concat(2) 
      for i = 1, nOutputPlane do
         local unit = nn.Sequential()
         unit:add(nn.Linear(nInputPlane, opt.multiFactor))
         unit:add(ShareGradInput(BatchNorm(opt.multiFactor), 'multi'))
         unit:add(ReLU(true))
         unit:add(nn.Linear(opt.multiFactor, 1))
         tableOut:add(unit)
      end
      s:add(tableOut)
      return s
   end

   -- RCN-Regional Convolutional Network
   local function rcn(nInputPlane, nOutputPlane)
      local s = nn.Sequential()
      local length = 4 
      local table = nn.ConcatTable()
      -- build the 4-part layer
      for i = 1, 2 do
         for j = 1, 2 do
            local part = nn.Sequential()
            local offset3 = (i - 1) * length + 1
            local offset4 = (j - 1) * length + 1
            part:add(nn.Narrow(3, offset3, length))
            part:add(nn.Narrow(4, offset4, length))
            -- 1x1 Convolution
            part:add(Convolution(nInputPlane, nOutputPlane, 1, 1))
            part:add(Avg(length, length, 1, 1))
            table:add(part)
         end
      end

      s:add(table) 
      s:add(nn.CAddTable())
      s:add(nn.View(nOutputPlane):setNumInputDims(3))
      -- s:add(unit)
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
      -- Multi-subclass
      if opt.widen_factor == 1 then
         model:add(Avg(7, 7, 1, 1))
         model:add(nn.View(nStages[5]):setNumInputDims(3))
         model:add(nn.Linear(nStages[5], 1000))
      else
         model:add(rcn(nStages[5], 1000))
      end
   elseif opt.dataset == 'cifar10' then
      assert((depth - 4) % 6 == 0, 'depth should be 6n+4')
      local n = (depth - 4) / 6

      local k = opt.widen_factor
      local nStages = torch.Tensor{16, 16*k, 32*k, 64*k}

      model:add(Convolution(3,nStages[1],3,3,1,1,1,1)) -- one conv at the beginning (spatial size: 32x32)
      local block = opt.nDilation == 1 and wide_basic or wide_basic_dilated
      model:add(layer(block, nStages[1], nStages[2], n, 1)) -- Stage 1 (spatial size: 32x32)
      model:add(layer(block, nStages[2], nStages[3], n, 2)) -- Stage 2 (spatial size: 16x16)
      model:add(layer(block, nStages[3], nStages[4], n, 2)) -- Stage 3 (spatial size: 8x8)
      model:add(ShareGradInput(SBatchNorm(nStages[4]), 'last'))
      model:add(ReLU(true))

      -- Multi pose subclass
      if opt.multiFactor > 1 then 
         model:add(rcn(nStages[4], 10))
      else
         model:add(Avg(8, 8, 1, 1))
         model:add(nn.View(nStages[4]):setNumInputDims(3))
         model:add(nn.Linear(nStages[4], 10))
      end
   else
      error('invalid dataset: ' .. opt.dataset)
   end

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
   -- Dilated Convolution Init
   local function DiConvInit(name)
      for k,v in pairs(model:findModules(name)) do
         v.bias:zero() 
         local n = v.kW*v.kH*v.nOutputPlane
         v.weight:normal(0,math.sqrt(2/n))
         local nInp = v.nInputPlane
         local nOut = v.nOutputPlane
         for i = 1, nInp < nOut and nInp or nOut do
            v.weight[i][i][2][2] = 1
         end
         if nInp < nOut then
            for i = nInp + 1, nOut do
               v.weight[i][torch.random() % nInp + 1][2][2] = 1
            end
         end
      end
   end
   --
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
   DiConvInit('nn.SpatialDilatedConvolution')
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
