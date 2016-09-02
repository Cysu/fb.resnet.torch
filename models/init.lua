--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Generic model creating code. For the specific ResNet model see
--  models/resnet.lua
--

require 'nn'
require 'cunn'
require 'cudnn'
require 'models/SequentialDropout'

local M = {}
local Convolution = cudnn.SpatialConvolution

function M.setup(opt, checkpoint)
   local model
   if checkpoint then
      local modelPath = paths.concat(opt.resume, checkpoint.modelFile)
      assert(paths.filep(modelPath), 'Saved model not found: ' .. modelPath)
      print('=> Resuming model from ' .. modelPath)
      model = torch.load(modelPath):cuda()
   elseif opt.retrain ~= 'none' then
      assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
      print('Loading model from file: ' .. opt.retrain)
      model = torch.load(opt.retrain):cuda()
   else
      print('=> Creating model from file: models/' .. opt.netType .. '.lua')
      model = require('models/' .. opt.netType)(opt)
   end

   -- Make the loaded network fully convolutional
   local sz = model:size()
   local fc = model:get(sz)
   while true do
      model:remove()
      sz = sz - 1
      if torch.type(model:get(sz)):find("ReLU") then break end
   end
   local nInputPlane = fc.weight:size(2)
   local nOutputPlane = fc.weight:size(1)
   local conv1 = Convolution(nInputPlane, nOutputPlane, 1, 1, 1, 1)
   local w = fc.weight:view(nOutputPlane, nInputPlane, 1, 1) 
   conv1.weight:copy(w)
   conv1.bias:copy(fc.bias)
   model:add(conv1)
   --

   -- First remove any DataParallelTable
   if torch.type(model) == 'nn.DataParallelTable' then
      model = model:get(1)
   end

   -- optnet is an general library for reducing memory usage in neural networks
   if opt.optnet then
      local optnet = require 'optnet'
      local imsize = opt.dataset == 'imagenet' and 224 or 32
      local sampleInput = torch.zeros(4,3,imsize,imsize):cuda()
      optnet.optimizeMemory(model, sampleInput, {inplace = false, mode = 'training'})
   end

   -- This is useful for fitting ResNet-50 on 4 GPUs, but requires that all
   -- containers override backwards to call backwards recursively on submodules
   if opt.shareGradInput then
      local function sharingKey(m)
         local key = torch.type(m)
         if m.__shareGradInputKey then
            key = key .. ':' .. m.__shareGradInputKey
         end
         return key
      end

      -- Share gradInput for memory efficient backprop
      local cache = {}
      model:apply(function(m)
         local moduleType = torch.type(m)
         -- Revise these part to seperate RFCN from the frame of sharing gradient input.
         if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' and m.rfcn ~= true then
            local key = sharingKey(m)
            if cache[key] == nil then
               cache[key] = torch.CudaStorage(1)
            end
            m.gradInput = torch.CudaTensor(cache[key], 1, 0)
         end
      end)
      for i, m in ipairs(model:findModules('nn.ConcatTable')) do
         if m.rfcn ~= true then
            if cache[i % 2] == nil then
               cache[i % 2] = torch.CudaStorage(1)
            end
            m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
         end
      end
   end

   -- For resetting the classifier when fine-tuning on a different Dataset
   if opt.resetClassifier and not checkpoint then
      print(' => Replacing classifier with ' .. opt.nClasses .. '-way classifier')

      local orig = model:get(#model.modules)
      assert(torch.type(orig) == 'nn.Linear',
         'expected last layer to be fully connected')

      local linear = nn.Linear(orig.weight:size(2), opt.nClasses)
      linear.bias:zero()

      model:remove(#model.modules)
      model:add(linear:cuda())
   end

   -- Set the CUDNN flags
   if opt.cudnn == 'fastest' then
      cudnn.fastest = true
      cudnn.benchmark = true
   elseif opt.cudnn == 'deterministic' then
      -- Use a deterministic convolution implementation
      model:apply(function(m)
         if m.setMode then m:setMode(1, 1, 1) end
      end)
   end

   -- Wrap the model with DataParallelTable, if using more than one GPU
   if opt.nGPU > 1 then
      local gpus = torch.range(opt.startGPU, opt.startGPU + opt.nGPU - 1):totable()
      local fastest, benchmark = cudnn.fastest, cudnn.benchmark

      local dpt = nn.DataParallelTable(1, true, true)
         :add(model, gpus)
         :threads(function()
            local cudnn = require 'cudnn'
            require 'models/SequentialDropout'
            cudnn.fastest, cudnn.benchmark = fastest, benchmark
         end)
      dpt.gradInput = nil

      model = dpt:cuda()
   end


   -- Revise DataParallelTable, make it capable of receiving images of different sizes on GPUs. 
   model._concatTensorRecursive = function(self, dst, src)  
      cutorch.setDevice(self.gpuAssignments[1])
      dst = {}

      for i, ret in ipairs(src) do
         local buff = torch.CudaTensor()
         local sz = #ret
         if #sz ~= 0 then
            buff:resize(ret:size()):copy(ret)
            dst[i] = buff
         end
      end

      return dst 
   end
   model._distributeTensorRecursive = function(self, dst, src, idx, n)
      local dst = torch.type(dst) == 'torch.CudaTensor' and dst or torch.CudaTensor()

      if idx > #src then
         dst:resize(0)
         return dst
      end

      dst:resize(src[idx]:size()):copyAsync(src[idx])
      -- WaitForDevices
      if src[idx].getDevice then
         local stream = cutorch.getStream()
         if stream ~= 0 then
            cutorch.streamWaitForMultiDevice(dst:getDevice(), stream, {[src[idx]:getDevice()] = {stream} })
         end
      end
      return dst
   end

   -- DEBUG
   -- tin = {}
   -- tin[1] = torch.Tensor(1, 3, 224, 224):cuda()
   -- tin[2] = torch.Tensor(1, 3, 223, 223):cuda()
   -- tin[3] = torch.Tensor(1, 3, 256, 256):cuda()
   -- tin[4] = torch.Tensor(1, 3, 256, 224):cuda()
   -- tin[5] = torch.Tensor(1, 3, 224, 224):cuda()
   -- tin[6] = torch.Tensor(1, 3, 223, 223):cuda()
   -- tin[7] = torch.Tensor(1, 3, 256, 256):cuda()
   -- local tst = model:forward(tin)
   --

   local criterion = nn.CrossEntropyCriterion():cuda()
   return model, criterion
end

return M
