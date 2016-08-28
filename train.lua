--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  The training loop and learning rate schedule
--

local optim = require 'optim'

local M = {}
local Trainer = torch.class('resnet.Trainer', M)
local Avg = cudnn.SpatialAveragePooling

function Trainer:__init(model, criterion, opt, optimState)
   self.model = model
   self.criterion = criterion
   self.optimState = optimState or {
      learningRate = opt.LR,
      learningRateDecay = 0.0,
      momentum = opt.momentum,
      nesterov = true,
      dampening = 0.0,
      weightDecay = opt.weightDecay,
   }
   self.opt = opt
   self.params, self.gradParams = model:getParameters()
   self.logFile = opt.logFile
end

function Trainer:train(epoch, dataloader)
   -- Trains the model for a single epoch
   self.optimState.learningRate = self:learningRate(epoch)

   local timer, dataTimer = torch.Timer(), torch.Timer()
   local totalTime, totalDataTime = 0, 0

   local function feval()
      return self.criterion.output, self.gradParams
   end

   local trainSize = dataloader:size()
   local top1Sum, top5Sum, lossSum = 0.0, 0.0, 0.0
   local N = 0

   print('=> Training epoch # ' .. epoch)
   -- set the batch norm to training mode
   self.model:training()
   
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real
      totalDataTime = totalDataTime + dataTime

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      local output = self.model:forward(self.input):float()
      local batchSize = output:size(1)
      local loss = self.criterion:forward(self.model.output, self.target)

      self.model:zeroGradParameters()
      self.criterion:backward(self.model.output, self.target)
      self.model:backward(self.input, self.criterion.gradInput)

      optim.sgd(feval, self.params, self.optimState)

      local top1, top5 = self:computeScore(output, sample.target, 1, 1)
      top1Sum = top1Sum + top1*batchSize
      top5Sum = top5Sum + top5*batchSize
      lossSum = lossSum + loss*batchSize
      N = N + batchSize

      local time = timer:time().real
      totalTime = totalTime + time
      print((' | Epoch: [%d][%d/%d]    Time %.3f (%.3f)  Data %.3f (%.3f)  Err %1.4f (%1.4f)  top1 %7.3f (%.3f)  top5 %7.3f (%6.3f)'):format(
         epoch, n, trainSize, time, totalTime / N, dataTime, totalDataTime / N, loss, lossSum / N, top1, top1Sum / N, top5, top5Sum / N))
      if self.logFile and n == trainSize then
         self.logFile:write((' | Epoch: [%d][%d/%d]    Time %.3f  Data %.3f  Err %1.4f  top1 %7.3f  top5 %7.3f'):format(
            epoch, n, trainSize, timer:time().real, dataTime, loss, top1Sum / N, top5Sum / N) .. '\n')
         self.logFile:flush()
      end

      -- check that the storage didn't get changed due to an unfortunate getParameters call
      if not self.opt.fixPretrain then
         assert(self.params:storage() == self.model:parameters()[1]:storage())
      end

      timer:reset()
      dataTimer:reset()
   end

   return top1Sum / N, top5Sum / N, lossSum / N
end

function Trainer:test(epoch, dataloader)
   -- Computes the top-1 and top-5 err on the validation set

   local timer = torch.Timer()
   local dataTimer = torch.Timer()
   local size = dataloader:size()

   local nCrops = self.opt.tenCrop and 10 or 1
   local nScales = self.opt.nScales
   local top1Sum, top5Sum = 0.0, 0.0
   local N = 0
   
   self.model:evaluate()
   
   local scores = torch.CudaTensor()
   local target = torch.CudaTensor(dataloader.__size):cuda()
   -- scores:zeros(nScales, dataloader.__size, 1000):cuda()

   for i = 1, nScales do
      N = 0
      -- For saveing
      scores = {}
      --
      for n, sample in dataloader:run(i) do
         local dataTime = dataTimer:time().real

         -- Copy input and target to the GPU
         self:copyInputs(sample) 

         -- local output = self.model:forward(self.input):float()
         -- local batchSize = output:size(1) / nCrops
         local output = self.model:forward(self.input)
         local batchSize = #output

         -- Different types of multi-scale test.
         -- if self.opt.testType == 1 then
         --    for j = 1, batchSize do
         --       local pooling = Avg(output[j]:size(4), output[j]:size(3), 1, 1)
         --       pooling:cuda()
         --       output[j]:cuda()
         --       output[j] = pooling:forward(output[j])
         --       output[j] = output[j]:view(output[j]:size(2))
         --    end
         -- elseif self.opt.testType == 2 then
         --    local softMax = nn.SoftMax() 
         --    softMax:cuda()
         --    for j = 1, batchSize do
         --       local pooling = Avg(output[j]:size(4), output[j]:size(3), 1, 1)
         --       pooling:cuda()
         --       output[j]:cuda()
         --       output[j] = pooling:forward(output[j])
         --       output[j] = softMax:forward(output[j])
         --       output[j] = output[j]:view(output[j]:size(2))
         --    end
         -- else
         --    local softMax = nn.SoftMax() 
         --    softMax:cuda()
         --    for j = 1, batchSize do
         --       local pooling = Avg(output[j]:size(4), output[j]:size(3), 1, 1)
         --       pooling:cuda()
         --       output[j]:cuda()
         --       output[j] = softMax:forward(output[j]:cuda())
         --       output[j] = pooling:forward(output[j])
         --       output[j] = output[j]:view(output[j]:size(2))
         --    end
         -- end

         -- for j = 1, batchSize do
         --    scores[i][N + j]:add(output[j])
         -- end
         -- for saving
         for j = 1, batchSize do
            scores[N + j] = output[j]:double()
         end
         -- 

         if i == 1 then
            target:narrow(1, N + 1, batchSize):copy(sample.target)
         end
         N = N + batchSize
         print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f'):format(
            i, n, size, timer:time().real, dataTime))
         timer:reset()
         dataTimer:reset()
      end

      -- Save scores of different scales
      assert(not (not paths.dirp('scores') and not paths.mkdir('scores')),
         'error: unable to create scores directory\n')
      -- For saving
      torch.save('scores/scores_' .. tostring(self.opt.scales[i]) .. '.t7', scores)
      --
      -- torch.save('scores/score_' .. tostring(self.opt.scales[i]) .. '.t7', scores[i])  
   end

   -- For saving
   torch.save('scores/target.t7', target:double())
   os.exit(0)
   --

   local top1, top5 = self:computeScore(scores, target, nCrops, nScales)
   top1Sum = top1 * N
   top5Sum = top5 * N
   -- top1Sum = top1Sum + top1*batchSize
   -- top5Sum = top5Sum + top5*batchSize

   -- print((' | Test: [%d][%d/%d]    Time %.3f  Data %.3f  top1 %7.3f (%7.3f)  top5 %7.3f (%7.3f)'):format(
   --    epoch, n, size, timer:time().real, dataTime, top1, top1Sum / N, top5, top5Sum / N))

   self.model:training()

   print((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
      epoch, top1Sum / N, top5Sum / N))

   if self.logFile then
      self.logFile:write((' * Finished epoch # %d     top1: %7.3f  top5: %7.3f\n'):format(
         epoch, top1Sum / N, top5Sum / N) .. '\n')
      self.logFile:flush()
   end

   return top1Sum / N, top5Sum / N
end

function Trainer:recomputeBatchNorm(dataloader)
   local timer = torch.Timer()
   local dataTimer = torch.Timer()

   local size = math.min(1000, dataloader:size())
   local N = 0

   local batchNorms = {}
   local means = {}
   local variances = {}
   local momentums = {}
   for _, m in ipairs(self.model:listModules()) do
      if torch.isTypeOf(m, 'nn.BatchNormalization') then
         table.insert(batchNorms, m)
         table.insert(means, m.running_mean:clone():zero())
         table.insert(variances, m.running_var:clone():zero())
         table.insert(momentums, m.momentum)
         -- Set momentum to 1
         m.momentum = 1
      end
   end

   print('=> Recomputing batch normalization staticstics')
   self.model:training()
   for n, sample in dataloader:run() do
      local dataTime = dataTimer:time().real

      -- Copy input and target to the GPU
      self:copyInputs(sample)

      -- Compute forward pass
      self.model:forward(self.input)

      -- Update running sum of batch mean and variance
      for i, sbn in ipairs(batchNorms) do
         means[i]:add(sbn.running_mean)
         variances[i]:add(sbn.running_var)
      end
      N = N + 1

      print((' | BatchNorm: [%d/%d]    Time %.3f  Data %.3f'):format(
         n, size, timer:time().real, dataTime))

      timer:reset()
      dataTimer:reset()

      if N == size then
         break
      end
   end

   for i, sbn in ipairs(batchNorms) do
      sbn.running_mean:copy(means[i]):div(N)
      sbn.running_var:copy(variances[i]):div(N)
      sbn.momentum = momentums[i]
   end

   -- Copy over running_mean/var from first GPU to other replicas, if using DPT
   if torch.type(self.model) == 'nn.DataParallelTable' then
      self.model.impl:applyChanges()
   end
end

function Trainer:computeScore(output, target, nCrops, nScales)
   -- if nCrops > 1 then
   --    -- Sum over crops
   --    output = output:view(output:size(1) / nCrops, nCrops, output:size(2))
   --       --:exp()
   --       :sum(2):squeeze(2)
   -- end

   -- Sum up scores from different scales
   output = output:sum(1):view(output:size(2), output:size(3))

   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _ , predictions = output:float():sort(2, true) -- descending

   -- Find which predictions match the target
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))

   -- Top-1 score
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)

   -- Top-5 score, if there are at least 5 classes
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)

   return top1 * 100, top5 * 100
end

function Trainer:copyInputs(sample)
   -- Copies the input to a CUDA tensor, if using 1 GPU, or to pinned memory,
   -- if using DataParallelTable. The target is always copied to a CUDA tensor
   self.input = {}
   for i = 1, #sample.input do
      self.input[i] = torch.CudaTensor(1, unpack(sample.input[i]:size():totable()))
      self.input[i]:copy(sample.input[i]):cuda()
   end
   -- self.input = self.input or (self.opt.nGPU == 1
   --    and torch.CudaTensor()
   --    or cutorch.createCudaHostTensor())
   -- self.target = self.target or torch.CudaTensor()

   -- self.input:resize(sample.input:size()):copy(sample.input)
   -- self.target:resize(sample.target:size()):copy(sample.target)
end

function Trainer:learningRate(epoch)
   -- Training schedule
   local decay = 0
   local ratio = 0.1
   if self.opt.dataset == 'imagenet' then
      -- decay = math.floor((epoch - 1) / 40)
      if epoch < 5 then
         -- decay = math.floor((epoch - 1) / 4)
         decay = 0
      else
         decay = 1
      end
   elseif self.opt.dataset == 'cifar10' or self.opt.dataset == 'cifar100' then
      decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
      ratio = 0.2
   end
   -- new LR policy
   return self.opt.LR * math.pow(ratio, decay)
end

return M.Trainer
