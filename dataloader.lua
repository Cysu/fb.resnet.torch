--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Multi-threaded data loader
--

local datasets = require 'datasets/init'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')

local M = {}
local DataLoader = torch.class('resnet.DataLoader', M)

function DataLoader.create(opt, distributer)
   -- The train and val loader
   local loaders = {}

   for i, split in ipairs{'train', 'val'} do
      local dataset = datasets.create(opt, split)
      loaders[i] = M.DataLoader(dataset, opt, split, distributer)
   end

   return table.unpack(loaders)
end

function DataLoader:__init(dataset, opt, split, distributer)
   -- TODO: Do we need the seed to be different on different machines?
   local manualSeed = opt.manualSeed
   local function init()
      require('datasets/' .. opt.dataset)
   end
   local function main(idx)
      if manualSeed ~= 0 then
         torch.manualSeed(manualSeed + idx)
      end
      torch.setnumthreads(1)
      _G.dataset = dataset
      _G.preprocess = dataset:preprocess()
      return dataset:size()
   end

   local threads, sizes = Threads(opt.nThreads, init, main)
   self.nCrops = (split == 'val' and opt.tenCrop) and 10 or 1
   self.threads = threads
   self.distributer = distributer
   self.__size = sizes[1][1] -- total size of the dataset
   self.__localSize = math.ceil(self.__size / distributer:getSize()) -- total size on each machine
   self.batchSize = math.floor(opt.batchSize / self.nCrops)
end

function DataLoader:size()
   return math.ceil(self.__localSize / self.batchSize)
end

function DataLoader:run()
   local threads = self.threads
   local size, localSize, batchSize = self.__size, self.__localSize, self.batchSize
   local perm = torch.randperm(size):float()
   self.distributer:bcastFromRoot(perm)

   local idx, sample = 1, nil
   if self.distributer:getRank() == self.distributer:getSize() - 1 then
      -- the last machine may overlap with the second last one to keep localSize equal
      idx = size - localSize + 1
      assert(idx > 0)
   else
      idx = self.distributer:getRank() * localSize + 1
      size = idx + localSize - 1
      assert(size <= self.__size)
   end

   local function enqueue()
      while idx <= size and threads:acceptsjob() do
         local indices = perm:narrow(1, idx, math.min(batchSize, size - idx + 1))
         threads:addjob(
            function(indices, nCrops)
               local sz = indices:size(1)
               local batch, imageSize
               local target = torch.IntTensor(sz)
               for i, idx in ipairs(indices:totable()) do
                  local sample = _G.dataset:get(idx)
                  local input = _G.preprocess(sample.input)
                  if not batch then
                     imageSize = input:size():totable()
                     if nCrops > 1 then table.remove(imageSize, 1) end
                     batch = torch.FloatTensor(sz, nCrops, table.unpack(imageSize))
                  end
                  batch[i]:copy(input)
                  target[i] = sample.target
               end
               collectgarbage()
               return {
                  input = batch:view(sz * nCrops, table.unpack(imageSize)),
                  target = target,
               }
            end,
            function(_sample_)
               sample = _sample_
            end,
            indices,
            self.nCrops
         )
         idx = idx + batchSize
      end
   end

   local n = 0
   local function loop()
      enqueue()
      if not threads:hasjob() then
         return nil
      end
      threads:dojob()
      if threads:haserror() then
         threads:synchronize()
      end
      enqueue()
      n = n + 1
      return n, sample
   end

   return loop
end

return M.DataLoader
