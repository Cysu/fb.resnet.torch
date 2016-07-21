--
--  Copyright (c) 2016, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'torch'
require 'paths'
require 'optim'
require 'nn'
local DataLoader = require 'dataloader'
local models = require 'models/init'
local Trainer = require 'train'
local opts = require 'opts'
local checkpoints = require 'checkpoints'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(1)

local opt = opts.parse(arg)
torch.manualSeed(opt.manualSeed)
cutorch.manualSeedAll(opt.manualSeed)

-- Load previous checkpoint, if it exists
local checkpoint, optimState = checkpoints.latest(opt)

-- Create model
local model, criterion = models.setup(opt, checkpoint)

-- Data loading
local trainLoader, valLoader = DataLoader.create(opt)

-- The trainer handles the training loop and evaluation on validation set
local trainer = Trainer(model, criterion, opt, optimState)

if opt.testOnly then
   local top1Err, top5Err = trainer:test(0, valLoader)
   print(string.format(' * Results top1: %6.3f  top5: %6.3f', top1Err, top5Err))
   return
end

local startEpoch = checkpoint and checkpoint.epoch + 1 or opt.epochNumber
local bestFirstTop1 = math.huge
local bestSecondTop1 = math.huge
for epoch = startEpoch, opt.nEpochs do
   -- Train for a single epoch
   local trainFirstTop1, trainSecondTop1, trainLoss = trainer:train(epoch, trainLoader)

   -- Run model on validation set
   local testFirstTop1, testSecondTop1 = trainer:test(epoch, valLoader)

   local bestModel = false
   if testSecondTop1 < bestSecondTop1 then
      bestModel = true
      bestFirstTop1 = testFirstTop1
      bestSecondTop1 = testSecondTop1
      print(' * Best model ', testFirstTop1, testSecondTop1)
   end

   checkpoints.save(epoch, model, trainer.optimState, bestModel, opt)
end

print(string.format(' * Finished first top1: %6.3f  second top1: %6.3f', bestFirstTop1, bestSecondTop1))
