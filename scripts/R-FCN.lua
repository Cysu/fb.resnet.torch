function help()
   print(
   "th R-FCN.lua [options]\n" .. 
   "\t-model:\trequired, saved file of the model to modify.\n" ..
   "\t-depth:\trequired, saved model's depth.\n" ..
   "\t-data:\trequired, path to dataset.\n" ..
   "\t-devices:\toptional, visible devices to the model(0,1,2,3,4,5,6,7)\n" ..
   "\t-multiFactor:\toptional, number of parts on each edge(2)\n" .. 
   "\t-nGPU:\t\toptional, numbers of available GPUs(8)\n" ..
   "\t-logFile:\toptional, file to log training data(nil)\n" ..
   "\t-newFile:\toptional, file to save model in the middle stage(pretrained/RFCN.t7)\n" ..
   "\t-batchSize:\toptional, batchSize of fine-tuning(256)\n" ..
   "\t-nEpochs:\toptional, epochs of batchSize(15)\n" ..
   "\t-LR:\t\toptional, learning rate of fine-tuning(0.001)\n" ..
   "\nThen, bash scripts/fine_tune.sh\n"
   )
end

if #arg <= 1 then
   help()
else
   require 'cunn';
   require 'cudnn';
   require 'nn';

   opt = {}
   opt.devices = '0,1,2,3,4,5,6,7'
   opt.multiFactor = '1'
   opt.nGPU = '8'
   opt.logFile = nil
   opt.newFile = 'pretrained/RFCN.t7'
   opt.batchSize = '256'
   opt.nEpochs = '15'
   opt.LR = '0.001'
   -- parse parameters
   for i = 1, #arg, 2 do
      local key = arg[i]
      local val = arg[i + 1]
      if key == '-model' then
         opt.model = val
      elseif key == '-depth' then
         opt.depth = tonumber(val)
      elseif key == '-data' then
         opt.data = val
      elseif key == '-devices' then
         opt.devices = val
      elseif key == '-multiFactor' then
         opt.multiFactor = tonumber(val)
      elseif key == '-nGPU' then
         opt.nGPU = val
      elseif key == '-logFile' then
         opt.logFile = val
      elseif key == '-newFile' then
         opt.newFile = val
      elseif key == '-batchSize' then
         opt.batchSize = val
      elseif key == '-nEpochs' then
         opt.nEpochs = val
      elseif key == 'LR' then
         opt.LR = val
      else
         assert(false, "illegal paramter: " .. key .."!\n")
      end
   end

   local preModel = torch.load(opt.model)
   print "- Pretrained model loaded."
   print "- Starting to add RFCN..."
   local model = require("models/rfcn")(opt.depth, opt.multiFactor, preModel)
   print "- RFCN added."

   torch.save(opt.newFile, model)
   print "- RFCN model saved."

   -- Fine-tune
   cmd = "CUDA_VISIBLE_DEVICES=".. opt.devices .. " OMP_NUM_THREADS=1 th main.lua " ..
   "-data " .. opt.data .. " -nGPU " .. opt.nGPU .. " -nThreads 16 -batchSize " ..
   opt.batchSize .. " -depth " .. tostring(opt.depth) .. " -dropout 0.5" ..
   " -nEpochs " .. opt.nEpochs .. " -LR " .. opt.LR ..
   " -retrain " .. opt.newFile .. ".t7"
   if opt.logFile then
      cmd = cmd .. " -logFile " .. opt.logFile
   end

   file = io.open('scripts/fine_tune.sh', 'w')
   file:write(cmd)
   file:close()

   print "- Ready to fine-tune."
end
