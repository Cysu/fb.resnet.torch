local URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz'

local M = {}

local function convertCifar100ToTensor(file)
   local m = torch.DiskFile(file, 'r'):binary()
   m:seekEnd()
   local length = m:position() - 1
   local nSamples = length / 3074 -- 1 coarse-label byte, 1 fine-label byte, 3072 pixel bytes

   assert(nSamples == math.floor(nSamples), 'expecting numSamples to be an exact integer')
   m:seek(1)

   local coarse = torch.ByteTensor(nSamples)
   local fine = torch.ByteTensor(nSamples)
   local data = torch.ByteTensor(nSamples, 3, 32, 32)
   for i = 1, nSamples do
      coarse[i] = m:readByte()
      fine[i] = m:readByte()
      local store = m:readByte(3072)
      data[i]:copy(torch.ByteTensor(store))
   end

   -- This is *very* important. The downloaded files have labels 0-9, which do
   -- not work with CrossEntropyCriterion
   fine:add(1)

   return {
      data = data,
      labels = fine,
      coarse_labels = coarse,
   }
end

function M.exec(opt, cacheFile)
   print("=> Downloading CIFAR-100 dataset from " .. URL)
   local ok = os.execute('curl ' .. URL .. ' | tar xz -C gen/')
   assert(ok == true or ok == 0, 'error downloading CIFAR-100')

   print(" | loading dataset")
   local trainData = convertCifar100ToTensor('gen/cifar-100-binary/train.bin')
   local testData = convertCifar100ToTensor('gen/cifar-100-binary/test.bin')

   print(" | saving CIFAR-100 dataset to " .. cacheFile)
   torch.save(cacheFile, {
      train = trainData,
      val = testData,
   })
end

return M
