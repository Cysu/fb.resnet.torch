require 'nn'

function help()
   print("Test different strategies:")
   print("\t\t-scales required, format like 224,256,288")
   print("\t\t-strategy required, format like 1/2/3 (choose only 1 strategy)")
end

if #arg < 2 then
   help()
   os.exit(0)
end

local scales
local strategy

-- Parse
for i = 1, #arg, 2 do
   if arg[i] == '-scales' then
      scales = arg[i + 1]:split(',')
   elseif arg[i] == '-strategy' then
      strategy = tonumber(arg[i + 1])
   else
      error("No such option: " .. arg[i] .. '!')
   end
end

function computeScore(output, target, nCrops, nScales)
   -- Coputes the top1 and top5 error rate
   local batchSize = output:size(1)

   local _, predictions = output:float():sort(2, true) -- descending

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

local output
local softMax = nn.SoftMax()
local target = torch.load("scores_test/target.t7")

for i = 1, #scales do
   print("- Processing scale-" .. scales[i])
   local scores = torch.load("scores_test/scores_" .. scales[i] .. ".t7") 
   local tmpoutput = torch.Tensor(#scores, 1000):zero()
   if not output then
      output = torch.Tensor(#scores, 1000):zero()
   end
   if strategy == 1 then
      for j = 1, #scores do 
         local pooling = nn.SpatialAveragePooling(scores[j]:size(4), scores[j]:size(3), 1, 1)
         scores[j] = pooling:forward(scores[j])
         scores[j] = scores[j]:view(scores[j]:size(2))
         output[j]:add(scores[j])
         tmpoutput[j]:add(scores[j])
      end
   elseif strategy == 2 then
      for j = 1, #scores do
         local pooling = nn.SpatialAveragePooling(scores[j]:size(4), scores[j]:size(3), 1, 1)
         scores[j] = pooling:forward(scores[j])
         scores[j] = softMax:forward(scores[j])
         scores[j] = scores[j]:view(scores[j]:size(2))
         output[j]:add(scores[j])
         tmpoutput[j]:add(scores[j])
      end
   else
      for j = 1, #scores do
         local pooling = nn.SpatialAveragePooling(scores[j]:size(4), scores[j]:size(3), 1, 1)
         scores[j] = softMax:forward(scores[j])
         scores[j] = pooling:forward(scores[j])
         scores[j] = scores[j]:view(scores[j]:size(2))
         output[j]:add(scores[j])
         tmpoutput[j]:add(scores[j])
      end
   end

   local top1, top5 = computeScore(tmpoutput, target, 1)
   print(("- Processed: [%d]/[%d]: Scale-%d\t\ttop1 %7.3f top5 %7.3f"):format(i, #scales, tonumber(scales[i]), top1, top5))
end

local top1Fin, top5Fin = computeScore(output, target, 1)
print(("- Final:\t\ttop1 %7.3f top5 %7.3f"):format(top1Fin, top5Fin))
