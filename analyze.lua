require 'nn';
require 'cudnn';
require 'cunn';

print '-Starting loading...'

local model = torch.load('./checkpoints/model_192.t7')
local rcn = model:get(7)
local table = rcn:get(2)

print '-Extracting weights'
w = {}
local norm = {0, 0, 0, 0}
local filter = {}
for i = 1, 4 do
   w[i] = table:get(i):get(3).weight
   
   local vec = {}
   for a = 1, w[i]:size(1) do
      for b = 1, w[i]:size(2) do
         norm[i] = norm[i] + (w[i][a][b][1][1])^2
         vec[(a - 1) * w[i]:size(2) + b] = w[i][a][b][1][1]
      end
   end
   filter[i] = vec
   norm[i] = math.sqrt(norm[i])
end

for a = 1, 3 do
   for b = a + 1, 4 do
      local val = 0
      for i = 1, 1600 do
         val = val + filter[a][i] * filter[b][i]
      end
      val = val / norm[a] / norm[b]
      print(("%d and %d:"):format(a, b))
      print(("    Cos(%d,%d) = "):format(a, b) .. tostring(val))
      print("    Arg = " .. tostring(math.acos(val) * 180 / math.pi))
   end
end
