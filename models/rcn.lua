local nn = require 'nn'
require 'cunn'
require 'cudnn'

local Convolution = cudnn.SpatialConvolution
local Max = nn.SpatialMaxPooling
local ReLU = cudnn.ReLU 

local function createModel(opt, preModel)
   -- remove the avg_pooling and fc part of ResNet-34
   for i = 1,3 do
      preModel:remove()
   end
   
   -- add rcn after the feature maps
   local function rcn(nInputPlane, nOutputPlane, multiFactor, w, h)
      local s = nn.Sequential()
      -- reduce dimension
      local nMiddle = 1024
      local conv0 = Convolution(nInputPlane, nMiddle, 1, 1)
      -- initialize
      conv0.weight:normal(0, 0.01)
      conv0.bias:zero()
      s:add(conv0)
      s:add(ReLU(true))

      local table = nn.ConcatTable()
      local len3 = math.ceil(w / multiFactor)
      local len4 = math.ceil(h / multiFactor)
      -- build the n-part layer
      for i = 0, multiFactor - 1 do
         for j = 0, multiFactor - 1 do
            local part = nn.Sequential()
            local offset3 = math.floor(i * w / multiFactor) + 1
            if offset3 + len3 > w + 1 then
               offset3 = w - len3 + 1
            end
            local offset4 = math.floor(j * h / multiFactor) + 1
            if offset4 + len4 > h + 1 then
               offset4 = w - len4 + 1
            end
            part:add(nn.Narrow(3, offset3, len3))
            part:add(nn.Narrow(4, offset4, len4))
            -- 1x1 Convolution
            local conv1 = Convolution(nMiddle, nOutputPlane, 1, 1)
            -- initialize
            conv1.weight:normal(0, 0.01)
            conv1.bias:zero()
            part:add(conv1)
            part:add(Max(len3, len4, 1, 1))
            table:add(part)
         end
      end

      s:add(table)
      s:add(nn.CAddTable())
      s:add(nn.View(nOutputPlane):setNumInputDims(3))
      return s
   end

   local cfg = {
      [18]  = {{2, 2, 2, 2}, 512, basicblock},
      [34]  = {{3, 4, 6, 3}, 512, basicblock},
      [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
      [101] = {{3, 4, 23, 3}, 2048, bottleneck},
      [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      [200] = {{3, 24, 36, 3}, 2048, bottleneck},
   }

   preModel:add(rcn(cfg[opt.depth][2], 1000, opt.multiFactor, 7, 7))
   preModel:cuda()
   return preModel
 
end

return createModel
