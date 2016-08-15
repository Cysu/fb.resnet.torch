local nn = require 'nn'
require 'cunn'
require 'cudnn'

local Convolution = cudnn.SpatialConvolution
local Max = nn.SpatialMaxPooling
local ReLU = cudnn.ReLU 
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(opt, preModel)
   local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
   end
   
   -- add rcn after the feature maps
   local function rcn(nInputPlane, nOutputPlane, multiFactor, w, h)
      local s = nn.Sequential()
      local sOutside = nn.Sequential()
      s:add(SBatchNorm(nInputPlane))
      s:add(ReLU(true))
      -- reduce dimension
      -- local nMiddle = nInputPlane / 2 
      -- local conv0 = Convolution(nInputPlane, nMiddle, 1, 1)
      -- initialize
      -- conv0.weight:normal(0, 0.01)
      -- conv0.bias:zero()
      -- s:add(conv0)
      -- s:add(ReLU(true))

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
            local conv1 = Convolution(nInputPlane, nOutputPlane, 1, 1)
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
      -- Balance the RFCN output and fc-layer output
      local mul = nn.Mul()
      mul.weight:fill(0.25)
      sOutside:add(s)
      sOutside:add(mul)
      sOutside:add(nn.View(nOutputPlane):setNumInputDims(3))
      return sOutside
   end

   local cfg = {
      [18]  = {{2, 2, 2, 2}, 512, basicblock},
      [34]  = {{3, 4, 6, 3}, 512, basicblock},
      [50]  = {{3, 4, 6, 3}, 2048, bottleneck},
      [101] = {{3, 4, 23, 3}, 2048, bottleneck},
      [152] = {{3, 8, 36, 3}, 2048, bottleneck},
      [200] = {{3, 24, 36, 3}, 2048, bottleneck},
   }

   local sz = preModel:size()
   local RFCN = nn.ConcatTable()
   local r = nn.Sequential()
   for i = 3, 0, -1 do
      r:add(preModel:get(sz - i))
   end
   for i = 0, 3 do
      preModel:remove()
   end

   local s = rcn(cfg[opt.depth][2] / 2, 1000, opt.multiFactor, 14, 14)

   RFCN:add(r)
   RFCN:add(s)

   preModel:add(RFCN)
   preModel:add(nn.CAddTable())

   preModel:cuda()

   return preModel
 
end

return createModel
