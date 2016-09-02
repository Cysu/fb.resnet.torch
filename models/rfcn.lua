local nn = require 'nn'
require 'cunn'
require 'cudnn'

local Convolution = cudnn.SpatialConvolution
local Max = nn.SpatialMaxPooling
local ReLU = cudnn.ReLU 
local SBatchNorm = nn.SpatialBatchNormalization

local function createModel(depth, multiFactor, preModel)
   local function ShareGradInput(module, key)
      assert(key)
      module.__shareGradInputKey = key
      return module
   end
   
   -- add rcn after the feature maps
   local function rcn(nInputPlane, nOutputPlane, multiFactor, w, h)
      local s = nn.Sequential()
      s:add(SBatchNorm(nInputPlane))
      s:add(ReLU(true))
      -- reduce dimension
      local nMiddle = nInputPlane / 2 
      local conv0 = Convolution(nInputPlane, nMiddle, 1, 1)
      -- initialize
      conv0.weight:normal(0, 0.001)
      conv0.bias:zero()
      s:add(conv0)
      s:add(SBatchNorm(nMiddle))
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
            conv1.weight:normal(0, 0.001)
            conv1.bias:zero()
            part:add(conv1)
            part:add(Max(len3, len4, 1, 1))
            table:add(part)
         end
      end

      s:add(table)
      s:add(nn.CAddTable())
      -- Balance the RFCN output and fc-layer output
      local mul = nn.MulConstant(0.25, true)
      s:add(mul)
      s:add(nn.View(nOutputPlane):setNumInputDims(3))

      -- Mark R-FCN layers to avoid conflicts when share gradient.
      s:apply(function(m)
         m.rfcn = true 
      end)

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

   local sz = preModel:size()
   local RFCN = nn.ConcatTable()
   local r = nn.Sequential()

   -- preModel:apply(function(m)
   --    m.updateParameters = function(self, lr)
   --       -- Fix learning rate of the original model
   --       lr = 0.0001
   --       local params, gradParams = self:parameters()
   --       if params then
   --          for i = 1, #params do
   --             params[i]:add(-lr, gradParams[i])
   --          end
   --       end
   --    end
   -- end)

   -- find the inject point.
   local pos = sz 
   local last = true
   while last or torch.type(preModel:get(pos)) ~= 'nn.Sequential' do
      if torch.type(preModel:get(pos)) == 'nn.Sequential' then
         last = false
      end
      pos = pos - 1
   end

   for i = pos + 1, sz do
      r:add(preModel:get(i))
   end
   for i = pos + 1, sz do
      preModel:remove()
   end

   local s = rcn(cfg[opt.depth] and cfg[opt.depth][2] / 2 or 1024, 1000, opt.multiFactor, 14, 14)

   RFCN:add(r)
   RFCN:add(s)

   preModel:add(RFCN)
   preModel:add(nn.CAddTable())

   preModel:cuda()

   return preModel
 
end

return createModel
