require 'nn';
require 'cudnn';
require 'cunn';

local function parse(y, m, output, tab_len)
   if torch.typename(m):find('Convolution') ~= nil then
      -- convolution memory
      output:write(tab_len .. torch.typename(m) .. (' %dx%d  pad:%d/%d stride:%d/%d'):format(m.kW, m.kH, m.padW, m.padH, m.dW, m.dH) .. ' :\n')
      local parameters = 0
      -- kernel weights
      parameters = parameters + m.nInputPlane * m.nOutputPlane * m.kW * m.kH
      -- kernel bias
      if m.bias ~= nil then
         parameters = parameters + m.nOutputPlane 
      end
      local ret = m:forward(y)
      -- layer output
      local layer_out = 1
      for i = 1, ret:dim() do
         layer_out = layer_out * ret:size(i)
      end
      parameters = parameters + layer_out
      parameters = toMiB(parameters)
      output:write(tab_len .. 'Memory: ' .. tostring(parameters))
      return ret, parameters  
   elseif torch.typename(m):find('BatchNormal') ~= nil or torch.typename(m):find('Pooling') ~= nil or torch.typename(m):find('CAddTable') then
      -- BN memory or Pooling memory
      output:write(tab_len .. torch.typename(m) .. ' :\n')
      local parameters = 1
      local ret = m:forward(y)
      for i = 1, ret:dim() do
         parameters = parameters * ret:size(i)
      end
      parameters = toMib(parameters + 2)
      output:write(tab_len .. 'Memory: ' .. tostring(parameters))
      return ret, parameters
   elseif torch.typename(m):find('Linear') ~= nil then
      output:write(tab_len .. torch.typename(m) .. ' :\n')
      local parameters = m.nInputPlane * m.nOutputPlane
      if m.bias ~= nil then
         parameters = parameters + m.nOutputPlane
      end
      local ret = m:forward(y)
      local layer_out = 1
      for i = 1, ret:dim() do
         layer_out = layer_out * ret:size(i)
      end
      parameters = toMiB(parameters + layer_out)
      output:write(tab_len .. 'Memory: ' .. tostring(parameters))
   elseif torch.typename(m):find('Sequential') ~= nil or torch.typename(m):find('ConcatTable') then
      output:write(tab_len .. torch.typename(m) .. ' :\n')  
      local sz = m:size()
      local mem = 0
      local inc = 0
      for i = 1, sz do
         y, inc = parse(y, m:get(i), output, tab_len .. '\t')
         mem = mem + inc
      end
      output:write(tab_len .. 'Memory: ' .. tostring(mem))
      return y, mem
   else
      output:write(tab_len .. torch.typename(m) .. ' : \n')
      y = m:forward(y)
      return y, 0
   end
end

local model = torch.load(arg[1])

local shape = tonumber(arg[2])
local batch = tonumber(arg[3])

output = io.open('Memory_Usage.log', 'w')
local y = torch.Tensor(batch, channel, shape, shape)

local tab = ''
parse(y, model, output, tab)

output:close()
