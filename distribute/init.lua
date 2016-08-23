require 'cutorch'
local mpi = require 'distribute/mvapich'

local M = {}
local Distributer = torch.class('resnet.Distributer', M)

local function checkType(data)
   assert(torch.type(data) == 'torch.FloatTensor' or
          torch.type(data) == 'torch.CudaTensor')
end

function Distributer:__init()
   self.comm = mpi.comm_world
   self.rank = mpi.rank(self.comm)
   self.size = mpi.size(self.comm)
   assert(self.size > 1, 'Must start more than one MPI processes')
end

function Distributer:rank()
   return self.rank
end

function Distributer:size()
   return self.size
end

function Distributer:isRoot()
   return self.rank == 0
end

function Distributer:bcastFromRoot(data)
   checkType(data)
   cutorch.synchronize()
   mpi.bcast(torch.data(data), data:nElement(), mpi.float, 0, self.comm)
end

function Distributer:averageToRoot(value)
   if torch.type(value) == 'number' then
      -- Put the number in a FloatTensor
      local num = torch.FloatTensor(1):fill(value)
      local sum = torch.FloatTensor(1):zero()
      mpi.reduce(torch.data(num), torch.data(sum), 1,
                 mpi.float, mpi.sum, 0, self.comm)
      return sum[1] / self.size
   else
      checkType(value)
      local prevGpuid = cutorch.getDevice()
      cutorch.setDevice(1) -- average to GPU#1 so that DPT can syncParameters to others
      local sum = torch.type(value) == 'torch.FloatTensor' and torch.FloatTensor()
                                                           or torch.CudaTensor() 
      sum:resize(value:size()):zero()
      cutorch.synchronize()
      -- average from GPU#1 of all the machines
      mpi.reduce(torch.data(value), torch.data(sum), value:nElement(),
                 mpi.float, mpi.sum, 0, self.comm)
      sum:div(self.size)
      if self:isRoot() then
         value:copy(sum)
      end
      cutorch.synchronize()
      cutorch.setDevice(prevGpuid)
   end
end

return M.Distributer()
