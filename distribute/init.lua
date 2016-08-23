require 'cutorch'
local mpi = require 'distribute/mvapich'

local M = {}
local Distributer = torch.class('resnet.Distributer', M)

function Distributer:__init()
   self.comm = mpi.comm_world
   self.rank = mpi.rank(self.comm)
   self.size = mpi.size(self.comm)
   assert(self.size > 1, 'Must start more than one MPI processes')
   self.localRootDevice = 1 -- DataParallelTable uses device 1 .. nGPU and 1 is the root by default
end

function Distributer:getRank()
   return self.rank
end

function Distributer:getSize()
   return self.size
end

function Distributer:isRoot()
   return self.rank == 0
end

function Distributer:bcastFromRoot(data)
   if torch.type(data) == 'torch.FloatTensor' then
      mpi.bcast(torch.data(data), data:nElement(), mpi.float, 0, self.comm)
   elseif torch.type(data) == 'torch.CudaTensor' then
      cutorch.synchronize()
      local cpuData = data:float()
      mpi.bcast(torch.data(cpuData), data:nElement(), mpi.float, 0, self.comm)
      data:copy(cpuData)
      cutorch.synchronize()
   else
      error('Only FloatTensor or CudaTensor can be broadcasted')
   end
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
      assert(torch.type(value) == 'torch.CudaTensor' and value:dim() == 1,
             'Only 1D CudaTensor can be reduced')

      -- average from GPU#1 of all the machines
      cutorch.synchronize()
      local cpuData = value:float()
      mpi.reduce(mpi.in_place, torch.data(cpuData), cpuData:nElement(),
                 mpi.float, mpi.sum, 0, self.comm)

      -- copy back to GPU
      value:copy(cpuData)
      value:div(self.size)
      cutorch.synchronize()
   end
end

function Distributer:finalize()
   mpi.finalize()
end

return M.Distributer()
