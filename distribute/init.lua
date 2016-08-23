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
   self.buf = torch.CudaTensor()
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
   assert(torch.type(data) == 'torch.CudaTensor' or
          torch.type(data) == 'torch.FloatTensor',
          'Only FloatTensor or CudaTensor can be broadcasted')
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
      assert(torch.type(value) == 'torch.CudaTensor' and value:dim() == 1,
             'Only 1D CudaTensor can be reduced')
      local prevGpuid = cutorch.getDevice()
      -- set device to local root so that DPT can syncParameters to other local GPUs
      cutorch.setDevice(self.localRootDevice)

      -- resize the buffer if necessary and set it to zero
      if self.buf:nElement() < value:nElement() then
         self.buf:resize(value:size())
      end
      self.buf:zero()
      cutorch.synchronize()

      -- average from GPU#1 of all the machines
      mpi.reduce(torch.data(value), torch.data(self.buf), value:nElement(),
                 mpi.float, mpi.sum, 0, self.comm)
      self.buf:div(self.size)

      -- copy back to the root input
      if self:isRoot() then
         value:copy(self.buf)
      end
      cutorch.synchronize()
      cutorch.setDevice(prevGpuid)
   end
end

function Distributer:finalize()
   mpi.finalize()
end

return M.Distributer()
