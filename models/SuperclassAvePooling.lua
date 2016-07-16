nn = require 'nn'
local SuperclassAvePooling, parent = torch.class('nn.SuperclassAvePooling', 'nn.Module')

function SuperclassAvePooling:__init(idxToSuperIdx)
   parent.__init(self)
   idxToSuperIdx = torch.IntTensor(idxToSuperIdx)
   local inputSize = idxToSuperIdx:size(1)
   local outputSize = idxToSuperIdx:max()
   self.w = torch.Tensor(outputSize, inputSize):zero()
   for i = 1, inputSize do
      self.w[idxToSuperIdx[i]][i] = 1.
   end
   self.w:cdiv(self.w:sum(2):expandAs(self.w))
end

function SuperclassAvePooling:updateOutput(input)
   if input:dim() == 1 then
      self.output:resize(self.w:size(1))
      self.output:addmv(1, self.w, input)
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.w:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.output:addmm(0, self.output, 1, input, self.w:t())
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function SuperclassAvePooling:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         self.gradInput:addmv(0, 1, self.w:t(), gradOutput)
      elseif input:dim() == 2 then
         self.gradInput:addmm(0, 1, gradOutput, self.w)
      end

      return self.gradInput
   end
end

function SuperclassAvePooling:accGradParameters(input, gradOutput, scale)
end

-- we do not need to accumulate parameters when sharing
SuperclassAvePooling.sharedAccUpdateGradParameters = SuperclassAvePooling.accUpdateGradParameters

function SuperclassAvePooling:clearState()
   return parent.clearState(self)
end

function SuperclassAvePooling:__tostring__()
  return torch.type(self) ..
      string.format('(%d -> %d)', self.w:size(2), self.w:size(1))
end