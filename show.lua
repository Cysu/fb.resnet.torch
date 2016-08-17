require 'models/SequentialDropout'
require 'nn'
require 'cunn'
require 'cudnn'

a = torch.load('pretrained/RFCN.t7')

print(a)
