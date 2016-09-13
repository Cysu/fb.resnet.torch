require 'torch'
require 'paths'
require 'nn'
local ffi = require 'ffi'
local matio = require 'matio'

local function parseArgs(arg)
  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Options:')
  cmd:option('-scoresDir', '', 'Path to scores directory')
  cmd:option('-save', '', 'Path to save ensemble result')
  cmd:option('-fileOrder', '', 'Save output according to this file list')
  cmd:text()
  local opt = cmd:parse(arg or {})
  assert(paths.dirp(opt.scoresDir))
  assert(opt.save ~= '')
  assert(paths.filep(opt.fileOrder))
  -- Find all scores*.t7 under this directory
  local f = io.popen('find -L ' .. opt.scoresDir .. ' -iname scores*.t7')
  opt.scoreFiles = {}
  while true do
    local line = f:read('*line')
    if not line then break end
    table.insert(opt.scoreFiles, line)
  end
  -- Read file order
  f = io.open(opt.fileOrder, 'r')
  opt.path2index = {}
  local count = 0
  while true do
    local line = f:read('*line')
    if not line then break end
    count = count + 1
    opt.path2index[line] = count
  end
  return opt
end

local function loadData()
  local info = torch.load('gen/imagenet.t7').val
  local im2cls = {}
  for i = 1, info.imageClass:size(1) do
    im2cls[ffi.string(info.imagePath[i]:data())] = info.imageClass[i]
  end
  return im2cls
end

local function computeScore(output, target)
   local batchSize = output:size(1)
   local _ , predictions = output:float():sort(2, true)
   local correct = predictions:eq(
      target:long():view(batchSize, 1):expandAs(output))
   local top1 = 1.0 - (correct:narrow(2, 1, 1):sum() / batchSize)
   local len = math.min(5, correct:size(2))
   local top5 = 1.0 - (correct:narrow(2, 1, len):sum() / batchSize)
   return top1 * 100, top5 * 100
end


-- Parse command line options
local opt = parseArgs(arg)


-- Ensemble multi-scale multi-crop scores
local ensemble = {}
for i = 1, #opt.scoreFiles do
  local tmp = torch.load(opt.scoreFiles[i])
  local scores = tmp.scores
  local paths = tmp.paths
  assert(scores:size(1) == #paths)
  scores = scores:exp()
  scores:cdiv(scores:sum(3):expandAs(scores))
  scores = scores:mean(2):squeeze()
  if i > 1 then
    for j = 1, scores:size(1) do
      assert(ensemble[paths[j]])
      ensemble[paths[j]]:add(scores[j])
    end
  else
    for j = 1, scores:size(1) do
      ensemble[paths[j]] = scores[j]
    end
  end
end


-- Evaluate result
local im2cls = loadData()
local output, target = {}, {}
for k, v in pairs(ensemble) do
  assert(im2cls[k])
  table.insert(output, v:view(1, v:size(1)))
  table.insert(target, im2cls[k])
end
output = nn.JoinTable(1):forward(output)
target = torch.LongTensor(target)
assert(target:size(1) == output:size(1))

local top1, top5 = computeScore(output, target)
print(string.format('Ensemble result top1: %7.3f  top5: %7.3f', top1, top5))


-- Merge predictions belonging to the same image
local ret = {}
local norm = #opt.scoreFiles
for k, v in pairs(ensemble) do
  v = v:view(1, v:size(1))
  v = torch.div(v, norm)
  k = paths.basename(k, '.JPEG')
  local index = opt.path2index[k]
  if ret[index] then
    assert(torch.abs(ret[index] - v):max() < 1e-5)
  else
    ret[index] = v
  end
end
ret = nn.JoinTable(1):forward(ret)
matio.save(opt.save, ret)

