CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 th main.lua -data /DATA/xiaotong/ilsvrc/datasets/ -nGPU 8 -nThreads 16 -batchSize 256 -depth 269 -dropout 0.5 -nEpochs 15 -LR 0.001 -retrain pretrained/RFCN.t7 -shareGradInput true -logFile fine_tune

