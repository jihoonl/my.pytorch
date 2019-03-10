# my.pytorch

My codebase to train and test various networks

### Mnist

* Resnet
* Mnist CNN
* FC 

## Appendix

### Leaderboard

Provide leaderboard with top1 config and model symlink

```
#Rank : Data                          ,   Test Acc   Test Loss   Train Acc  Train Loss
   1  : ../Mar09-2054/final.pth       ,   0.993700,   0.025870,   0.999500,   0.001450
   2  : ../Mar09-2025/final.pth       ,   0.993300,   0.024709,   0.999800,   0.000804
   3  : ../Mar09-1630/final.pth       ,   0.993200,   0.023206,   1.000000,   0.001310
   ...
```

### Includable YAML

```
# configs/mnist_cnn.yaml

# MNIST dataset
DATASET: !include dataset/mnist.yaml
MODEL: !include model/mnist_base.yaml

TRAIN:
  EPOCH: 50
  BATCH_SIZE: 512
  OPTIMIZER:
    PARAM:
      lr: 0.05
      momentum: 0.9
      weight_decay: 0.0001
  LR_SCHEDULER: !include lr_scheduler/sgdr.yaml
```
