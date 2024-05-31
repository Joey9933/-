## 任务1

* 切换工作路径为当前文件夹,并将数据文件夹`CUB_200_2011`和预训练参数文件`resnet18-5c106cde.pth`保存在该目录下;
* 执行`data.py`,将在数据文件夹`./CUB_200_2011/`下新建`dataset/`文件夹,内含train数据和val数据(即test数据);
* 执行`python train.py --epoch <EPOCH> --batchsize <BATCHSIZE> --lr <LR> --mode <USE_PRETRAIN>`训练模型
* 将模型参数保存在目录`./checkpoints/`下，执行 `python train.py --batchsize <BATCHSIZE> --ckp <checkpoins.pth>`测试模型

## 任务2