# How to get here

## Environment

16GB的显卡在启动了图形界面必须的X和一些桌面组件后，显存差1个多GB才能放下本项目（默认batch）。开启显存的动态扩展分配，这能有效缓解由于不断分配释放张量导致的显存碎片化。这使项目可以正常训练。

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

没有按照作者的推荐安装环境，作者的环境太老了，凑齐很难。新环境导出到了`environment.yml`。由于使用了新环境，需要修改一点点代码:
1. 在`spkjelly/src/time_encoding.py`将`gaussian`函数修改一下调用方式。
2. 在`spkjelly/train.py`将np的数据结构调整一下。

作者的环境是2卡，用单卡复现的话，在`spkjelly/train.py`和`spkjelly/test.py`将GPU调整为仅0号GPU。

## Train and Test

训练。

```bash
cd spkjelly

python train.py \
--path_to_yaml '../configs/culif_dvsgesture.yaml' \
--log-interval 5 \
--lr 1e-4 \
--batch-size 16 \
--test-batch-size 32 \
--epochs 70 \
--T_empty 40 \
--T 120 \
--dt 10 \
--loss_mode 'first_time' \
--FS 0.1 4 300 \
--neuron1 5. 5. 0.5 \
--neuron2 60. 60. 1.0 \
--treg 0.01 0.02
```

测试。

```bash
python test.py \
--path_to_yaml '../configs/culif_dvsgesture.yaml' \
--test-batch-size 6 \
--T 250 \
--T_empty 0 \
--load_pt_file '../models/dvsgesture_fs0.1_tau60_culif.pt' \
--best_model
```