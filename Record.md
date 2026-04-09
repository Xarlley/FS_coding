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

## CUDA baseline

`snn_inference.cu`是一个与python实现完全等价的CUDA实现，对训练集的推理准确率达到99.91%。

需要使用`export_dataset.py`和`export_weights.py`导出数据集和权重，以供CUDA程序加载使用。

## First-Spike -> Time-to-First-Spike

原项目是一个FS编码的脉冲神经网络，但我想要TTFS编码。

`spkjelly/src/models.py`, `spkjelly/src/neuron_ex.py`, `configs/culif_dvsgesture.yaml`经过修改，现在已经不是原仓库的TS编码，变成了严格的TTFS编码。TTFS编码要求所有神经元在整个时间窗口上仅能发放一次脉冲。

经过70 epoch训练，严格TTFS版本的SNN在训练集上的准确率达到94.81%（pytorch实现），在测试集上达到84.09%(pytorch实现)。

再次使用`export_weights.py`导出权重。

使用CUDA实现严格TTFS版的脉冲神经网络，即`snn_inference_TTFS.cu`，准确率达到96.29%（cuda实现）。

## CUDA TTFS

现在已经可以实现CUDA VD-TTFS算法的正确推理了。

`extract_single_sample.py`实现了仅提取出数据集中单个样例和label。

`snn_inference_single.cu`实现了仅用VD-TTFS推理单个样例。

`snn_inference_TTFS_layer.cu`实现了仅将step-by-step SNN改为了层优先step-by-step。

`snn_inference_TTFS_layer_prefix.cu`实现了VD-TTFS。

`snn_inference_TTFS_layer_prefix_statistics.cu`在实现VD-TTFS的基础上统计出论文所需的统计值。