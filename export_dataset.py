import torch
import numpy as np
import os
import sys

# 将当前项目根目录加入模块搜索路径，以便导入原仓库的代码
sys.path.append(os.path.abspath('.'))

from data_io.dvs_gesture_reader import create_datasets

def export_dataset_via_dataloader(hdf5_path, group, output_dir, prefix):
    if not os.path.exists(hdf5_path):
        print(f"❌ 找不到文件: {hdf5_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n🚀 开始调用原仓库逻辑处理 {group} 数据集: {hdf5_path}")
    
    # 根据 culif_dvsgesture.yaml 配置参数实例化原仓库的 Dataset
    # T=120, T_empty=40, dt=10ms (10000 us)
    dataset = create_datasets(
        filename=hdf5_path,
        group=group,
        ds=[4, 4], # 空间降采样 128x128 -> 32x32
        dt=10 * 1000, 
        chunk_size=120,
        empty_size=40,
        num_classes=10,
        target_transform='onehot' # 触发 toOneHot 转换
    )
    
    num_samples = len(dataset)
    print(f"📦 成功加载数据集，总样本数: {num_samples}")
    
    all_data = []
    all_labels = []
    
    for i in range(num_samples):
        # 触发原仓库的 __getitem__，执行复杂的事件累加计算
        data, label = dataset[i]
        
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        if isinstance(label, torch.Tensor):
            label = label.numpy()
            
        all_data.append(data)
        all_labels.append(label)
        
        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            print(f"⏳ 正在转换... {i + 1}/{num_samples}")
            
    print("🛠️ 正在合并张量并强制连续化内存布局...")
    
    # 堆叠所有样本，转为 float32 (推理用的浮点数) 和 int32 (标签)
    data_np = np.stack(all_data).astype(np.float32)
    labels_np = np.stack(all_labels).astype(np.int32)
    
    # 强制连续内存 (C-Contiguous)，这对 CUDA 的指针读取极其重要
    data_np = np.ascontiguousarray(data_np)
    labels_np = np.ascontiguousarray(labels_np)
    
    print(f"\n📊 最终数据 Shape: {data_np.shape}, dtype: {data_np.dtype}")
    print(f"🏷️ 最终标签 Shape: {labels_np.shape}, dtype: {labels_np.dtype}")
    
    data_bin_path = os.path.join(output_dir, f'{prefix}_data.bin')
    label_bin_path = os.path.join(output_dir, f'{prefix}_labels.bin')
    
    with open(data_bin_path, 'wb') as f:
        f.write(data_np.tobytes())
        
    with open(label_bin_path, 'wb') as f:
        f.write(labels_np.tobytes())
        
    print(f"\n✅ 数据已导出至: {data_bin_path} ({data_np.nbytes / 1024 / 1024:.2f} MB)")
    print(f"✅ 标签已导出至: {label_bin_path} ({labels_np.nbytes / 1024 / 1024:.2f} MB)")

if __name__ == '__main__':
    output_dir = './cuda_assets'
    # 你的文件路径
    train_hdf5 = './datasets/DVSGesture/DVS-Gesture-train10.hdf5'
    export_dataset_via_dataloader(train_hdf5, 'train', output_dir, 'train')