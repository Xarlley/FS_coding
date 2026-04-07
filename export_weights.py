import torch
import numpy as np
import os

def export_weights_to_bin(pt_file_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading checkpoint from: {pt_file_path}")
    # 加载模型权重 (映射到CPU上处理)
    checkpoint = torch.load(pt_file_path, map_location='cpu')
    
    # 兼容 PyTorch 不同的保存习惯
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'net' in checkpoint:
        state_dict = checkpoint['net']
    else:
        state_dict = checkpoint

    bin_path = os.path.join(output_dir, 'dvsgesture_weights.bin')
    meta_path = os.path.join(output_dir, 'dvsgesture_weights_meta.txt')
    
    offset = 0 # 记录字节偏移量
    
    with open(bin_path, 'wb') as f_bin, open(meta_path, 'w') as f_meta:
        f_meta.write(f"{'Layer_Name':<40} | {'Shape':<25} | {'Size(Floats)':<12} | {'Offset(Bytes)':<15}\n")
        f_meta.write("-" * 100 + "\n")
        
        for name, param in state_dict.items():
            # 我们只需要提取权重和偏置，过滤掉可能存在的冗余状态参数
            if 'weight' in name or 'bias' in name:
                # 转化为标准的单精度浮点数 (float32)，SNN 推理的标配
                np_weight = param.detach().cpu().numpy().astype(np.float32)
                # 强制转换为内存连续数组 (C-Contiguous)，这对 CUDA 指针至关重要
                np_weight = np.ascontiguousarray(np_weight)
                
                bytes_data = np_weight.tobytes()
                size_in_floats = np_weight.size
                bytes_len = len(bytes_data)
                
                # 写入二进制文件
                f_bin.write(bytes_data)
                
                # 记录元数据
                meta_line = f"{name:<40} | {str(list(np_weight.shape)):<25} | {size_in_floats:<12} | {offset:<15}\n"
                f_meta.write(meta_line)
                
                print(f"Exported: {name:<35} | Shape: {np_weight.shape}")
                
                # 累加偏移量
                offset += bytes_len

    print(f"\n✅ 权重已导出至: {bin_path}")
    print(f"✅ 内存布局元数据已导出至: {meta_path}")
    print("👉 提示：请在编写 CUDA 代码时参考 meta.txt 文件来切分显存指针。")

if __name__ == '__main__':
    # 根据 Record.md 记录，你的模型保存在 models 目录下
    pt_file = './models/dvsgesture_fs0.1_tau60_culif.pt'
    
    if not os.path.exists(pt_file):
        print(f"❌ 找不到文件: {pt_file}。请检查路径或确认是否已运行过训练脚本。")
    else:
        export_weights_to_bin(pt_file, './cuda_assets')