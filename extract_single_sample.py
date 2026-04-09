import numpy as np
import os
import argparse

def extract_sample(idx, data_path, label_path, out_dir):
    # 根据你的 CUDA 代码中定义的维度
    T_TOTAL = 160
    C = 2
    H = 32
    W = 32
    NUM_CLASSES = 10

    # 计算单个样本的字节数 (float32 和 int32 都是 4 bytes)
    data_size_floats = C * H * W * T_TOTAL
    data_bytes_per_sample = data_size_floats * 4
    label_bytes_per_sample = NUM_CLASSES * 4

    os.makedirs(out_dir, exist_ok=True)
    out_data_path = os.path.join(out_dir, 'single_data.bin')
    out_label_path = os.path.join(out_dir, 'single_label.bin')

    try:
        # 提取数据
        with open(data_path, 'rb') as f:
            f.seek(idx * data_bytes_per_sample)
            single_data = f.read(data_bytes_per_sample)
            with open(out_data_path, 'wb') as f_out:
                f_out.write(single_data)

        # 提取标签
        with open(label_path, 'rb') as f:
            f.seek(idx * label_bytes_per_sample)
            single_label = f.read(label_bytes_per_sample)
            with open(out_label_path, 'wb') as f_out:
                f_out.write(single_label)

        print(f"✅ 成功提取样本 [{idx}]")
        print(f"  数据保存至: {out_data_path}")
        print(f"  标签保存至: {out_label_path}")
    except Exception as e:
        print(f"❌ 提取失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="提取单个二进制样本供CUDA推理验证")
    parser.add_argument('--index', type=int, default=0, help="要提取的样本序号 (例如 0 到 1077)")
    args = parser.parse_args()

    # 假设你的原始测试数据叫 test_data.bin，请根据实际文件名修改
    # 注意：这里请填入你测试集的 bin 文件路径，消融实验通常在测试集上做
    data_file = 'cuda_assets/train_data.bin' 
    label_file = 'cuda_assets/train_labels.bin'
    
    extract_sample(args.index, data_file, label_file, 'cuda_assets')