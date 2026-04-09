#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(EXIT_FAILURE); \
    } \
}

#define GET_BLOCKS(total) ((total + 255) / 256)

// ==========================================================
// TTFS-SNN 超参数与配置
// ==========================================================
const int NUM_SAMPLES = 1078;
const int T_TOTAL = 160; 
const int NUM_CLASSES = 10;

const float decay_syn_conv = exp(-1.0f / 5.0f);
const float decay_syn_fc = exp(-1.0f / 60.0f);

const float v_th_conv = 5.0f;
const float v_th_fc = 10.0f;

// ==========================================================
// 融合了 CuIF 时间步迭代的优化算子
// ==========================================================

// --- Layer 1: 从原始稠密数据提取并运行 CuIF ---
__global__ void conv1_fused_kernel(const float* all_data, float* out_fire_times, 
                                   const float* weight, int N, float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * 64 * 32 * 32;
    if (idx >= total) return;
    
    int n = idx / (64 * 32 * 32);
    int rem = idx % (64 * 32 * 32);
    int c_out = rem / (32 * 32);
    int y = (rem / 32) % 32;
    int x = rem % 32;
    
    // 在寄存器/Local Memory中预先累加整个时间窗口的输入
    float in_val[160] = {0.0f};
    
    for (int c_in = 0; c_in < 2; ++c_in) {
        for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
                int in_y = y + ky - 1;
                int in_x = x + kx - 1;
                if (in_y >= 0 && in_y < 32 && in_x >= 0 && in_x < 32) {
                    int w_idx = ((c_out * 2 + c_in) * 3 + ky) * 3 + kx;
                    float w = weight[w_idx];
                    int in_idx = ((n * 2 + c_in) * 32 + in_y) * 32 + in_x;
                    const float* data_ptr = &all_data[in_idx * 160];
                    for (int t = 0; t < 160; ++t) {
                        in_val[t] += w * data_ptr[t];
                    }
                }
            }
        }
    }
    
    // 在线程内部独立运行 CuIF 动力学
    float I = 0, V = 0;
    float fire_time = 9999.0f;
    for (int t = 0; t < 160; ++t) {
        I = I * decay + in_val[t];
        V = V + I;
        if (V >= th) {
            fire_time = (float)t;
            break; // 触发脉冲，立刻早退！
        }
    }
    out_fire_times[idx] = fire_time;
}

// --- Layer 2 & 4: 基于首发时间的事件驱动卷积融合算子 ---
__global__ void conv_fused_kernel(const float* in_fire_times, float* out_fire_times, 
                                  const float* weight, int N, int C_in, int C_out, int H, int W, float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;
    if (idx >= total) return;
    
    int n = idx / (C_out * H * W);
    int rem = idx % (C_out * H * W);
    int c_out = rem / (H * W);
    int y = (rem / W) % H;
    int x = rem % W;
    
    float in_val[160] = {0.0f};
    
    for (int c_in = 0; c_in < C_in; ++c_in) {
        for (int ky = 0; ky < 3; ++ky) {
            for (int kx = 0; kx < 3; ++kx) {
                int in_y = y + ky - 1;
                int in_x = x + kx - 1;
                if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                    int in_idx = ((n * C_in + c_in) * H + in_y) * W + in_x;
                    float t_f = in_fire_times[in_idx];
                    if (t_f < 160.0f) { // 如果前突触神经元发放了脉冲
                        int w_idx = ((c_out * C_in + c_in) * 3 + ky) * 3 + kx;
                        in_val[(int)t_f] += weight[w_idx];
                    }
                }
            }
        }
    }
    
    float I = 0, V = 0;
    float fire_time = 9999.0f;
    for (int t = 0; t < 160; ++t) {
        I = I * decay + in_val[t];
        V = V + I;
        if (V >= th) {
            fire_time = (float)t;
            break;
        }
    }
    out_fire_times[idx] = fire_time;
}

// --- Layer 3 & 5: 基于首发时间的 MaxPool 融合算子 ---
__global__ void pool_fused_kernel(const float* in_fire_times, float* out_fire_times, 
                                  int N, int C, int H_in, int W_in, float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in / 2;
    int W_out = W_in / 2;
    int total = N * C * H_out * W_out;
    if (idx >= total) return;
    
    int n = idx / (C * H_out * W_out);
    int rem = idx % (C * H_out * W_out);
    int c = rem / (H_out * W_out);
    int y_out = (rem / W_out) % H_out;
    int x_out = rem % W_out;
    
    float in_val[160] = {0.0f};
    
    int y_in = y_out * 2;
    int x_in = x_out * 2;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int in_idx = ((n * C + c) * H_in + (y_in + dy)) * W_in + (x_in + dx);
            float t_f = in_fire_times[in_idx];
            if (t_f < 160.0f) {
                in_val[(int)t_f] = 1.0f; // 空间池化：只要邻域内有脉冲，本处电流输入即为1
            }
        }
    }
    
    float I = 0, V = 0;
    float fire_time = 9999.0f;
    for (int t = 0; t < 160; ++t) {
        I = I * decay + in_val[t];
        V = V + I;
        if (V >= th) {
            fire_time = (float)t;
            break;
        }
    }
    out_fire_times[idx] = fire_time;
}

// --- Layer 6: 带有递归连接 (Recurrent) 的 FC1 算子 ---
// [精妙设计] 1 Block = 1 Sample, 128 线程对应 128 个神经元
__global__ void fc1_fused_kernel(const float* in_fire_times, float* out_fire_times, 
                                 const float* w_fc1, const float* w_fc1_rec, 
                                 int N, int in_features, int out_features, float decay, float th) {
    int n = blockIdx.x; // Block 处理 1 个 Sample
    if (n >= N) return;
    int i = threadIdx.x; // Thread 处理该 Sample 中的 1 个神经元 (0 ~ 127)
    
    // 1. 预先计算前馈 (FF) 的电流输入序列
    float in_val[160] = {0.0f};
    for (int j = 0; j < in_features; ++j) {
        float t_f = in_fire_times[n * in_features + j];
        if (t_f < 160.0f) {
            in_val[(int)t_f] += w_fc1[i * in_features + j];
        }
    }
    
    // 2. 利用 Shared Memory 并行处理时间步间的递归依赖
    __shared__ bool fired_t_minus_1[128];
    fired_t_minus_1[i] = false;
    
    float I = 0, V = 0;
    float my_fire = 9999.0f;
    bool has_fired = false;
    
    for (int t = 0; t < 160; ++t) {
        __syncthreads(); // 等待所有神经元准备好前一时刻的脉冲状态
        
        float in_rec = 0.0f;
        for (int j = 0; j < 128; ++j) {
            if (fired_t_minus_1[j]) {
                in_rec += w_fc1_rec[i * 128 + j];
            }
        }
        
        I = I * decay + in_val[t] + in_rec;
        V = V + I;
        
        bool fire_now = false;
        if (V >= th && !has_fired) {
            fire_now = true;
            has_fired = true;
            my_fire = (float)t;
            V = -10000.0f; 
        }
        
        __syncthreads(); // 等待所有神经元用完旧状态，再写入新状态
        fired_t_minus_1[i] = fire_now; // 只有在激发的严格时刻为 true
    }
    
    out_fire_times[n * out_features + i] = my_fire;
}

// --- Layer 7: FC2 输出算子 ---
__global__ void fc2_fused_kernel(const float* in_fire_times, float* out_fire_times, 
                                 const float* w_fc2, int N, int in_features, int out_features, float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_features;
    if (idx >= total) return;
    
    int n = idx / out_features;
    int i = idx % out_features;
    
    float in_val[160] = {0.0f};
    for (int j = 0; j < in_features; ++j) {
        float t_f = in_fire_times[n * in_features + j];
        if (t_f < 160.0f) {
            in_val[(int)t_f] += w_fc2[i * in_features + j];
        }
    }
    
    float I = 0, V = 0;
    float fire_time = 9999.0f;
    for (int t = 0; t < 160; ++t) {
        I = I * decay + in_val[t];
        V = V + I;
        if (V >= th) {
            fire_time = (float)t;
            break;
        }
    }
    out_fire_times[idx] = fire_time;
}

// ==========================================================
// 辅助与主流程
// ==========================================================
void load_binary(const char* filepath, void* host_ptr, size_t bytes) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) { std::cerr << "Failed to open " << filepath << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(host_ptr), bytes);
    file.close();
}

int main() {
    std::cout << "🚀 Starting Fused Layer-by-Layer TTFS-SNN..." << std::endl;
    int batch_size = NUM_SAMPLES;
    
    size_t w_bytes = 1292032 * sizeof(float);
    float* h_weights = new float[1292032];
    load_binary("cuda_assets/dvsgesture_weights.bin", h_weights, w_bytes);
    float* d_weights;
    CHECK_CUDA(cudaMalloc(&d_weights, w_bytes));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, w_bytes, cudaMemcpyHostToDevice));

    float *w_conv1 = d_weights + 0, *w_conv2 = d_weights + 1152, *w_conv3 = d_weights + 74880;
    float *w_fc1_rec = d_weights + 222336, *w_fc1 = d_weights + 238720, *w_fc2 = d_weights + 1287296;

    size_t data_bytes = batch_size * 2 * 32 * 32 * T_TOTAL * sizeof(float);
    float* h_data = new float[batch_size * 2 * 32 * 32 * T_TOTAL];
    load_binary("cuda_assets/train_data.bin", h_data, data_bytes);
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, data_bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

    int* h_labels = new int[batch_size * NUM_CLASSES];
    load_binary("cuda_assets/train_labels.bin", h_labels, batch_size * NUM_CLASSES * sizeof(int));

    // 计算各层神经元总量
    int n_c1 = batch_size * 64 * 32 * 32, n_p1 = batch_size * 128 * 16 * 16;
    int n_c2 = batch_size * 128 * 32 * 32, n_p2 = batch_size * 128 * 8 * 8;
    int n_c3 = batch_size * 128 * 16 * 16, n_fc1 = batch_size * 128, n_fc2 = batch_size * 10;

    // 分配 Spike-Time 张量 (只记录触发时间，显存占用断崖式下降！)
    float *d_c1_ft, *d_c2_ft, *d_p1_ft, *d_c3_ft, *d_p2_ft, *d_fc1_ft, *d_fc2_ft;
    CHECK_CUDA(cudaMalloc(&d_c1_ft, n_c1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c2_ft, n_c2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p1_ft, n_p1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c3_ft, n_c3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p2_ft, n_p2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fc1_ft, n_fc1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fc2_ft, n_fc2 * sizeof(float)));

    // ==========================================================
    // 极致推理阶段 (完全消除了时间维度的大循环)
    // ==========================================================
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt); cudaEventCreate(&stop_evt);
    cudaEventRecord(start_evt);

    // 一次性跑穿每一层的所有时间跨度
    conv1_fused_kernel<<<GET_BLOCKS(n_c1), 256>>>(d_data, d_c1_ft, w_conv1, batch_size, decay_syn_conv, v_th_conv);
    conv_fused_kernel<<<GET_BLOCKS(n_c2), 256>>>(d_c1_ft, d_c2_ft, w_conv2, batch_size, 64, 128, 32, 32, decay_syn_conv, v_th_conv);
    pool_fused_kernel<<<GET_BLOCKS(n_p1), 256>>>(d_c2_ft, d_p1_ft, batch_size, 128, 32, 32, decay_syn_conv, v_th_conv);
    conv_fused_kernel<<<GET_BLOCKS(n_c3), 256>>>(d_p1_ft, d_c3_ft, w_conv3, batch_size, 128, 128, 16, 16, decay_syn_conv, v_th_conv);
    pool_fused_kernel<<<GET_BLOCKS(n_p2), 256>>>(d_c3_ft, d_p2_ft, batch_size, 128, 16, 16, decay_syn_conv, v_th_conv);
    
    // FC1特殊调度: 强制 1 Block = 1 Sample 处理循环依赖
    fc1_fused_kernel<<<batch_size, 128>>>(d_p2_ft, d_fc1_ft, w_fc1, w_fc1_rec, batch_size, 8192, 128, decay_syn_fc, v_th_fc);
    
    fc2_fused_kernel<<<GET_BLOCKS(n_fc2), 256>>>(d_fc1_ft, d_fc2_ft, w_fc2, batch_size, 128, 10, decay_syn_fc, v_th_fc);

    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_evt, stop_evt);

    std::cout << "=================================================" << std::endl;
    std::cout << "⏱️ Fused 逐层运算推理耗时: " << milliseconds << " ms" << std::endl;
    std::cout << "🔥 最终吞吐量: " << (batch_size * 1000.0f) / milliseconds << " samples/sec" << std::endl;
    std::cout << "=================================================" << std::endl;

    // ==========================================================
    // 验证准确率 (Argmin over FC2 output fire times)
    // ==========================================================
    float* h_first_times = new float[n_fc2];
    CHECK_CUDA(cudaMemcpy(h_first_times, d_fc2_ft, n_fc2 * sizeof(float), cudaMemcpyDeviceToHost));

    int correct_fs = 0;
    for (int i = 0; i < batch_size; ++i) {
        int true_label = 0;
        int max_gt = h_labels[i * 10];
        for (int c = 1; c < 10; ++c) {
            if (h_labels[i * 10 + c] > max_gt) { max_gt = h_labels[i * 10 + c]; true_label = c; }
        }
        int pred_fs = 0;
        float min_time = h_first_times[i * 10];
        for (int c = 1; c < 10; ++c) {
            if (h_first_times[i * 10 + c] < min_time) { min_time = h_first_times[i * 10 + c]; pred_fs = c; }
        }
        if (pred_fs == true_label) correct_fs++;
    }

    std::cout << "✅ Strict Fused-TTFS Accuracy: " << std::fixed << std::setprecision(2) << ((float)correct_fs / batch_size) * 100.0f << " %" << std::endl;
    std::cout << "=================================================" << std::endl;

    return 0;
}