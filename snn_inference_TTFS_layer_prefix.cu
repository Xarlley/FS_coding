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
const int CHUNK_SIZE = 32; // 将 160 分为 5 个 32步的 Block

const float decay_syn_conv = exp(-1.0f / 5.0f);
const float decay_syn_fc = exp(-1.0f / 60.0f);

const float v_th_conv = 5.0f;
const float v_th_fc = 10.0f;

// ==========================================================
// 极致截流分块算子 (严格仅加载当前 Block 数据)
// ==========================================================

// --- Layer 1: 真正意义上对稠密显存的块加载与早退 ---
__global__ void conv1_block_trunc_kernel(const float* all_data, float* out_fire_times, 
                                         const float* weight, int N, float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * 64 * 32 * 32;
    if (idx >= total) return;
    
    int n = idx / (64 * 32 * 32);
    int rem = idx % (64 * 32 * 32);
    int c_out = rem / (32 * 32);
    int y = (rem / 32) % 32;
    int x = rem % 32;
    
    float I = 0.0f, V = 0.0f;
    
    // 提取循环不变量：基准指针偏移
    int n_c_offset = n * 2 * 32 * 32 * 160;
    int w_out_offset = c_out * 2 * 9;
    
    for (int chunk = 0; chunk < 160; chunk += CHUNK_SIZE) {
        float in_val[CHUNK_SIZE] = {0.0f}; // 严格驻留寄存器
        
        // 步骤1：仅加载当前 Block [chunk, chunk+31] 的慢速显存数据
        for (int c_in = 0; c_in < 2; ++c_in) {
            int in_c_base = n_c_offset + c_in * 32 * 32 * 160;
            int w_in_base = w_out_offset + c_in * 9;
            
            for (int ky = 0; ky < 3; ++ky) {
                int in_y = y + ky - 1;
                if (in_y < 0 || in_y >= 32) continue;
                int in_y_base = in_c_base + in_y * 32 * 160;
                int w_ky_base = w_in_base + ky * 3;
                
                for (int kx = 0; kx < 3; ++kx) {
                    int in_x = x + kx - 1;
                    if (in_x < 0 || in_x >= 32) continue;
                    
                    int data_base = in_y_base + in_x * 160 + chunk; 
                    float w = weight[w_ky_base + kx];
                    
                    // 硬件展开，极速合并读取
                    #pragma unroll
                    for (int t = 0; t < CHUNK_SIZE; ++t) {
                        in_val[t] += w * all_data[data_base + t];
                    }
                }
            }
        }
        
        // 步骤2：处理当前 Block，发现超阈值直接熔断整个循环！免除后续 Block 所有访存！
        for (int t = 0; t < CHUNK_SIZE; ++t) {
            I = I * decay + in_val[t];
            V += I;
            if (V >= th) {
                out_fire_times[idx] = (float)(chunk + t);
                return; 
            }
        }
    }
    out_fire_times[idx] = 9999.0f;
}

// --- Layer 2 & 4: 隐藏层的截流分块 (消除ALU瓶颈) ---
__global__ void conv_block_trunc_kernel(const float* in_fire_times, float* out_fire_times, 
                                        const float* weight, int N, int C_in, int C_out, int H, int W, float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;
    if (idx >= total) return;
    
    int n = idx / (C_out * H * W);
    int rem = idx % (C_out * H * W);
    int c_out = rem / (H * W);
    int y = (rem / W) % H;
    int x = rem % W;
    
    float I = 0.0f, V = 0.0f;
    
    // 极其重要的优化：在块循环外预先算好指针基址，彻底消除 inner loop 的整数乘法开销
    int n_base = n * C_in * H * W;
    int w_out_base = c_out * C_in * 9;
    
    for (int chunk = 0; chunk < 160; chunk += CHUNK_SIZE) {
        float in_val[CHUNK_SIZE] = {0.0f};
        
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int in_c_base = n_base + c_in * H * W;
            int w_in_base = w_out_base + c_in * 9;
            
            for (int ky = 0; ky < 3; ++ky) {
                int in_y = y + ky - 1;
                if (in_y < 0 || in_y >= H) continue;
                int in_y_base = in_c_base + in_y * W;
                int w_ky_base = w_in_base + ky * 3;
                
                for (int kx = 0; kx < 3; ++kx) {
                    int in_x = x + kx - 1;
                    if (in_x < 0 || in_x >= W) continue;
                    
                    float t_f = in_fire_times[in_y_base + in_x];
                    
                    // 仅当脉冲落在此 block 内，才读取巨型 weight 显存
                    if (t_f >= chunk && t_f < chunk + CHUNK_SIZE) {
                        in_val[(int)t_f - chunk] += weight[w_ky_base + kx];
                    }
                }
            }
        }
        
        for (int t = 0; t < CHUNK_SIZE; ++t) {
            I = I * decay + in_val[t];
            V += I;
            if (V >= th) {
                out_fire_times[idx] = (float)(chunk + t);
                return; // 直接跳出，不再搜寻后续块的 weight！
            }
        }
    }
    out_fire_times[idx] = 9999.0f;
}

// --- Layer 3 & 5: MaxPool 截流分块 ---
__global__ void pool_block_trunc_kernel(const float* in_fire_times, float* out_fire_times, 
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
    
    float I = 0.0f, V = 0.0f;
    int base_offset = (n * C + c) * H_in * W_in;
    
    for (int chunk = 0; chunk < 160; chunk += CHUNK_SIZE) {
        float in_val[CHUNK_SIZE] = {0.0f};
        
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                int in_idx = base_offset + (y_out * 2 + dy) * W_in + (x_out * 2 + dx);
                float t_f = in_fire_times[in_idx];
                if (t_f >= chunk && t_f < chunk + CHUNK_SIZE) {
                    in_val[(int)t_f - chunk] = 1.0f;
                }
            }
        }
        
        for (int t = 0; t < CHUNK_SIZE; ++t) {
            I = I * decay + in_val[t];
            V += I;
            if (V >= th) { out_fire_times[idx] = (float)(chunk + t); return; }
        }
    }
    out_fire_times[idx] = 9999.0f;
}

// --- Layer 6: FC1 递归层的截流分块 ---
__global__ void fc1_block_trunc_kernel(const float* in_fire_times, float* out_fire_times, 
                                       const float* w_fc1, const float* w_fc1_rec, 
                                       int N, int in_features, int out_features, float decay, float th) {
    int n = blockIdx.x; 
    if (n >= N) return;
    int i = threadIdx.x; 
    
    __shared__ bool fired_t_minus_1[128];
    fired_t_minus_1[i] = false;
    
    float I = 0.0f, V = 0.0f;
    float my_fire = 9999.0f;
    bool has_fired = false;
    
    int in_offset = n * in_features;
    int w_offset = i * in_features;
    int rec_offset = i * 128;
    
    for (int chunk = 0; chunk < 160; chunk += CHUNK_SIZE) {
        float in_val_ff[CHUNK_SIZE] = {0.0f}; 
        
        // 仅加载当前 Block 内的前向输入
        for (int j = 0; j < in_features; ++j) {
            float t_f = in_fire_times[in_offset + j];
            if (t_f >= chunk && t_f < chunk + CHUNK_SIZE) {
                in_val_ff[(int)t_f - chunk] += w_fc1[w_offset + j];
            }
        }
        
        for (int t = 0; t < CHUNK_SIZE; ++t) {
            __syncthreads(); 
            float in_rec = 0.0f;
            for (int j = 0; j < 128; ++j) {
                if (fired_t_minus_1[j]) in_rec += w_fc1_rec[rec_offset + j];
            }
            
            I = I * decay + in_val_ff[t] + in_rec;
            V += I;
            
            bool fire_now = false;
            if (V >= th && !has_fired) {
                fire_now = true;
                has_fired = true;
                my_fire = (float)(chunk + t);
                V = -10000.0f; 
            }
            __syncthreads();
            fired_t_minus_1[i] = fire_now;
        }
    }
    out_fire_times[n * out_features + i] = my_fire;
}

// --- Layer 7: FC2 输出层截流分块 ---
__global__ void fc2_block_trunc_kernel(const float* in_fire_times, float* out_fire_times, 
                                       const float* w_fc2, int N, int in_features, int out_features, float decay, float th) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_features;
    if (idx >= total) return;
    
    int n = idx / out_features;
    int i = idx % out_features;
    float I = 0.0f, V = 0.0f;
    
    int in_offset = n * in_features;
    int w_offset = i * in_features;
    
    for (int chunk = 0; chunk < 160; chunk += CHUNK_SIZE) {
        float in_val[CHUNK_SIZE] = {0.0f};
        for (int j = 0; j < in_features; ++j) {
            float t_f = in_fire_times[in_offset + j];
            if (t_f >= chunk && t_f < chunk + CHUNK_SIZE) {
                in_val[(int)t_f - chunk] += w_fc2[w_offset + j];
            }
        }
        for (int t = 0; t < CHUNK_SIZE; ++t) {
            I = I * decay + in_val[t];
            V += I;
            if (V >= th) {
                out_fire_times[idx] = (float)(chunk + t);
                return; // 直接截断
            }
        }
    }
    out_fire_times[idx] = 9999.0f;
}

// ==========================================================
// 主函数与流程
// ==========================================================
void load_binary(const char* filepath, void* host_ptr, size_t bytes) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) { std::cerr << "Failed to open " << filepath << std::endl; exit(1); }
    file.read(reinterpret_cast<char*>(host_ptr), bytes);
    file.close();
}

int main() {
    std::cout << "🚀 Starting Block-Truncated TTFS-SNN (Full ALU & Mem Optimization)..." << std::endl;
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

    int n_c1 = batch_size * 64 * 32 * 32, n_p1 = batch_size * 128 * 16 * 16;
    int n_c2 = batch_size * 128 * 32 * 32, n_p2 = batch_size * 128 * 8 * 8;
    int n_c3 = batch_size * 128 * 16 * 16, n_fc1 = batch_size * 128, n_fc2 = batch_size * 10;

    float *d_c1_ft, *d_c2_ft, *d_p1_ft, *d_c3_ft, *d_p2_ft, *d_fc1_ft, *d_fc2_ft;
    CHECK_CUDA(cudaMalloc(&d_c1_ft, n_c1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c2_ft, n_c2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p1_ft, n_p1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_c3_ft, n_c3 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_p2_ft, n_p2 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fc1_ft, n_fc1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_fc2_ft, n_fc2 * sizeof(float)));

    // ==========================================================
    // 推理阶段
    // ==========================================================
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt); cudaEventCreate(&stop_evt);
    cudaEventRecord(start_evt);

    conv1_block_trunc_kernel<<<GET_BLOCKS(n_c1), 256>>>(d_data, d_c1_ft, w_conv1, batch_size, decay_syn_conv, v_th_conv);
    conv_block_trunc_kernel<<<GET_BLOCKS(n_c2), 256>>>(d_c1_ft, d_c2_ft, w_conv2, batch_size, 64, 128, 32, 32, decay_syn_conv, v_th_conv);
    pool_block_trunc_kernel<<<GET_BLOCKS(n_p1), 256>>>(d_c2_ft, d_p1_ft, batch_size, 128, 32, 32, decay_syn_conv, v_th_conv);
    conv_block_trunc_kernel<<<GET_BLOCKS(n_c3), 256>>>(d_p1_ft, d_c3_ft, w_conv3, batch_size, 128, 128, 16, 16, decay_syn_conv, v_th_conv);
    pool_block_trunc_kernel<<<GET_BLOCKS(n_p2), 256>>>(d_c3_ft, d_p2_ft, batch_size, 128, 16, 16, decay_syn_conv, v_th_conv);
    
    fc1_block_trunc_kernel<<<batch_size, 128>>>(d_p2_ft, d_fc1_ft, w_fc1, w_fc1_rec, batch_size, 8192, 128, decay_syn_fc, v_th_fc);
    fc2_block_trunc_kernel<<<GET_BLOCKS(n_fc2), 256>>>(d_fc1_ft, d_fc2_ft, w_fc2, batch_size, 128, 10, decay_syn_fc, v_th_fc);

    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_evt, stop_evt);

    std::cout << "=================================================" << std::endl;
    std::cout << "⏱️ 块截断级极速推理耗时: " << milliseconds << " ms" << std::endl;
    std::cout << "🔥 最终吞吐量: " << (batch_size * 1000.0f) / milliseconds << " samples/sec" << std::endl;
    std::cout << "=================================================" << std::endl;

    // ==========================================================
    // 验证准确率
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

    std::cout << "✅ Strict Block-Truncated TTFS Accuracy: " << std::fixed << std::setprecision(2) << ((float)correct_fs / batch_size) * 100.0f << " %" << std::endl;
    std::cout << "=================================================" << std::endl;

    return 0;
}