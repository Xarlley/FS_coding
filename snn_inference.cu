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

// 捕获异步 Kernel 启动错误的宏
#define LAUNCH_CHECK() { CHECK_CUDA(cudaGetLastError()); }

// ==========================================================
// 网络参数常量配置
// ==========================================================
const int NUM_SAMPLES = 1078;
const int T_TOTAL = 160; 
const int NUM_CLASSES = 10;

const float decay_mem_conv = exp(-1.0f / 5.0f);
const float decay_syn_conv = exp(-1.0f / 5.0f);
const float v_th_conv = 0.5f;

const float decay_mem_fc = exp(-1.0f / 60.0f);
const float decay_syn_fc = exp(-1.0f / 60.0f);
const float v_th_fc = 1.0f;

// ==========================================================
// 核心 CUDA 算子 (全部使用防溢出的 1D Grid)
// ==========================================================

__global__ void extract_input_step(const float* all_data, float* step_input, int N, int C, int H, int W, int T_total, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = N * C * H * W;
    if (idx < total_pixels) {
        // 取出当前时间步的数据 [N, C, H, W, T] -> [N, C, H, W]
        step_input[idx] = all_data[idx * T_total + t];
    }
}

__global__ void conv2d_kernel(const float* input, const float* weight, float* output, 
                              int N, int C_in, int C_out, int H, int W, int K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C_out * H * W;
    if (idx < total) {
        int x = idx % W;
        int y = (idx / W) % H;
        int c_out = (idx / (W * H)) % C_out;
        int n = idx / (W * H * C_out);

        int pad = K / 2;
        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    int in_y = y + ky - pad;
                    int in_x = x + kx - pad;
                    if (in_y >= 0 && in_y < H && in_x >= 0 && in_x < W) {
                        int in_idx = ((n * C_in + c_in) * H + in_y) * W + in_x;
                        int w_idx = ((c_out * C_in + c_in) * K + ky) * K + kx;
                        sum += input[in_idx] * weight[w_idx];
                    }
                }
            }
        }
        output[idx] = sum;
    }
}

__global__ void maxpool2d_kernel(const float* input, float* output, 
                                 int N, int C, int H_in, int W_in) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H_in / 2;
    int W_out = W_in / 2;
    int total = N * C * H_out * W_out;
    
    if (idx < total) {
        int x_out = idx % W_out;
        int y_out = (idx / W_out) % H_out;
        int c = (idx / (W_out * H_out)) % C;
        int n = idx / (W_out * H_out * C);

        int x_in = x_out * 2;
        int y_in = y_out * 2;
        
        float max_val = -1e9f;
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                int in_idx = ((n * C + c) * H_in + (y_in + dy)) * W_in + (x_in + dx);
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
        output[idx] = max_val;
    }
}

__global__ void fc_kernel(const float* input, const float* weight, float* output, 
                          int N, int in_features, int out_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * out_features;
    
    if (idx < total) {
        int col = idx % out_features; // 相当于当前输出的通道/神经元
        int row = idx / out_features; // 相当于 N (Batch Index)

        float sum = 0.0f;
        for (int i = 0; i < in_features; ++i) {
            sum += input[row * in_features + i] * weight[col * in_features + i];
        }
        output[idx] = sum;
    }
}

__global__ void add_tensors_kernel(float* a, const float* b, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        a[idx] += b[idx];
    }
}

__global__ void culif_kernel(const float* input, float* v, float* I, float* spikes, 
                             float decay_mem, float decay_syn, float v_th, int total_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_neurons) {
        float curr_v = v[idx];
        float curr_I = I[idx];
        float x = input[idx];

        // 动力学更新顺序与原仓库完全一致 (使用 I[t-1] 计算 H)
        float h = curr_I + (curr_v - curr_I) * decay_mem;
        curr_I = curr_I * decay_syn + x;
        
        float spike = (h >= v_th) ? 1.0f : 0.0f;
        
        // 关键修复：当 v_reset=0.0 时的硬重置 (Hard Reset) 方程
        curr_v = (1.0f - spike) * h;

        v[idx] = curr_v;
        I[idx] = curr_I;
        spikes[idx] = spike;
    }
}

__global__ void record_output_stats_kernel(const float* spikes, float* spike_counts, float* first_times, int t, int total_neurons) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_neurons) {
        if (spikes[idx] > 0.5f) {
            spike_counts[idx] += 1.0f;
            if (first_times[idx] > 9000.0f) {
                first_times[idx] = (float)t;
            }
        }
    }
}

// ==========================================================
// 辅助函数
// ==========================================================
void load_binary(const char* filepath, void* host_ptr, size_t bytes) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << filepath << std::endl;
        exit(1);
    }
    file.read(reinterpret_cast<char*>(host_ptr), bytes);
    file.close();
}

int main() {
    std::cout << "🚀 Starting Corrected DVSGesture TTFS-SNN CUDA Inference..." << std::endl;

    int batch_size = NUM_SAMPLES;
    std::cout << "⚙️ Selected Batch Size: " << batch_size << std::endl;

    // 1. 加载权重
    size_t w_bytes = 1292032 * sizeof(float);
    float* h_weights = new float[1292032];
    load_binary("cuda_assets/dvsgesture_weights.bin", h_weights, w_bytes);
    float* d_weights;
    CHECK_CUDA(cudaMalloc(&d_weights, w_bytes));
    CHECK_CUDA(cudaMemcpy(d_weights, h_weights, w_bytes, cudaMemcpyHostToDevice));

    float* w_conv1 = d_weights + 0;
    float* w_conv2 = d_weights + 1152;
    float* w_conv3 = d_weights + 74880;
    float* w_fc1_recurrent = d_weights + 222336; 
    float* w_fc1 = d_weights + 238720;           
    float* w_fc2 = d_weights + 1287296;          

    // 2. 加载数据与标签
    size_t data_bytes = batch_size * 2 * 32 * 32 * T_TOTAL * sizeof(float);
    float* h_data = new float[batch_size * 2 * 32 * 32 * T_TOTAL];
    load_binary("cuda_assets/train_data.bin", h_data, data_bytes);
    float* d_data;
    CHECK_CUDA(cudaMalloc(&d_data, data_bytes));
    CHECK_CUDA(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));

    int* h_labels = new int[batch_size * NUM_CLASSES];
    load_binary("cuda_assets/train_labels.bin", h_labels, batch_size * NUM_CLASSES * sizeof(int));

    // 3. 显存分配
    int n_c1 = batch_size * 64 * 32 * 32;
    int n_c2 = batch_size * 128 * 32 * 32;
    int n_p1 = batch_size * 128 * 16 * 16;
    int n_c3 = batch_size * 128 * 16 * 16;
    int n_p2 = batch_size * 128 * 8 * 8;
    int n_fc1 = batch_size * 128;
    int n_fc2 = batch_size * 10;

    float *d_step_in;
    CHECK_CUDA(cudaMalloc(&d_step_in, batch_size * 2 * 32 * 32 * sizeof(float)));

    auto allocate_layer = [&](int num_neurons, float*& out, float*& v, float*& i, float*& s) {
        CHECK_CUDA(cudaMalloc(&out, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&v, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&i, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&s, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMemset(v, 0, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMemset(i, 0, num_neurons * sizeof(float)));
        CHECK_CUDA(cudaMemset(s, 0, num_neurons * sizeof(float)));
    };

    float *d_c1_out, *d_c1_v, *d_c1_i, *d_c1_s;
    float *d_c2_out, *d_c2_v, *d_c2_i, *d_c2_s;
    float *d_p1_out, *d_p1_v, *d_p1_i, *d_p1_s;
    float *d_c3_out, *d_c3_v, *d_c3_i, *d_c3_s;
    float *d_p2_out, *d_p2_v, *d_p2_i, *d_p2_s;
    float *d_fc1_out, *d_fc1_rec_out, *d_fc1_v, *d_fc1_i, *d_fc1_s;
    float *d_fc2_out, *d_fc2_v, *d_fc2_i, *d_fc2_s;

    allocate_layer(n_c1, d_c1_out, d_c1_v, d_c1_i, d_c1_s);
    allocate_layer(n_c2, d_c2_out, d_c2_v, d_c2_i, d_c2_s);
    allocate_layer(n_p1, d_p1_out, d_p1_v, d_p1_i, d_p1_s);
    allocate_layer(n_c3, d_c3_out, d_c3_v, d_c3_i, d_c3_s);
    allocate_layer(n_p2, d_p2_out, d_p2_v, d_p2_i, d_p2_s);
    allocate_layer(n_fc1, d_fc1_out, d_fc1_v, d_fc1_i, d_fc1_s);
    CHECK_CUDA(cudaMalloc(&d_fc1_rec_out, n_fc1 * sizeof(float)));
    allocate_layer(n_fc2, d_fc2_out, d_fc2_v, d_fc2_i, d_fc2_s);

    float *d_fc2_spike_counts, *d_first_times;
    CHECK_CUDA(cudaMalloc(&d_fc2_spike_counts, n_fc2 * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_fc2_spike_counts, 0, n_fc2 * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_first_times, n_fc2 * sizeof(float)));
    float* h_first_times_init = new float[n_fc2];
    std::fill_n(h_first_times_init, n_fc2, 9999.0f);
    CHECK_CUDA(cudaMemcpy(d_first_times, h_first_times_init, n_fc2 * sizeof(float), cudaMemcpyHostToDevice));
    delete[] h_first_times_init;

    // ==========================================================
    // 推理执行与计时
    // ==========================================================
    cudaEvent_t start_evt, stop_evt;
    cudaEventCreate(&start_evt);
    cudaEventCreate(&stop_evt);
    
    // 利用宏辅助计算 1D Grid 的 Blocks 数量
    #define GET_BLOCKS(total) ((total + 255) / 256)
    
    cudaEventRecord(start_evt);

    for (int t = 0; t < T_TOTAL; ++t) {
        extract_input_step<<<GET_BLOCKS(batch_size * 2 * 32 * 32), 256>>>(d_data, d_step_in, batch_size, 2, 32, 32, T_TOTAL, t);
        LAUNCH_CHECK();

        // --- Layer 1: Conv ---
        conv2d_kernel<<<GET_BLOCKS(n_c1), 256>>>(d_step_in, w_conv1, d_c1_out, batch_size, 2, 64, 32, 32, 3);
        LAUNCH_CHECK();
        culif_kernel<<<GET_BLOCKS(n_c1), 256>>>(d_c1_out, d_c1_v, d_c1_i, d_c1_s, decay_mem_conv, decay_syn_conv, v_th_conv, n_c1);

        // --- Layer 2: Conv ---
        conv2d_kernel<<<GET_BLOCKS(n_c2), 256>>>(d_c1_s, w_conv2, d_c2_out, batch_size, 64, 128, 32, 32, 3);
        LAUNCH_CHECK();
        culif_kernel<<<GET_BLOCKS(n_c2), 256>>>(d_c2_out, d_c2_v, d_c2_i, d_c2_s, decay_mem_conv, decay_syn_conv, v_th_conv, n_c2);

        // --- Layer 3: Pool ---
        maxpool2d_kernel<<<GET_BLOCKS(n_p1), 256>>>(d_c2_s, d_p1_out, batch_size, 128, 32, 32);
        LAUNCH_CHECK();
        culif_kernel<<<GET_BLOCKS(n_p1), 256>>>(d_p1_out, d_p1_v, d_p1_i, d_p1_s, decay_mem_conv, decay_syn_conv, v_th_conv, n_p1);

        // --- Layer 4: Conv ---
        conv2d_kernel<<<GET_BLOCKS(n_c3), 256>>>(d_p1_s, w_conv3, d_c3_out, batch_size, 128, 128, 16, 16, 3);
        LAUNCH_CHECK();
        culif_kernel<<<GET_BLOCKS(n_c3), 256>>>(d_c3_out, d_c3_v, d_c3_i, d_c3_s, decay_mem_conv, decay_syn_conv, v_th_conv, n_c3);

        // --- Layer 5: Pool ---
        maxpool2d_kernel<<<GET_BLOCKS(n_p2), 256>>>(d_c3_s, d_p2_out, batch_size, 128, 16, 16);
        LAUNCH_CHECK();
        culif_kernel<<<GET_BLOCKS(n_p2), 256>>>(d_p2_out, d_p2_v, d_p2_i, d_p2_s, decay_mem_conv, decay_syn_conv, v_th_conv, n_p2);

        // --- Layer 6: FC1 (Recurrent) ---
        fc_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_p2_s, w_fc1, d_fc1_out, batch_size, 8192, 128);
        LAUNCH_CHECK();
        fc_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_fc1_s, w_fc1_recurrent, d_fc1_rec_out, batch_size, 128, 128);
        LAUNCH_CHECK();
        add_tensors_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_fc1_out, d_fc1_rec_out, n_fc1);
        culif_kernel<<<GET_BLOCKS(n_fc1), 256>>>(d_fc1_out, d_fc1_v, d_fc1_i, d_fc1_s, decay_mem_fc, decay_syn_fc, v_th_fc, n_fc1);

        // --- Layer 7: FC2 (Output) ---
        fc_kernel<<<GET_BLOCKS(n_fc2), 256>>>(d_fc1_s, w_fc2, d_fc2_out, batch_size, 128, 10);
        LAUNCH_CHECK();
        culif_kernel<<<GET_BLOCKS(n_fc2), 256>>>(d_fc2_out, d_fc2_v, d_fc2_i, d_fc2_s, decay_mem_fc, decay_syn_fc, v_th_fc, n_fc2);

        // --- 统计结果 ---
        record_output_stats_kernel<<<GET_BLOCKS(n_fc2), 256>>>(d_fc2_s, d_fc2_spike_counts, d_first_times, t, n_fc2);
    }

    cudaEventRecord(stop_evt);
    cudaEventSynchronize(stop_evt);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_evt, stop_evt);
    
    std::cout << "=================================================" << std::endl;
    std::cout << "⏱️ Total GPU Compute Time: " << milliseconds << " ms" << std::endl;
    std::cout << "🔥 Throughput: " << (batch_size * 1000.0f) / milliseconds << " samples/sec" << std::endl;
    std::cout << "=================================================" << std::endl;

    // ==========================================================
    // 准确率验证计算
    // ==========================================================
    float* h_fc2_spike_counts = new float[n_fc2];
    float* h_first_times = new float[n_fc2];
    CHECK_CUDA(cudaMemcpy(h_fc2_spike_counts, d_fc2_spike_counts, n_fc2 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_first_times, d_first_times, n_fc2 * sizeof(float), cudaMemcpyDeviceToHost));

    int correct_fr = 0;
    int correct_fs = 0;

    for (int i = 0; i < batch_size; ++i) {
        int true_label = 0;
        int max_gt = h_labels[i * 10];
        for (int c = 1; c < 10; ++c) {
            if (h_labels[i * 10 + c] > max_gt) {
                max_gt = h_labels[i * 10 + c];
                true_label = c;
            }
        }

        int pred_fr = 0;
        float max_spikes = h_fc2_spike_counts[i * 10];
        for (int c = 1; c < 10; ++c) {
            if (h_fc2_spike_counts[i * 10 + c] > max_spikes) {
                max_spikes = h_fc2_spike_counts[i * 10 + c];
                pred_fr = c;
            }
        }
        if (pred_fr == true_label) correct_fr++;

        int pred_fs = 0;
        float min_time = h_first_times[i * 10];
        for (int c = 1; c < 10; ++c) {
            if (h_first_times[i * 10 + c] < min_time) {
                min_time = h_first_times[i * 10 + c];
                pred_fs = c;
            }
        }
        if (pred_fs == true_label) correct_fs++;
    }

    std::cout << "✅ Firing Rate (FR) Accuracy : " << std::fixed << std::setprecision(2) << ((float)correct_fr / batch_size) * 100.0f << " %" << std::endl;
    std::cout << "✅ First Time (FS) Accuracy  : " << std::fixed << std::setprecision(2) << ((float)correct_fs / batch_size) * 100.0f << " %" << std::endl;
    std::cout << "=================================================" << std::endl;

    cudaFree(d_data); cudaFree(d_weights);
    delete[] h_data; delete[] h_weights; delete[] h_labels;
    delete[] h_fc2_spike_counts; delete[] h_first_times;
    return 0;
}