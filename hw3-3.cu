#include <immintrin.h>
#include <omp.h>
#include <pthread.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <queue>
#include <thread>
#include <utility>
#include <vector>

#ifdef DEBUG
#define DEBUG_PRINT(fmt, args...) fprintf(stderr, fmt, ##args);
#define DEBUG_MSG(str) std::cout << str << "\n";
#define CUDA_EXE(F)                                                \
    {                                                              \
        cudaError_t err = F;                                       \
        if ((err != cudaSuccess)) {                                \
            printf("Error %s at %s:%d\n", cudaGetErrorString(err), \
                   __FILE__, __LINE__);                            \
            exit(-1);                                              \
        }                                                          \
    }
#define CUDA_CHECK()                                                                    \
    {                                                                                   \
        cudaError_t err = cudaGetLastError();                                           \
        if ((err != cudaSuccess)) {                                                     \
            printf("Error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(-1);                                                                   \
        }                                                                               \
    }
#else
#define DEBUG_PRINT(fmt, args...)
#define DEBUG_MSG(str)
#define CUDA_EXE(F) F;
#define CUDA_CHECK()
#endif  // DEBUG

#ifdef TIMING
#include <ctime>
#define TIMING_START(arg)          \
    struct timespec __start_##arg; \
    clock_gettime(CLOCK_MONOTONIC, &__start_##arg);
#define TIMING_END(arg)                                                                       \
    {                                                                                         \
        struct timespec __temp_##arg, __end_##arg;                                            \
        double __duration_##arg;                                                              \
        clock_gettime(CLOCK_MONOTONIC, &__end_##arg);                                         \
        if ((__end_##arg.tv_nsec - __start_##arg.tv_nsec) < 0) {                              \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec - 1;              \
            __temp_##arg.tv_nsec = 1000000000 + __end_##arg.tv_nsec - __start_##arg.tv_nsec;  \
        } else {                                                                              \
            __temp_##arg.tv_sec = __end_##arg.tv_sec - __start_##arg.tv_sec;                  \
            __temp_##arg.tv_nsec = __end_##arg.tv_nsec - __start_##arg.tv_nsec;               \
        }                                                                                     \
        __duration_##arg = __temp_##arg.tv_sec + (double)__temp_##arg.tv_nsec / 1000000000.0; \
        printf("%s took %lfs.\n", #arg, __duration_##arg);                                    \
    }
#else
#define TIMING_START(arg)
#define TIMING_END(arg)
#endif  // TIMING

#define TILE 26
#define block_size 78
#define div_block 3
const int INF = ((1 << 30) - 1);

__global__ void proc_1_glob(int *blk_dist, int k, int pitch);
__global__ void proc_2_glob(int *blk_dist, int s, int k, int pitch);
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int pitch);

__global__ void init_dist(int *blk_dist, int pitch);
__global__ void build_dist(int *edge, int E, int *blk_dist, int pitch);

int main(int argc, char **argv) {
    assert(argc == 3);

    char *input_filename = argv[1];
    char *output_filename = argv[2];
    FILE *input_file;
    FILE *output_file;
    int ncpus = omp_get_max_threads();
    int device_cnt;
    int V, E;
    int *edge;
    int *edge_dev[2];
    int *dist;
    int *dist_dev[2];
    int VP;
    int nblocks;
    size_t pitch[2], int_pitch[2];

    cudaGetDeviceCount(&device_cnt);

    TIMING_START(hw3_3);

    /* input */
    TIMING_START(input);
    input_file = fopen(input_filename, "rb");
    assert(input_file);
    fread(&V, sizeof(int), 1, input_file);
    fread(&E, sizeof(int), 1, input_file);
    edge = (int *)malloc(sizeof(int) * 3 * E);
    fread(edge, sizeof(int), 3 * E, input_file);
    dist = (int *)malloc(sizeof(int) * V * V);
    fclose(input_file);
    DEBUG_PRINT("vertices: %d\nedges: %d\n", V, E);
    TIMING_END(input);

    /* calculate */
    TIMING_START(calculate);
    nblocks = (int)ceilf(float(V) / block_size);
    VP = nblocks * block_size;
#pragma omp parallel num_threads(2) default(shared)
    {
        int tid = omp_get_thread_num();
        int peerid = !tid;
        int start, range;
        if (tid == 0) {
            start = 0;
            range = nblocks / 2;
        } else {
            start = nblocks / 2;
            range = nblocks - start;
        }

        cudaSetDevice(tid);
        CUDA_CHECK();
        cudaMalloc(&edge_dev[tid], sizeof(int) * 3 * E);
        CUDA_CHECK();
        CUDA_EXE(cudaMallocPitch(&dist_dev[tid], &pitch[tid], sizeof(int) * VP, VP));
        int_pitch[tid] = pitch[tid] >> 2;
        CUDA_CHECK();

        cudaMemcpy(edge_dev[tid], edge, sizeof(int) * 3 * E, cudaMemcpyDefault);
        CUDA_CHECK();
#pragma omp barrier

        CUDA_CHECK();
        init_dist<<<dim3(VP / TILE, VP / TILE), dim3(TILE, TILE)>>>(dist_dev[tid], int_pitch[tid]);
        CUDA_CHECK();
        build_dist<<<(int)ceilf((float)E / (TILE * TILE)), TILE * TILE>>>(edge_dev[tid], E, dist_dev[tid], int_pitch[tid]);
        CUDA_CHECK();
        cudaFree(edge_dev[tid]);

        dim3 blk(TILE, TILE);
        for (int k = 0, nk = nblocks - 1; k < nblocks; k++, nk--) {
            /* Sync */
            if (range > 0 && k >= start && k < start + range)
                CUDA_EXE(cudaMemcpy2D(
                    dist_dev[peerid] + int_pitch[peerid] * block_size * k, pitch[peerid],
                    dist_dev[tid] + int_pitch[tid] * block_size * k, pitch[tid],
                    sizeof(int) * VP, block_size, cudaMemcpyDefault));
#pragma omp barrier
            /* Phase 1 */
            proc_1_glob<<<1, blk>>>(dist_dev[tid], k, int_pitch[tid]);
            /* Phase 2 */
            if (nblocks - 1 > 0)
                proc_2_glob<<<dim3(nblocks - 1, 2), blk>>>(dist_dev[tid], 0, k, int_pitch[tid]);
            /* Phase 3 */
            if (nblocks - 1 > 0 && range > 0)
                proc_3_glob<<<dim3(nblocks - 1, range), blk>>>(dist_dev[tid], start, 0, k, int_pitch[tid]);
        }
        if (tid == 0) {
            CUDA_EXE(cudaHostRegister(dist, sizeof(int) * V * V, cudaHostRegisterDefault));
#pragma omp barrier
            CUDA_EXE(cudaMemcpy2D(dist, sizeof(int) * V, dist_dev[tid], pitch[tid], sizeof(int) * V, V, cudaMemcpyDefault));
            CUDA_EXE(cudaHostUnregister(dist));
        } else {
            if (range > 0)
                CUDA_EXE(cudaMemcpy2D(
                    dist_dev[peerid] + int_pitch[peerid] * block_size * start, pitch[peerid],
                    dist_dev[tid] + int_pitch[tid] * block_size * start, pitch[tid],
                    sizeof(int) * VP, block_size * range, cudaMemcpyDefault));
#pragma omp barrier
        }
        CUDA_CHECK();
        cudaFree(dist_dev[tid]);
    }
    TIMING_END(calculate);

    /* output */
    TIMING_START(output);
    output_file = fopen(output_filename, "w");
    assert(output_file);
    fwrite(dist, 1, sizeof(int) * V * V, output_file);
    fclose(output_file);
    TIMING_END(output);
    TIMING_END(hw3_3);

    /* finalize */
    free(edge);
    free(dist);
    return 0;
}

#define _ref(i, j, r, c) blk_dist[i * block_size * pitch + j * block_size + (r)*pitch + c]
__global__ void proc_1_glob(int *blk_dist, int k, int pitch) {
    __shared__ int k_k_sm[block_size][block_size];

    int r = threadIdx.y;
    int c = threadIdx.x;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_sm[r + rr * TILE][c + cc * TILE] = _ref(k, k, r + rr * TILE, c + cc * TILE);
        }
    }
    __syncthreads();

#pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                k_k_sm[r + rr * TILE][c + cc * TILE] = min(k_k_sm[r + rr * TILE][c + cc * TILE], k_k_sm[r + rr * TILE][b] + k_k_sm[b][c + cc * TILE]);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            _ref(k, k, r + rr * TILE, c + cc * TILE) = k_k_sm[r + rr * TILE][c + cc * TILE];
        }
    }
}
__global__ void proc_2_glob(int *blk_dist, int s, int k, int pitch) {
    __shared__ int k_k_sm[block_size][block_size];
    __shared__ int sm[block_size][block_size];

    int i = s + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;

    if (i >= k)
        i++;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_k_sm[r + rr * TILE][c + cc * TILE] = _ref(k, k, r + rr * TILE, c + cc * TILE);
        }
    }
    if (blockIdx.y == 0) {
        /* rows */
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                sm[r + rr * TILE][c + cc * TILE] = _ref(i, k, r + rr * TILE, c + cc * TILE);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < block_size; b++) {
#pragma unroll
            for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
                for (int cc = 0; cc < div_block; cc++) {
                    sm[r + rr * TILE][c + cc * TILE] = min(sm[r + rr * TILE][c + cc * TILE], sm[r + rr * TILE][b] + k_k_sm[b][c + cc * TILE]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                _ref(i, k, r + rr * TILE, c + cc * TILE) = sm[r + rr * TILE][c + cc * TILE];
            }
        }
    } else {
        /* cols */
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                sm[r + rr * TILE][c + cc * TILE] = _ref(k, i, r + rr * TILE, c + cc * TILE);
            }
        }
        __syncthreads();

#pragma unroll
        for (int b = 0; b < block_size; b++) {
#pragma unroll
            for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
                for (int cc = 0; cc < div_block; cc++) {
                    sm[r + rr * TILE][c + cc * TILE] = min(sm[r + rr * TILE][c + cc * TILE], k_k_sm[r + rr * TILE][b] + sm[b][c + cc * TILE]);
                }
            }
            __syncthreads();
        }
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                _ref(k, i, r + rr * TILE, c + cc * TILE) = sm[r + rr * TILE][c + cc * TILE];
            }
        }
    }
}
__global__ void proc_3_glob(int *blk_dist, int s_i, int s_j, int k, int pitch) {
    __shared__ int i_k_sm[block_size][block_size];
    __shared__ int k_j_sm[block_size][block_size];

    int i = s_i + blockIdx.y;
    int j = s_j + blockIdx.x;
    int r = threadIdx.y;
    int c = threadIdx.x;
    int loc[div_block][div_block];

    if (i == k)
        return;
    if (j >= k)
        j++;

#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            i_k_sm[r + rr * TILE][c + cc * TILE] = _ref(i, k, r + rr * TILE, c + cc * TILE);
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            k_j_sm[r + rr * TILE][c + cc * TILE] = _ref(k, j, r + rr * TILE, c + cc * TILE);
        }
    }
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            loc[rr][cc] = _ref(i, j, r + rr * TILE, c + cc * TILE);
        }
    }

#pragma unroll
    for (int b = 0; b < block_size; b++) {
#pragma unroll
        for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
            for (int cc = 0; cc < div_block; cc++) {
                loc[rr][cc] = min(loc[rr][cc], i_k_sm[r + rr * TILE][b] + k_j_sm[b][c + cc * TILE]);
            }
        }
    }
#pragma unroll
    for (int rr = 0; rr < div_block; rr++) {
#pragma unroll
        for (int cc = 0; cc < div_block; cc++) {
            _ref(i, j, r + rr * TILE, c + cc * TILE) = loc[rr][cc];
        }
    }
}
__global__ void init_dist(int *blk_dist, int pitch) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    blk_dist[r * pitch + c] = (r != c) * INF;
}
__global__ void build_dist(int *edge, int E, int *blk_dist, int pitch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < E) {
        int src = *(edge + idx * 3);
        int dst = *(edge + idx * 3 + 1);
        int w = *(edge + idx * 3 + 2);
        blk_dist[src * pitch + dst] = w;
    }
}