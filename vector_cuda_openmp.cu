#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cuda_runtime.h>

#define N 1048576
#define THREADS_PER_BLOCK 256

__global__ void add_vectors(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    float *A, *B, *C_cpu, *C_gpu;
    float *d_A, *d_B, *d_C;

    A = (float*) malloc(N * sizeof(float));
    B = (float*) malloc(N * sizeof(float));
    C_cpu = (float*) malloc(N * sizeof(float));
    C_gpu = (float*) malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    double start_cpu = omp_get_wtime();

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        C_cpu[i] = A[i] + B[i];
    }

    double end_cpu = omp_get_wtime();
    double time_cpu = end_cpu - start_cpu;

    cudaEvent_t start_gpu, end_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&end_gpu);

    cudaEventRecord(start_gpu);

    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    add_vectors<<<blocks, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    cudaMemcpy(C_gpu, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaEventRecord(end_gpu);
    cudaEventSynchronize(end_gpu);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, end_gpu);

    double time_gpu = milliseconds / 1000.0;

    int errores = 0;

    for (int i = 0; i < N; i++) {
        if (C_gpu[i] != C_cpu[i]) {
            errores++;
        }
    }

    printf("Elementos procesados: %d\n", N);
    printf("Errores encontrados: %d\n", errores);
    printf("Tiempo CPU OpenMP: %f segundos\n", time_cpu);
    printf("Tiempo GPU CUDA: %f segundos\n", time_gpu);
    printf("Speedup: %f\n", time_cpu / time_gpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C_cpu);
    free(C_gpu);

    return 0;
}
