#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "util.h"
#define N 10000000
#define MAX_ERR 1e-6


__global__ void vector_add(float *out, float *a, float *b, int n) {
    int tid = global_thread_id() ;
    int stride = block_size();
    for(int i = tid; i < n; i += stride){
       out[i] = a[i] + b[i];
    }
}
int main(){
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 

    // Allocate host memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize host arrays
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Allocate device memory
    gpuErrchk(cudaMalloc((void**)&d_a, sizeof(float) * N));
    gpuErrchk(cudaMalloc((void**)&d_b, sizeof(float) * N));
    gpuErrchk(cudaMalloc((void**)&d_out, sizeof(float) * N));

    // Transfer data from host to device memory
    gpuErrchk(cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice));

    // Executing kernel 
    int block_size = 1;
    int thread_size = 256;
    double start = get_time();

    vector_add<<block_size,thread_size>>>(d_out, d_a, d_b, N);
    // Transfer data back to host memory
    cudaDeviceSynchronize();

    double end = get_time();


    gpuErrchk(cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost));

    // Verification
    for(int i = 0; i < N; i++){
    
        // printf("out = %f\n", out[i]);
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }
    // printf("out[0] = %f\n", out[0]);

    auto cost_time = end - start;

    printf("PASSED, computation time GPU_one_grid = %f secs\n", cost_time);

    // Deallocate device memory
    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_out));

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}
