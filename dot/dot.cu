
//dot
//(x1,x2,x3)(y1,y2,y3)=x1y1+x2y2+x3y3
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "util.h"
#define imin(a,b) (a<b?a:b)
const int N = 33 * 1024;

// ntheads = 2^i for reduction.
const int ntheads=256;
const int nblocks=imin( 32, (N+ntheads-1) / ntheads );


__global__ void dot(float *out, float *a, float *b, int n)
{
    int tid = global_thread_id();
    //if not use multiblocks, stride = block_size() is ok
    int stride = block_size() * block_num();
    int ltid = local_thread_id();
    __shared__ float block_cache[ntheads];
    float thead_sum = 0;
    for(int i = tid; i < N; i+= stride)
    {
        thead_sum += a[i] *b[i];
    }
    block_cache[ltid]=thead_sum;
    __syncthreads();


    //reduction
    //for example
    //if #0-7  blocks 
    //so we should do log2(8)=3time  
    // reduction step1:-----------i=4
    //thread 0: add 0+4 -->0
    //thread 1: add 1+5 -->1
    //thread 2: add 2+6 -->2
    //thread 3: add 3+7 -->3
    // redcution step2:-----------i=2
    //thread 0: add 0+2 -->0
    //thread 1: add 1+3 -->3
    // reduction step3:-----------i=1
    //thread 0: add 0+1 -->0
    //reduction step3:: i==0 resutn 0
    //this will return the current block sum, the dot result should add all block sum

    int i = block_size()/2;
    while(i!=0)
    {
            if(ltid<i)
                block_cache[ltid]+= block_cache[ltid+i];
            __syncthreads();
            i/=2;
    }
    if (ltid ==0 )
        out[block_id()]=block_cache[0];

    //----------------not use reduction--------------
    // if(ltid==0)
    // {   float result =0;
    //     for(int i =0;i<256;i++)
    //     {
    //         result += block_cache[i];
    //     }
    //     out[block_id()]= result;
    //     // out[blockIdx.x ]=block_cache[0];
    // }
    //----------------end not use reduction--------------

}
int main(){ 
    float *a, *b, *out;
    float *d_a, *d_b, *d_out; 
    printf("block is %d\n",nblocks);
    printf("nthead is %d\n",ntheads);
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
    

    dot<<<nblocks,ntheads>>>(d_out,d_a,d_b,N);
    gpuErrchk(cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost));

    float result = 0;
    for(int i = 0; i < nblocks; i++){  
        printf("%f\n",out[i])  ;
        result += out[i];
        // assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

     printf("out = %f\n", result);
    // Deallocate device memory
    gpuErrchk(cudaFree(d_a));
    gpuErrchk(cudaFree(d_b));
    gpuErrchk(cudaFree(d_out));

    // Deallocate host memory
    free(a); 
    free(b); 
    free(out);
}