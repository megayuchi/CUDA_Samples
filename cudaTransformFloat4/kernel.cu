#include <Windows.h>
#include <cuda.h>
#include <vector_types.h>
#include <cuda_fp16.h>
#include <helper_math.h>
#include "typedef.h"
#include "CUDA_Util.h"
#include "kernel.cuh"

__constant__ MATRIX4	g_mat;

__host__ __device__ __inline__ void cuTransformVector4(float4* pDest, float4* pSrc, MATRIX4* pMat)
{
	float4 r;
	r.x = pSrc->x * pMat->_11 + pSrc->y * pMat->_21 + pSrc->z * pMat->_31 + pSrc->w * pMat->_41;
	r.y = pSrc->x * pMat->_12 + pSrc->y * pMat->_22 + pSrc->z * pMat->_32 + pSrc->w * pMat->_42;
	r.z = pSrc->x * pMat->_13 + pSrc->y * pMat->_23 + pSrc->z * pMat->_33 + pSrc->w * pMat->_43;
	r.w = pSrc->x * pMat->_14 + pSrc->y * pMat->_24 + pSrc->z * pMat->_34 + pSrc->w * pMat->_44;
	*pDest = r;
}


__global__ void Kernel_TransformVector4(float4* pDest, float4* pSrc, MATRIX4* pMat, DWORD dwVertexNum)
{
	DWORD ThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (ThreadIndex >= dwVertexNum)
		return;
	cuTransformVector4(pDest + ThreadIndex, pSrc + ThreadIndex, pMat);
}
__global__ void Kernel_TransformVector4_ConstMemory(float4* pDest, float4* pSrc, DWORD dwVertexNum)
{
	DWORD ThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	if (ThreadIndex >= dwVertexNum)
		return;

	cuTransformVector4(pDest + ThreadIndex, pSrc + ThreadIndex, &g_mat);
}

__global__ void Kernel_TransformVector4_4Threads(float4* pDest, float4* pSrc, DWORD dwVertexNum)
{
	DWORD ThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
	DWORD dwVertexIndex = ThreadIndex / 4;
	DWORD x = ThreadIndex % 4;
	
	if (dwVertexIndex >= dwVertexNum)
		return;

	float*	pSrc4x1 = (float*)(pSrc + dwVertexIndex);
	float*	pDest4x1 = (float*)(pDest + dwVertexIndex);
	pDest4x1[x] = pSrc4x1[0] * g_mat.f[0][x] + pSrc4x1[1] * g_mat.f[1][x] + pSrc4x1[2] * g_mat.f[2][x] + pSrc4x1[3] * g_mat.f[3][x];
}


void LaunchKernel(float4* pDestDev, float4* pSrcDev, MATRIX4* pMatDev, DWORD dwVertexNum)
{
	const DWORD MAX_VERTEX_NUM_PER_ONCE = 65536 * 10;
	const DWORD THREAD_NUM_PER_BLOCK = 1024;	// 32 - 1024 까지 스레드 개수에 따라 성능 향상

	while (dwVertexNum)
	{
		DWORD dwVertexNumPerOnce = dwVertexNum;
		if (dwVertexNumPerOnce > MAX_VERTEX_NUM_PER_ONCE)
		{
			dwVertexNumPerOnce = MAX_VERTEX_NUM_PER_ONCE;
		}

		dim3	threadPerBlock(1, 1);
		dim3	blockPerGrid(1, 1, 1);
	
		threadPerBlock.x = THREAD_NUM_PER_BLOCK;
		threadPerBlock.y = 1;

		blockPerGrid.x = (dwVertexNumPerOnce / THREAD_NUM_PER_BLOCK) + ((dwVertexNumPerOnce % THREAD_NUM_PER_BLOCK) != 0);
		blockPerGrid.y = 1;

		Kernel_TransformVector4 << < blockPerGrid, threadPerBlock, 0 >> > (pDestDev, pSrcDev, pMatDev, dwVertexNumPerOnce);
		
		VerifyCudaError(cudaThreadSynchronize());
		pDestDev += dwVertexNumPerOnce;
		pSrcDev += dwVertexNumPerOnce;
		dwVertexNum -= dwVertexNumPerOnce;
	}
}

void LaunchKernel_ConstMemory(float4* pDestDev, float4* pSrcDev, MATRIX4* pMatDev, DWORD dwVertexNum)
{
	const DWORD MAX_VERTEX_NUM_PER_ONCE = 65536 * 10;
	const DWORD THREAD_NUM_PER_BLOCK = 1024;	// 32 - 1024 까지 스레드 개수에 따라 성능 향상

	VerifyCudaError(cudaMemcpyToSymbolAsync(g_mat, pMatDev, sizeof(MATRIX4), 0, cudaMemcpyDeviceToDevice));

	while (dwVertexNum)
	{
		DWORD dwVertexNumPerOnce = dwVertexNum;
		if (dwVertexNumPerOnce > MAX_VERTEX_NUM_PER_ONCE)
		{
			dwVertexNumPerOnce = MAX_VERTEX_NUM_PER_ONCE;
		}

		dim3	threadPerBlock(1, 1);
		dim3	blockPerGrid(1, 1, 1);
	
		threadPerBlock.x = THREAD_NUM_PER_BLOCK;
		threadPerBlock.y = 1;

		blockPerGrid.x = (dwVertexNumPerOnce / THREAD_NUM_PER_BLOCK) + ((dwVertexNumPerOnce % THREAD_NUM_PER_BLOCK) != 0);
		blockPerGrid.y = 1;

		Kernel_TransformVector4_ConstMemory <<< blockPerGrid, threadPerBlock, 0 >>> (pDestDev, pSrcDev, dwVertexNumPerOnce);
	
		VerifyCudaError(cudaThreadSynchronize());
		pDestDev += dwVertexNumPerOnce;
		pSrcDev += dwVertexNumPerOnce;
		dwVertexNum -= dwVertexNumPerOnce;
	}
	int a = 0;
}
void LaunchKernel_4Threads_ConstMemory(float4* pDestDev, float4* pSrcDev, MATRIX4* pMatDev, DWORD dwVertexNum)
{
	const DWORD MAX_VERTEX_NUM_PER_ONCE = 65536 * 10;
	const DWORD THREAD_NUM_PER_BLOCK = 1024;	// 32 - 1024 까지 스레드 개수에 따라 성능 향상

	VerifyCudaError(cudaMemcpyToSymbolAsync(g_mat, pMatDev, sizeof(MATRIX4), 0, cudaMemcpyDeviceToDevice));

	while (dwVertexNum)
	{
		DWORD dwVertexNumPerOnce = dwVertexNum;
		if (dwVertexNumPerOnce > MAX_VERTEX_NUM_PER_ONCE)
		{
			dwVertexNumPerOnce = MAX_VERTEX_NUM_PER_ONCE;
		}

		dim3	threadPerBlock(1, 1);
		dim3	blockPerGrid(1, 1, 1);
	
		threadPerBlock.x = THREAD_NUM_PER_BLOCK;
		threadPerBlock.y = 1;

		// 점 1개당 4스레드 필요
		blockPerGrid.x = ((dwVertexNumPerOnce * 4) / THREAD_NUM_PER_BLOCK) + ((dwVertexNumPerOnce % THREAD_NUM_PER_BLOCK) != 0);
		blockPerGrid.y = 1;

		Kernel_TransformVector4_4Threads <<< blockPerGrid, threadPerBlock, 0 >>> (pDestDev, pSrcDev, dwVertexNumPerOnce);
		VerifyCudaError(cudaThreadSynchronize());
		pDestDev += dwVertexNumPerOnce;
		pSrcDev += dwVertexNumPerOnce;
		dwVertexNum -= dwVertexNumPerOnce;
	}
	int a = 0;
}