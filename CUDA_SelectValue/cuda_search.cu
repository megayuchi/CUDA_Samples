#include "stdafx.h"
#include "cuda_search.h"
#include "typedef.h"


GPU_INFO	g_GPU_Info;


//#define USE_SHARED_MEMORY



#define THREAD_NUM_PER_BLOCK	256
__global__ void SearchMaxValue_A(CELL* pCell,DWORD dwCellNum);
__global__ void SearchMaxValue_B(CELL* pCellList,DWORD dwCellNum);


BOOL InitCUDA(GPU_SELECT_MODE mode)
{
	BOOL	bResult = FALSE;

	int iDeviceCount = 0;

	int DisplayDevice = -1;
	int	NonDisplayDevice = -1;
	int	MaxGFlopsDeivce = -1;
	int	MaxPCIBusID = 0;
	int	MaxGFlops = 0;
	
	if (cudaSuccess != cudaGetDeviceCount(&iDeviceCount))
	{
		goto lb_return;
	}

	if (!iDeviceCount)
	{
		goto lb_return;
	}
	if (!iDeviceCount)
	{
		goto lb_return;
	}


	cudaDeviceProp prop;
	for (int i=0; i<iDeviceCount; i++)
	{
		if (cudaSuccess != cudaGetDeviceProperties(&prop, i)) 
			__debugbreak();

		if (prop.major < 2) 
			continue;

		if (prop.pciBusID > MaxPCIBusID)
		{
			MaxPCIBusID = prop.pciBusID;
			NonDisplayDevice = i;
		}
		else
		{
			DisplayDevice = i;
		}
		float	clock_rate = (float)prop.clockRate / (1000.0f*1000.0f);
		int		sm_per_multiproc = _ConvertSMVer2Cores(prop.major, prop.minor);
		float	GFlops = (float)(prop.multiProcessorCount * sm_per_multiproc) * clock_rate*2.0f;
		if (GFlops > MaxGFlops)
		{
			MaxGFlops = GFlops;
			MaxGFlopsDeivce = i;
		}
	}
	
	int iSelectedDeviceID = -1;
	
	switch (mode)
	{
	case FIRST_DEBUG_DEVICE:
		iSelectedDeviceID = NonDisplayDevice;
		if (-1 == iSelectedDeviceID)
		{
			// CUDA 2.0이상을 지원하는 디버그용 디바이스가 없다.
			iSelectedDeviceID = MaxGFlopsDeivce;
		}
		break;
	case FIRST_DISPLAY_DEVICE:
		iSelectedDeviceID = DisplayDevice;
		break;

	case FIRST_MAX_GFLOPS:
		iSelectedDeviceID = MaxGFlopsDeivce;
		break;
	}
	if (-1 == iSelectedDeviceID)
	{
		// CUDA 2.0이상을 지원하는 어떤 디바이스도 없다.
		goto lb_return;
	}

	if (cudaSetDevice(iSelectedDeviceID) != cudaSuccess)
		goto lb_return;


	cudaGetDeviceProperties(&prop,iSelectedDeviceID);
	
	strcpy_s(g_GPU_Info.szDeviceName,prop.name);
	
	prop.kernelExecTimeoutEnabled;
	int		sm_per_multiproc;
	if (prop.major == 9999 && prop.minor == 9999)
	{
		sm_per_multiproc = 1;
	}
	else 
	{
		sm_per_multiproc = _ConvertSMVer2Cores(prop.major, prop.minor);
	}

	float	clock_rate = (float)prop.clockRate / (1000.0f*1000.0f);

	g_GPU_Info.sm_per_multiproc = sm_per_multiproc;
	g_GPU_Info.clock_rate = prop.clockRate;
	g_GPU_Info.multiProcessorCount = prop.multiProcessorCount;
	g_GPU_Info.GFlops = (UINT64)prop.multiProcessorCount * (UINT64)sm_per_multiproc * (UINT64)prop.clockRate*2;

	bResult = TRUE;
	
lb_return:
	return bResult;
}

float SearchMaxValue_CUDA_A(CELL* pCellList,DWORD dwCellNum,StopWatchInterface* cudaTimer)
{
	
	char*	pCellDev = NULL;
	
	
	DWORD	dwSize = sizeof(CELL)*dwCellNum;
	
	checkCudaErrors( cudaMalloc((void**) &pCellDev,dwSize));
	checkCudaErrors( cudaMemcpy(pCellDev,pCellList,dwSize,cudaMemcpyHostToDevice));
	

	DWORD	dwBlockNumPerGrid = dwCellNum / THREAD_NUM_PER_BLOCK + ((dwCellNum % THREAD_NUM_PER_BLOCK) != 0);

	dim3	blockPerGrid(dwBlockNumPerGrid,1);
	dim3	threadPerBlock(THREAD_NUM_PER_BLOCK,1);

	sdkResetTimer(&cudaTimer);
	sdkStartTimer(&cudaTimer);

	
	SearchMaxValue_A <<< blockPerGrid,threadPerBlock >>> ((CELL*)pCellDev,dwCellNum);
	checkCudaErrors(cudaDeviceSynchronize());

	sdkStopTimer(&cudaTimer);
	float ellapsed_time = (float)sdkGetTimerValue(&cudaTimer);
	sdkResetTimer(&cudaTimer);

	
	checkCudaErrors( cudaMemcpy(pCellList,pCellDev,dwSize, cudaMemcpyDeviceToHost));

	checkCudaErrors( cudaFree(pCellDev));

	return ellapsed_time;
}
float SearchMaxValue_CUDA_B(CELL* pCellList,DWORD dwCellNum,StopWatchInterface* cudaTimer)
{
	
	char*	pCellDev = NULL;
	
	
	DWORD	dwSize = sizeof(CELL)*dwCellNum;
	
	checkCudaErrors( cudaMalloc((void**) &pCellDev,dwSize));
	checkCudaErrors( cudaMemcpy(pCellDev,pCellList,dwSize,cudaMemcpyHostToDevice));
	

	DWORD	dwBlockNumPerGrid = dwCellNum;

	dim3	blockPerGrid(dwBlockNumPerGrid,1);
	dim3	threadPerBlock(THREAD_NUM_PER_BLOCK,1);

	sdkResetTimer(&cudaTimer);
	sdkStartTimer(&cudaTimer);

	
	SearchMaxValue_B <<< blockPerGrid,threadPerBlock >>> ((CELL*)pCellDev,dwCellNum);
	checkCudaErrors(cudaDeviceSynchronize());

	sdkStopTimer(&cudaTimer);
	float ellapsed_time = (float)sdkGetTimerValue(&cudaTimer);
	sdkResetTimer(&cudaTimer);

	
	checkCudaErrors( cudaMemcpy(pCellList,pCellDev,dwSize, cudaMemcpyDeviceToHost));

	checkCudaErrors( cudaFree(pCellDev));

	return ellapsed_time;
}

__global__ void SearchMaxValue_A(CELL* pCellList,DWORD dwCellNum)
{
	// 블럭당 스레드 총 개수
	DWORD ThreadNumPerBlock = blockDim.x * blockDim.y;

	// 그리드당 블럭 총 개수
	DWORD BlockNumpPerGrid = gridDim.x * gridDim.y;

	// 블럭 인덱스
	DWORD UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;

	// 블럭 내에서의 스레드 인덱스
	DWORD ThreadIndexInBlock = threadIdx.y * blockDim.x + threadIdx.x;

	// 그리드에서의 스레드 인덱스
	DWORD UniqueThreadIndex = UniqueBlockIndex * ThreadNumPerBlock + ThreadIndexInBlock;

	// 그리드에서 스레드 총 개수
	DWORD ThreadNumPerGrid = ThreadNumPerBlock*BlockNumpPerGrid;

		
	if (UniqueThreadIndex >= dwCellNum)
	{
		return;
	}

	CELL*	pCell = pCellList+UniqueThreadIndex;

	float	MaxValue = -999999.0;
	
	for (DWORD i=0; i<pCell->CountOfValue; i++)
	{
		if (MaxValue < pCell->Value[i])
		{
			MaxValue = pCell->Value[i];
		}
	}
	pCell->MaxValue = MaxValue;
}


__global__ void SearchMaxValue_B(CELL* pCellList,DWORD dwCellNum)
{
	__shared__	float	g_SharedMaxValue[THREAD_NUM_PER_BLOCK];
	// 블럭당 스레드 총 개수
	DWORD ThreadNumPerBlock = blockDim.x * blockDim.y;

	// 그리드당 블럭 총 개수
	DWORD BlockNumpPerGrid = gridDim.x * gridDim.y;

	// 블럭 인덱스
	DWORD UniqueBlockIndex = blockIdx.y * gridDim.x + blockIdx.x;

	// 블럭 내에서의 스레드 인덱스
	DWORD ThreadIndexInBlock = threadIdx.y * blockDim.x + threadIdx.x;
			
	if (ThreadIndexInBlock >= dwCellNum)
	{
		return;
	}

	CELL*	pCell = pCellList+UniqueBlockIndex;

	float	MaxValue = -999999.0;
	g_SharedMaxValue[ThreadIndexInBlock] = 999999.0;

	DWORD	CurIndex = ThreadIndexInBlock;
	while (CurIndex < pCell->CountOfValue)
	{
		if (MaxValue < pCell->Value[CurIndex])
		{
			MaxValue = pCell->Value[CurIndex];
		}

		CurIndex += ThreadNumPerBlock;
	}
	g_SharedMaxValue[ThreadIndexInBlock] = MaxValue;

	__syncthreads();

	DWORD	dwHalfThreadCount = THREAD_NUM_PER_BLOCK>>1;
	while (dwHalfThreadCount)
	{
		if (ThreadIndexInBlock < dwHalfThreadCount)
		{
			DWORD	dwTargetIndex = ThreadIndexInBlock+dwHalfThreadCount;
			
			if (g_SharedMaxValue[ThreadIndexInBlock] < g_SharedMaxValue[dwTargetIndex])
			{
				g_SharedMaxValue[ThreadIndexInBlock] = g_SharedMaxValue[dwTargetIndex];
			}
		}
		__syncthreads();

		dwHalfThreadCount = dwHalfThreadCount>>1;
	}
	if (0 == ThreadIndexInBlock)
	{
		pCell->MaxValue = g_SharedMaxValue[0];
	}
}

