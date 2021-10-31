// cudaTransformFloat4.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include <Windows.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <helper_functions.h> // helper utility functions 
#include <helper_cuda.h>      // helper functions for CUDA error checking and initialization
#include "CUDA_Util.h"
#include "Math.h"
#include "QueryPerfCounter.h"
#include "kernel.cuh"
#include <conio.h>

BOOL InitCUDA(GPU_SELECT_MODE mode, int iSpecifiedDeivceID);
void CleanupCUDA();

int		g_iSelectedDeviceID = -1;
GPU_INFO	g_GPU_Info = {};
BOOL		g_bCanPrefetch = FALSE;

void Test();
int main()
{
#ifdef _DEBUG
	int	flag = _CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF;
	_CrtSetDbgFlag(flag);
#endif
	QCInit();

	srand(GetTickCount());

	InitCUDA(SPECIFY_DEVICE_ID, 0);

	Test();

	CleanupCUDA();

	wprintf_s(L"Press any key.\n");
	_getch();
#ifdef _DEBUG
	_ASSERT(_CrtCheckMemory());
#endif

}


BOOL InitCUDA(GPU_SELECT_MODE mode, int iSpecifiedDeivceID)
{
	QCInit();

	BOOL	bResult = FALSE;

	int iDeviceCount = 0;

	if (cudaSuccess != cudaGetDeviceCount(&iDeviceCount))
	{
		goto lb_return;
	}
	struct CUDA_DEVICE_PROPERTY
	{
		cudaDeviceProp prop;
		int iDeviceID;
	};
	enum CUDA_DEVICE_TYPE
	{
		CUDA_DEVICE_FIRST_PCI_BUS_ID,
		CUDA_DEVICE_LAST_PCI_BUS_ID,
		CUDA_DEVICE_MAX_GFLOPS,
		CUDA_DEVICE_MIN_GFLOPS,
		CUDA_DEVICE_SPECIFIED
	};
	const DWORD CUDA_DEVICE_TYPE_NUM = 5;
	CUDA_DEVICE_PROPERTY	devProp[CUDA_DEVICE_TYPE_NUM] = {};

	for (DWORD i = 0; i<CUDA_DEVICE_TYPE_NUM; i++)
	{
		devProp[i].iDeviceID = -1;
	}

	if (!iDeviceCount)
	{
		goto lb_return;
	}
	int	LastPCIBusID = -1;
	int FirstPCIBusID = INT_MAX;
	int	MaxGFlops = -1;
	int MinGFlops = INT_MAX;
		
	cudaDeviceProp prop;
	for (int i = 0; i < iDeviceCount; i++)
	{
		if (cudaSuccess != cudaGetDeviceProperties(&prop, i))
			__debugbreak();

		if (prop.major < 2)
			continue;
	
		if (i == iSpecifiedDeivceID)
		{
			devProp[CUDA_DEVICE_SPECIFIED].prop = prop;
			devProp[CUDA_DEVICE_SPECIFIED].iDeviceID = i;
		}
		if (prop.pciBusID > LastPCIBusID)
		{
			LastPCIBusID = prop.pciBusID;
			devProp[CUDA_DEVICE_LAST_PCI_BUS_ID].prop = prop;
			devProp[CUDA_DEVICE_LAST_PCI_BUS_ID].iDeviceID = i;
		}
		if (prop.pciBusID < FirstPCIBusID)
		{
			FirstPCIBusID = prop.pciBusID;
			devProp[CUDA_DEVICE_FIRST_PCI_BUS_ID].prop = prop;
			devProp[CUDA_DEVICE_FIRST_PCI_BUS_ID].iDeviceID = i;
		}

		float	clock_rate = (float)prop.clockRate / (1000.0f*1000.0f);
		int		sm_per_multiproc = _ConvertSMVer2Cores(prop.major, prop.minor);
		float	GFlops = (float)(prop.multiProcessorCount * sm_per_multiproc) * clock_rate*2.0f;
		if (GFlops > MaxGFlops)
		{
			MaxGFlops = (int)GFlops;
			devProp[CUDA_DEVICE_MAX_GFLOPS].prop = prop;
			devProp[CUDA_DEVICE_MAX_GFLOPS].iDeviceID = i;
		}
		if (GFlops < MinGFlops)
		{
			MinGFlops = (int)GFlops;
			devProp[CUDA_DEVICE_MIN_GFLOPS].prop = prop;
			devProp[CUDA_DEVICE_MIN_GFLOPS].iDeviceID = i;
		}
		/*
		// 일단 TCC모드면 디스플레이 어댑터가 아닌건 확실.
		if (prop.tccDriver)
		{
			NonDisplayDevice = i;
		}
		else
		{
			DisplayDevice = i;
		}
		*/
	}
	//(cudaDeviceProp)deviceProp.concurrentManagedAccess

	int iSelectedDeviceID = -1;
	switch (mode)
	{
		case FIRST_PCI_BUS_ID:
			iSelectedDeviceID = devProp[CUDA_DEVICE_FIRST_PCI_BUS_ID].iDeviceID;
			break;
		case LAST_PCI_BUS_ID:
			iSelectedDeviceID = devProp[CUDA_DEVICE_LAST_PCI_BUS_ID].iDeviceID;
			break;
		case MAX_GFLOPS:
			iSelectedDeviceID = devProp[CUDA_DEVICE_MAX_GFLOPS].iDeviceID;
			break;
		case MIN_GFLOPS:
			iSelectedDeviceID = devProp[CUDA_DEVICE_MIN_GFLOPS].iDeviceID;
			break;
		case SPECIFY_DEVICE_ID:
			iSelectedDeviceID = devProp[CUDA_DEVICE_SPECIFIED].iDeviceID;
			break;
	}
	if (-1 == iSelectedDeviceID)
	{
		// CUDA 2.0이상을 지원하는 어떤 디바이스도 없다.
		goto lb_return;
	}

	if (cudaSetDevice(iSelectedDeviceID) != cudaSuccess)
		goto lb_return;


	g_iSelectedDeviceID = iSelectedDeviceID;

	cudaGetDeviceProperties(&prop, iSelectedDeviceID);

	strcpy_s(g_GPU_Info.szDeviceName, prop.name);

	prop.kernelExecTimeoutEnabled;
	DWORD		sm_per_multiproc;
	if (prop.major == 9999 && prop.minor == 9999)
	{
		sm_per_multiproc = 1;
	}
	else
	{
		sm_per_multiproc = (DWORD)_ConvertSMVer2Cores(prop.major, prop.minor);
	}
	g_GPU_Info.sm_per_multiproc = (DWORD)sm_per_multiproc;
	g_GPU_Info.clock_rate = (DWORD)prop.clockRate;
	g_GPU_Info.multiProcessorCount = (DWORD)prop.multiProcessorCount;
	UINT64 KFlops = (UINT64)(DWORD)prop.multiProcessorCount * (UINT64)(DWORD)sm_per_multiproc * (UINT64)(DWORD)prop.clockRate * 2;
	g_GPU_Info.TFlops = (float)KFlops / (float)(1024 * 1024 * 1024);
	g_bCanPrefetch = prop.concurrentManagedAccess != 0;

	bResult = TRUE;

lb_return:
	return bResult;
}

void CleanupCUDA()
{
	VerifyCudaError(cudaDeviceReset());
}
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file

#pragma optimize( "gpsy",off)
void Test()
{
	const DWORD VERTEX_COUNT = 1024 * 1024 * 100;

	InitCUDA(SPECIFY_DEVICE_ID, 0);

	MATRIX4* pMatHost = nullptr;
	MATRIX4* pMatDev = nullptr;
	cudaMallocHost(&pMatHost, sizeof(MATRIX4));
	FillMatrix(pMatHost);

	cudaMalloc(&pMatDev, sizeof(MATRIX4));
	cudaMemcpy(pMatDev, pMatHost, sizeof(MATRIX4), cudaMemcpyHostToDevice);

	const DWORD SAMPLE_VERTEX_COUNT = 1024;
	float4*		pSampleVertexList = new float4[SAMPLE_VERTEX_COUNT];
	for (DWORD i = 0; i < SAMPLE_VERTEX_COUNT; i++)
	{
		FillVector4(pSampleVertexList + i);
	}

	float4*	pSrcVertexListHost = nullptr;
	float4*	pDestVertexListHost = nullptr;
	float4*	pSrcVertexListDev = nullptr;
	float4*	pDestVertexListDev = nullptr;
	cudaMallocHost(&pSrcVertexListHost, sizeof(float4) * VERTEX_COUNT);
	cudaMallocHost(&pDestVertexListHost, sizeof(float4) * VERTEX_COUNT);
	cudaMalloc(&pSrcVertexListDev, sizeof(float4) * VERTEX_COUNT);
	cudaMalloc(&pDestVertexListDev, sizeof(float4) * VERTEX_COUNT);

	// cuda 
	for (DWORD i = 0; i < VERTEX_COUNT; i++)
	{
		pSrcVertexListHost[i] = pSampleVertexList[i % SAMPLE_VERTEX_COUNT];
	}


	cudaMemcpy(pSrcVertexListDev, pSrcVertexListHost, sizeof(float4) * VERTEX_COUNT, cudaMemcpyHostToDevice);

	LARGE_INTEGER	PrvCounter = QCGetCounter();

	// kernel 호출

	// only global memory
	//LaunchKernel(pDestVertexListDev, pSrcVertexListDev, pMatDev, VERTEX_COUNT);				

	// using constant memory
	LaunchKernel_ConstMemory(pDestVertexListDev, pSrcVertexListDev, pMatDev, VERTEX_COUNT);

	// 4 threads per a vertex , using constant memory
	//LaunchKernel_4Threads_ConstMemory(pDestVertexListDev, pSrcVertexListDev, pMatDev, VERTEX_COUNT);

	float fElapsedTick = QCMeasureElapsedTick(QCGetCounter(), PrvCounter);

	cudaMemcpy(pDestVertexListHost, pDestVertexListDev, sizeof(float4) * VERTEX_COUNT, cudaMemcpyDeviceToHost);
	
	wprintf_s(L"CUDA - ([0] : %.2f,%.2f,%.2f,%.2f) ~ ([1] : %.2f,%.2f,%.2f,%.2f) , %.4f ms elapsed\n",
			  pDestVertexListHost[0].x, pDestVertexListHost[0].y, pDestVertexListHost[0].z, pDestVertexListHost[0].w,
			  pDestVertexListHost[VERTEX_COUNT - 1].x, pDestVertexListHost[VERTEX_COUNT - 1].y, pDestVertexListHost[VERTEX_COUNT - 1].z, pDestVertexListHost[VERTEX_COUNT - 1].w,
			  fElapsedTick);

	// cpu
	PrvCounter = QCGetCounter();

	TransformVector4(pDestVertexListHost, pSrcVertexListHost, pMatHost, VERTEX_COUNT);

	fElapsedTick = QCMeasureElapsedTick(QCGetCounter(), PrvCounter);
	wprintf_s(L"CPU - ([0] : %.2f,%.2f,%.2f,%.2f) ~ ([1] : %.2f,%.2f,%.2f,%.2f) , %.4f ms elapsed\n",
			  pDestVertexListHost[0].x, pDestVertexListHost[0].y, pDestVertexListHost[0].z, pDestVertexListHost[0].w,
			  pDestVertexListHost[VERTEX_COUNT - 1].x, pDestVertexListHost[VERTEX_COUNT - 1].y, pDestVertexListHost[VERTEX_COUNT - 1].z, pDestVertexListHost[VERTEX_COUNT - 1].w,
			  fElapsedTick);

	
	// cleanup
	if (pSampleVertexList)
	{
		delete[] pSampleVertexList;
		pSampleVertexList = nullptr;
	}
	if (pMatHost)
	{
		cudaFreeHost(pMatHost);
		pMatHost = nullptr;
	}
	if (pMatDev)
	{
		cudaFree(pMatDev);
		pMatDev = nullptr;
	}
	if (pSrcVertexListHost)
	{
		cudaFreeHost(pSrcVertexListHost);
		pSrcVertexListHost = nullptr;
	}
	if (pDestVertexListHost)
	{
		cudaFreeHost(pDestVertexListHost);
		pDestVertexListHost = nullptr;
	}
	if (pSrcVertexListDev)
	{
		cudaFree(pSrcVertexListDev);
		pSrcVertexListDev = nullptr;
	}
	if (pDestVertexListDev)
	{
		cudaFree(pDestVertexListDev);
		pDestVertexListDev = nullptr;
	}
	CleanupCUDA();
}
#pragma optimize( "gpsy",on)
