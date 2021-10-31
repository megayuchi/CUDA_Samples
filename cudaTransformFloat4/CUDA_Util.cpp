#include "pch.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <helper_functions.h> // helper utility functions 
#include <helper_cuda.h>      // helper functions for CUDA error checking and initialization
#include "WriteDebugString.h"
#include "CUDA_Util.h"

void ReportGPUMemeoryUsage()
{
	size_t free_byte = 0;
	size_t total_byte = 0;
	cudaError cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
	WriteDebugStringW(DEBUG_OUTPUT_TYPE_DEBUG_CONSOLE, L"Cuda Memory : %I64u / %I64u\n", free_byte, total_byte);
}
void VerifyCudaError(cudaError err)
{
	WCHAR*	wchErr = L"Unknown";
	switch (err)
	{
		case cudaErrorMemoryAllocation:
			wchErr = L"cudaErrorMemoryAllocation";
			break;

		case cudaErrorLaunchFailure:
			wchErr = L"cudaErrorLaunchFailure";
			break;
		case cudaErrorLaunchTimeout:
			wchErr = L"cudaErrorLaunchTimeout";
			break;

		case cudaErrorMisalignedAddress:
			wchErr = L"cudaErrorMisalignedAddress";
			break;
		case cudaErrorInvalidValue:
			wchErr = L"cudaErrorInvalidValue";
			break;

	}
	if (err != cudaSuccess)
	{
		WriteDebugStringW(DEBUG_OUTPUT_TYPE_DEBUG_CONSOLE, L"cuda error = %s(%d)\n", wchErr, err);
		__debugbreak();
	}
}