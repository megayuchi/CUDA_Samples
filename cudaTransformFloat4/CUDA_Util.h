#pragma once

enum GPU_SELECT_MODE
{
	FIRST_PCI_BUS_ID,
	LAST_PCI_BUS_ID,
	MAX_GFLOPS,
	MIN_GFLOPS,
	SPECIFY_DEVICE_ID,
	GPU_SELECT_MODE_LAST = SPECIFY_DEVICE_ID
};

struct GPU_INFO
{
	char	szDeviceName[256];
	DWORD	sm_per_multiproc;
	DWORD	clock_rate;
	DWORD	multiProcessorCount;
	float	TFlops;
};


void ReportGPUMemeoryUsage();
void VerifyCudaError(cudaError err);