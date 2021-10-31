#include "StdAfx.h"
#include <helper_cuda.h>
#include <helper_functions.h>
#include "CPU_Search.h"


void SearchMaxValue_CPU(CELL* pCell)
{
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
float SearchMaxValue_CPU(CELL* pCellList,DWORD dwCellNum,StopWatchInterface* cudaTimer)
{
	sdkResetTimer(&cudaTimer);
	sdkStartTimer(&cudaTimer);
	
	for (DWORD i=0; i<MAX_CELL_NUM; i++)
	{
		SearchMaxValue_CPU(pCellList+i);
	}
	sdkStopTimer(&cudaTimer);
	float ellapsed_time = (float)sdkGetTimerValue(&cudaTimer);
	sdkResetTimer(&cudaTimer);

	return ellapsed_time;
}

	

	