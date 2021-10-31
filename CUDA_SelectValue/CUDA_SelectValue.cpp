// CUDA_SelectValue.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <conio.h>
#include "CPU_Search.h"
#include "cuda_search.h"


void PrintResult(CELL* pCellList,DWORD CellNum);

int _tmain(int argc, _TCHAR* argv[])
{
	InitCUDA(FIRST_MAX_GFLOPS);
	//InitCUDA(FIRST_DEBUG_DEVICE);	

	CELL*	pCellList = new CELL[MAX_CELL_NUM];
	memset(pCellList,0,sizeof(CELL)*MAX_ITEM_NUM_PER_CELL);

	for (DWORD i=0; i<MAX_CELL_NUM; i++)
	{
		pCellList[i].CountOfValue = rand()%MAX_ITEM_NUM_PER_CELL + 1;
		for (DWORD j=0; j<pCellList[i].CountOfValue; j++)
		{
			pCellList[i].Value[j] = (float)j;
		}
	}
	StopWatchInterface*	cudaTimer = NULL;
	sdkCreateTimer(&cudaTimer);

	float	elapsed_tick = 0.0f;

	
	
	elapsed_tick = SearchMaxValue_CPU(pCellList,MAX_CELL_NUM,cudaTimer);
	PrintResult(pCellList,MAX_CELL_NUM);
	printf("\n%fms Elapsed.\n",elapsed_tick);
	_getch();

	elapsed_tick = SearchMaxValue_CUDA_A(pCellList,MAX_CELL_NUM,cudaTimer);
	PrintResult(pCellList,MAX_CELL_NUM);
	printf("\n%fms Elapsed.\n",elapsed_tick);
	_getch();
	
	elapsed_tick = SearchMaxValue_CUDA_B(pCellList,MAX_CELL_NUM,cudaTimer);
	PrintResult(pCellList,MAX_CELL_NUM);
	printf("\n%fms Elapsed.\n",elapsed_tick);
	_getch();

	delete [] pCellList;
	pCellList = NULL;

	sdkDeleteTimer(&cudaTimer);
	cudaTimer = NULL;

	return 0;
}

void PrintResult(CELL* pCellList,DWORD CellNum)
{
	for (DWORD i=0; i<CellNum; i++)
	{
		if ((int)pCellList[i].MaxValue != (int)(pCellList[i].CountOfValue-1))
			__debugbreak();

		printf("%d/%u ",(int)pCellList[i].MaxValue,pCellList[i].CountOfValue);
	}

}

