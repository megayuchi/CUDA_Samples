#include <windows.h>

struct GPU_INFO
{
	char	szDeviceName[256];
	int		sm_per_multiproc;
	int		clock_rate;
	int		multiProcessorCount;
	UINT64	GFlops;
};

enum GPU_SELECT_MODE
{
	FIRST_DEBUG_DEVICE,
	FIRST_MAX_GFLOPS,
	FIRST_DISPLAY_DEVICE
};



BOOL InitCUDA(GPU_SELECT_MODE mode);
struct CELL;

float SearchMaxValue_CUDA_A(CELL* pCellList,DWORD dwCellNum,StopWatchInterface* cudaTimer);
float SearchMaxValue_CUDA_B(CELL* pCellList,DWORD dwCellNum,StopWatchInterface* cudaTimer);