#pragma once

#include <Windows.h>

#define MAX_ITEM_NUM_PER_CELL	8192
#define MAX_CELL_NUM			8192


struct CELL
{
	float	Value[MAX_ITEM_NUM_PER_CELL];
	DWORD	CountOfValue;
	float	MaxValue;
};