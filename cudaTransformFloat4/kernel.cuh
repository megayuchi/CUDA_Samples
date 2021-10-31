#pragma once

void LaunchKernel(float4* pDestDev, float4* pSrcDev, MATRIX4* pMatDev, DWORD dwVertexNum);
void LaunchKernel_ConstMemory(float4* pDestDev, float4* pSrcDev, MATRIX4* pMatDev, DWORD dwVertexNum);
void LaunchKernel_4Threads_ConstMemory(float4* pDestDev, float4* pSrcDev, MATRIX4* pMatDev, DWORD dwVertexNum);