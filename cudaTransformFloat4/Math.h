#pragma once

void TransformVector4(float4* pDest, float4* pSrc, MATRIX4* pMat, DWORD dwNum);
void FillMatrix(MATRIX4* pOutMat);
void FillVector4(float4* pf4Out);
void PrintMatrix(MATRIX4* pMat);
void SetIdentityMatrix(MATRIX4* pOutMat);