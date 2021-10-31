#include "pch.h"
#include "typedef.h"
#include "Math.h"


void TransformVector4(float4* pDest, float4* pSrc, MATRIX4* pMat, DWORD dwNum)
{
	float4 r;
	for (DWORD i = 0; i < dwNum; i++)
	{
		r.x = pSrc->x * pMat->_11 + pSrc->y * pMat->_21 + pSrc->z * pMat->_31 + pSrc->w * pMat->_41;
		r.y = pSrc->x * pMat->_12 + pSrc->y * pMat->_22 + pSrc->z * pMat->_32 + pSrc->w * pMat->_42;
		r.z = pSrc->x * pMat->_13 + pSrc->y * pMat->_23 + pSrc->z * pMat->_33 + pSrc->w * pMat->_43;
		r.w = pSrc->x * pMat->_14 + pSrc->y * pMat->_24 + pSrc->z * pMat->_34 + pSrc->w * pMat->_44;
		*pDest = r;

		pSrc++;
		pDest++;
	}
}
void FillMatrix(MATRIX4* pOutMat)
{
	SetIdentityMatrix(pOutMat);

	for (DWORD y = 0; y < 4; y++)
	{
		for (DWORD x = 0; x < 4; x++)
		{
			pOutMat->f[y][x] = (float)((rand() % 10) + 1) / 10.0f;
		}
	}
}
void FillVector4(float4* pf4Out)
{
	float* pf = &pf4Out->x;
	for (DWORD i = 0; i < 4; i++)
	{
		pf[i] = (float)((rand() % 10) + 1) / 10.0f;
	}
}
void PrintMatrix(MATRIX4* pMat)
{
	for (DWORD y = 0; y < 4; y++)
	{
		wprintf_s(L"%.2f %.2f %.2f %.2f\n", pMat->f[y][0], pMat->f[y][1], pMat->f[y][2], pMat->f[y][3]);
	}
}


void SetIdentityMatrix(MATRIX4* pOutMat)
{
	pOutMat->_12 = pOutMat->_13 = pOutMat->_14 = pOutMat->_21 = pOutMat->_23 = pOutMat->_24 = 0.0f;
	pOutMat->_31 = pOutMat->_32 = pOutMat->_34 = pOutMat->_41 = pOutMat->_42 = pOutMat->_43 = 0.0f;
	pOutMat->_11 = pOutMat->_22 = pOutMat->_33 = pOutMat->_44 = 1.0f;
}