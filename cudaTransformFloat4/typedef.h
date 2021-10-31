#pragma once


//#include <cuda.h>
#include <vector_types.h>
#include <cuda_fp16.h>
//#include <helper_math.h>

struct MATRIX4
{
	union
	{
		struct
		{
			float	_11;
			float	_12;
			float	_13;
			float	_14;

			float	_21;
			float	_22;
			float	_23;
			float	_24;

			float	_31;
			float	_32;
			float	_33;
			float	_34;

			float	_41;
			float	_42;
			float	_43;
			float	_44;
		};
		struct
		{
			float f[4][4];
		};
	};
};