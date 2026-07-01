#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "color.h"



using namespace sf;

#define WIDTH 1920
#define HEIGHT 1056
#define THREAD_BLOCK_SIZE_X 32
#define THREAD_BLOCK_SIZE_Y 8


#define V_DIFFUSE 0.25f
#define C_DIFFUSE 0.25f
#define JACOBI_ITERATION_COUNT 40

#define V_VORTICITY 4.0f
#define V_DENSITY 1.0f


#define GET(from, x, y) from[(y) * WIDTH + (x)]


struct Field
{
	Color3f* color;
	float2* velocity;
	float* pressure;

	void init() {
		cudaMalloc(&color, WIDTH * HEIGHT * sizeof(Color3f));
		cudaMalloc(&velocity, WIDTH * HEIGHT * sizeof(float2));
		cudaMalloc(&pressure, WIDTH * HEIGHT * sizeof(float));
	}

	void initVelocity(float2* initVelocity) {
		cudaMemcpy(velocity, initVelocity, WIDTH * HEIGHT * sizeof(float2), cudaMemcpyHostToDevice);
	}

	void release() {
		cudaFree(color);
		cudaFree(velocity);
		cudaFree(pressure);
	}
};


class CudaFluidSim
{
public:
	CudaFluidSim();
	~CudaFluidSim();

	__host__ void step(Uint32* pixels, float dt);

protected:
	Uint32* pixelField;
	Field field1;
	Field field2;

	float* curlField;
};
