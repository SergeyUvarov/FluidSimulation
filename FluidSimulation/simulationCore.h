#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <vector>
#include <SFML/Graphics.hpp>
#include "simulationProperties.h"
#include "color.h"

#include <iostream>

struct FluidField
{
	Color3f* color;
	float2* velocity;
	float* pressure;
	float* curl;
	bool* grid;
	int width;
	int height;

	FluidField(): color(nullptr), velocity(nullptr), pressure(nullptr), curl(nullptr), width(0), height(0), grid(nullptr)
	{ }

	__host__ __device__ void updateGrid(bool* gridField)
	{
		grid = gridField;
	}

	__host__ __device__ Color3f getColor(int x, int y, int dx, int dy) const{
		if (x + dx < 0 || x + dx > width - 1 || y + dy < 0 || y + dy > height - 1) {
			Color3f c = color[(y)*width + (x)];
			Color3f c2 = color[(y-dy)*width + (x-dx)];

			if (c.R < c2.R) c.R *= 0.9f;
			if (c.G < c2.G) c.G *= 0.9f;
			if (c.B < c2.B) c.B *= 0.9f;

			return c;
		}
		else if (grid && grid[(y + dy) * width + (x + dx)]) {
			Color3f c = color[(y)*width + (x)];
			Color3f c2 = color[(y - dy) * width + (x - dx)];

			if (c.R < c2.R) c.R *= 0.9f;
			if (c.G < c2.G) c.G *= 0.9f;
			if (c.B < c2.B) c.B *= 0.9f;

			return c;
		}
		return color[(y + dy) * width + (x + dx)];
	}

	__host__ __device__ float2 getVelocity(int x, int y, int dx, int dy) const {
		if (x + dx < 0)			return make_float2(-abs(velocity[(y)*width + (x)].x), velocity[(y)*width + (x)].y);
		else if (x + dx > width - 1) return make_float2(abs(velocity[(y)*width + (x)].x), velocity[(y)*width + (x)].y);
		else if (y + dy < 0)			return make_float2(velocity[(y)*width + (x)].x, -abs(velocity[(y)*width + (x)].y));
		else if (y + dy > height - 1)return make_float2(velocity[(y)*width + (x)].x, abs(velocity[(y)*width + (x)].y));

		//if (x + dx < 0 || x + dx > width - 1 || y + dy < 0 || y + dy > height - 1) return make_float2(velocity[(y)*width + (x)].x * 0.9f, velocity[(y)*width + (x)].y * 0.9f);
		else if (grid && grid[(y)*width + (x + dx)]) return make_float2(0.0f, 0.0f);
		else if (grid && grid[(y + dy) * width + (x)]) return make_float2(0.0f, 0.0f);
		/*else if (grid && grid[(y) * width + (x + dx)]) return make_float2(-velocity[(y)*width + (x)].x, velocity[(y)*width + (x)].y);
		else if (grid && grid[(y + dy) * width + (x)]) return make_float2(velocity[(y)*width + (x)].x, -velocity[(y)*width + (x)].y);*/
		else return velocity[(y + dy) * width + (x + dx)];
	}

	__host__ __device__ float getPressure(int x, int y, int dx, int dy) const {
		if (x + dx < 0 || x + dx > width - 1 || y + dy < 0 || y + dy > height - 1) return 0.0f;
		else if (grid && grid[(y + dy) * width + (x + dx)]) return pressure[(y)*width + (x)];
		return pressure[(y + dy) * width + (x + dx)];
	}

	__host__ __device__ float getCurl(int x, int y) const {
		if (x < 0 || y < 0 || x > width - 1 || y > height - 1) return 0.0f;
		else return curl[y * width + x];
	}

	__host__ void transform(const Properties& prop) {
		release();

		width = prop.width;
		height = prop.height;

		cudaMalloc(&color, width * height * sizeof(Color3f));
		cudaMalloc(&velocity, width * height * sizeof(float2));
		cudaMalloc(&pressure, width * height * sizeof(float));
		cudaMalloc(&curl, width * height * sizeof(float));
	}

	__host__ void setVelocity(float2* newVelocityField) {
		cudaMemcpy(velocity, newVelocityField, width * height * sizeof(float2), cudaMemcpyHostToDevice);
	}

	__host__ void setColor(Color3f* newColorField) {
		cudaMemcpy(color, newColorField, width * height * sizeof(Color3f), cudaMemcpyHostToDevice);
	}

	__host__ void setPressure(float* newPressureField) {
		cudaMemcpy(pressure, newPressureField, width * height * sizeof(float), cudaMemcpyHostToDevice);
	}

	__host__ void release() {
		cudaFree(color);
		cudaFree(velocity);
		cudaFree(pressure);
		cudaFree(curl);

		width = height = 0;
	}
};


struct SolidObject
{
	float x;
	float y;

	int width;
	int height;

	float scale;
	float angle;

	Uint32* texture;
	bool* grid;

	void load(const std::string& filePath);
	void clear();
};

struct FlowSource
{
	Color3f color;

	float x = 0.04f;
	float y = 0.5f;

	float size = 50.0f;
	float angle = 0.0f;
	float strength = 10.0f;
	float rainbowSpeed = 2.0f;

	bool isRainbow = true;
};


class CudaFluidSim
{
public:
	CudaFluidSim();
	~CudaFluidSim();

	void setRunStatus(bool status);
	bool isRun() const;
	void restart();
	void setProperties(const Properties& newProp);

	void loadSolidObject(const std::string& filePath);
	bool isSolidObject() const;
	SolidObject& getSolidObject();
	std::vector<FlowSource>& getFlowSource();


	float getElapsedTime() const;
	Properties& getProperties();


	float step(Uint32* pixels, float dt);

protected:
	FluidField field1;
	FluidField field2;
	SolidObject solidObj;
	std::vector<FlowSource> flowSources;

	Properties simProp;
	bool* solidGrid;

	Uint32* pixelsField = nullptr;
	float elapsedTime = 0.0f;
	bool pause = false;
};


__global__ void jacobi(FluidField field, FluidField oldField, Properties prop, float dt,
	float alpha_v, float beta_v, float alpha_c, float beta_c, int gItr);
__global__ void advect(FluidField field, FluidField oldField, Properties prop, float dt);
__global__ void computeCurl(FluidField field, Properties prop);
__global__ void vorticity(FluidField field, FluidField oldField, Properties prop, float dt);
__global__ void computePressure(FluidField field, FluidField oldField, Properties prop);
__global__ void clearField(FluidField field, Properties prop);
__global__ void applyInteractive(Uint32* pixels, FluidField field, FlowSource* flowSources, int sourcesCount, Properties prop, float dt, float t);
__global__ void processingSolidObjects(Uint32* pixels, FluidField field, Properties prop, SolidObject obj, float dt, bool* solidGrid);
__global__ void draw(Uint32* pixels, FluidField field, Properties prop);