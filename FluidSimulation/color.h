#pragma once

#include <SFML/Config.hpp>
#include <device_launch_parameters.h>

#define MIN(a, b) (((a) > (b)) ? (b) : (a))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define LIM(val) (MAX(MIN(val, 1.0f), -1.0f))
#define sRGB(r,g,b) ((0xFF << 24) | (static_cast<Uint8>(b) << 16) | (static_cast<Uint8>(g) << 8) | static_cast<Uint8>(r))

using sf::Uint32;
using sf::Uint8;

struct Color3f
{
	float R = 0.0f;
	float G = 0.0f;
	float B = 0.0f;

	__host__ __device__ Color3f operator+ (const Color3f& other) const
	{
		Color3f res;
		res.R = this->R + other.R;
		res.G = this->G + other.G;
		res.B = this->B + other.B;
		return res;
	}

	__host__ __device__ Color3f& operator+= (const Color3f& other)
	{
		R += other.R;
		G += other.G;
		B += other.B;
		return *this;
	}

	__host__ __device__ Color3f operator* (float d) const
	{
		Color3f res;
		res.R = this->R * d;
		res.G = this->G * d;
		res.B = this->B * d;
		return res;
	}

	__host__ __device__ Uint32 toRGBA() const
	{
		return sRGB(
			Uint8(MAX(0, MIN(255, 255 * R))),
			Uint8(MAX(0, MIN(255, 255 * G))),
			Uint8(MAX(0, MIN(255, 255 * B)))
		);
	}
};