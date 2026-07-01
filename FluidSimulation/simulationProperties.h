#pragma once

struct Properties{
	int width = 1920;
	int height = 1056;
	int threadsBlockSizeX = 32;
	int threadsBlockSizeY = 8;
	int jacobiIterationCount = 40;
	int displayMode = 0;
	int arrowStepSize = 7;
	int vAccuracy = -4;
	int cAccuracy = -7;
	int pAccuracy = -4;

	float scale = 1.0f;
	float colorAmplify = 1.0f;
	float arrowAmplify = 1.0f;

	float velocityDiffuseCoef = 5.0f;
	float colorDiffuseCoef = 10.0f;
	float vorticityForceCoef = 3.0f;
	float densityCoef = 0.5f;
	float dt = 1.0f / 30.0f;
	float gravity = 0.0f;
	
	bool fastJacobi = true;
	bool solidEdges = false;
	bool displayArrow = false;

};