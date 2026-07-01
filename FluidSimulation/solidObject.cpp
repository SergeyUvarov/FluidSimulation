#include "simulationCore.h"

void SolidObject::load(const std::string& filePath)
{
	clear();

	sf::Image image;
	if (!image.loadFromFile(filePath)) return;

	sf::Vector2u size = image.getSize();
	width = size.x; height = size.y;
	
	cudaMalloc(&grid, width * height * sizeof(bool));
	cudaMalloc(&texture, width * height * sizeof(Uint32));


	bool* tmpGrid = new bool[width * height];
	Uint32* tmpTexture = new Uint32[width * height];
	float2* tmpNorm = new float2[width * height];


	for(int x = 0; x < width; x++)
		for (int y = 0; y < height; y++)
		{
			tmpTexture[y * width + x] = sRGB(image.getPixel(x, y).r, image.getPixel(x, y).g, image.getPixel(x, y).b);
			tmpGrid[y * width + x] = image.getPixel(x, y).a >= 256 / 2;
			tmpNorm[y * width + x] = make_float2(0.0f, 0.0f);
		}



	cudaMemcpy(grid, tmpGrid, width * height * sizeof(bool), cudaMemcpyHostToDevice);
	cudaMemcpy(texture, tmpTexture, width * height * sizeof(Uint32), cudaMemcpyHostToDevice);
	

	delete[] tmpGrid;
	delete[] tmpTexture;

}


void SolidObject::clear()
{
	if (!grid) cudaFree(grid);
	if (!texture) cudaFree(texture);

	width = height = 0;
	x = y = 0.0f;
	angle = 0.0f;
	scale = 1.0f;
}
