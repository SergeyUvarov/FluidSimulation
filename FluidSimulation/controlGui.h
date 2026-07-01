#pragma once

#include "imgui.h"
#include "imgui-SFML.h"

#include <SFML/Graphics.hpp>
#include <SFML/System.hpp>

#include "simulationCore.h"

#define MAX_WIDTH 1920
#define MAX_HEIGHT 1056

class PropertiesPanel
{
public:
	PropertiesPanel(sf::RenderWindow& mw, CudaFluidSim& simulation);
	~PropertiesPanel();

	void resize();

	Uint32* update(float dt);

	void draw();

protected:
	float fpsStatistic[100];

	sf::RenderWindow& mainWindow;
	CudaFluidSim& sim;
	Properties& prop;
	sf::Texture* txtr;
	sf::Sprite canvas;
	Uint32* pixels;
};