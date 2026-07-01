#include <string.h>
#include <time.h>

#include "controlGui.h"


using namespace sf;


int main()
{
    srand(time(0));

    RenderWindow window(VideoMode(MAX_WIDTH, MAX_HEIGHT), L"Fluid simulation", Style::Default);
    ImGui::SFML::Init(window);

    CudaFluidSim simulationProcess;
    PropertiesPanel propPanel(window, simulationProcess);
    Uint32* pixels = nullptr;


    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        std::cout << "Device error" << std::endl;
        return -1;
    }

    float dt = 1 / 200.f;

    sf::Clock deltaClock;
    while (window.isOpen())
    {
        Event event;
        while (window.pollEvent(event))
        {
            ImGui::SFML::ProcessEvent(window, event);
            if (event.type == Event::Closed)
                window.close();
        }

        ImGui::SFML::Update(window, deltaClock.restart());
        window.clear(Color::Black);

        pixels = propPanel.update(dt);
        simulationProcess.step(pixels, dt);
        propPanel.draw();

        ImGui::SFML::Render(window);
        window.display();

        dt = deltaClock.getElapsedTime().asSeconds();
    }

    ImGui::SFML::Shutdown();
    return 0;
}