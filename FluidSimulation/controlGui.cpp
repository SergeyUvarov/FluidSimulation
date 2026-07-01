#include "controlGui.h"

PropertiesPanel::PropertiesPanel(sf::RenderWindow& mw, CudaFluidSim& simulation): 
    mainWindow(mw), sim(simulation), prop(simulation.getProperties()), pixels(nullptr), txtr(nullptr)
{
    resize();
}

PropertiesPanel::~PropertiesPanel()
{
    if(!pixels)
        delete[] pixels;
    if (!txtr)
        delete txtr;
}

void PropertiesPanel::resize()
{
    delete[] pixels;
    delete txtr;
    pixels = new Uint32[prop.width * prop.height];
    txtr = new sf::Texture;
    txtr->create(prop.width, prop.height);
    canvas.setTexture(*txtr, true);
    canvas.setScale(1.0f / prop.scale, 1.0f / prop.scale);
    sim.restart();
}

Uint32* PropertiesPanel::update(float dt)
{
    static char path[1024];
    static char currentSolidObj[1024];
    static float timer = 0.0f;
    static int flowIdx = 0;

    char flowSourceList[16][4];
    static char* currentSource = flowSourceList[flowIdx];
    

    if (timer <= 0.0f)
    {
        for (int i = 0; i < 100-1; i++) fpsStatistic[i] = fpsStatistic[i + 1];
        fpsStatistic[99] = 1.0f / dt;
        timer = 0.2f;
    }
    timer -= dt;

    ImGui::Begin("Params");
    ImGui::SetWindowFontScale(1.6f);

    if (ImGui::Button("Restart")) {
        sim.restart();
    }

    ImGui::SameLine();

    if (sim.isRun())
    {
        if (ImGui::Button("Pause")) {
            mainWindow.setFramerateLimit(30);
            sim.setRunStatus(false);
        }
    }
    else
    {
        if (ImGui::Button("Run")) {
            sim.setRunStatus(true);
            mainWindow.setFramerateLimit(10000);

        }
    }

    ImGui::Spacing();

    if (ImGui::CollapsingHeader("Statistic"))
    {
        ImGui::Text("FPS: %d", int(1.0f / dt));
        ImGui::Text("Frame render time: %.4fs", dt);
        ImGui::Text("Elapsed time: %.3fs", sim.getElapsedTime());

        ImGui::PlotLines("FPS", fpsStatistic, 100, 0, "", 0.0f, 150.f, ImVec2(0, 80.0f));

        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader("Global"))
    {
        ImGui::Text("Width: %d", prop.width);
        ImGui::Text("Height: %d", prop.height);

        if (ImGui::SliderFloat("Scale", &prop.scale, 0.1f, 4.0f, "%.1f")) {
            prop.width = MAX_WIDTH * prop.scale;
            prop.height = MAX_HEIGHT * prop.scale;
            resize();
        }

        ImGui::Separator();

        if (prop.dt <= 0) {
            ImGui::SliderFloat("Time step", &prop.dt, -0.00001f, 0.1f, "Real time");
        }
        else {
            ImGui::SliderFloat("Time step", &prop.dt, -0.00001f, 0.1f, "%.4f s");
        }
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader("Visual"))
    {
        if (prop.displayMode == 0)
            ImGui::Text("Color display mode");
        else if (prop.displayMode == 1)
            ImGui::Text("Pressure display mode");
        else if (prop.displayMode == 2)
            ImGui::Text("Velocity display mode");

        if (ImGui::Button("Change display mode")) {
            prop.displayMode = (prop.displayMode + 1) % 3;
        }

        ImGui::SliderFloat("Color amplify", &prop.colorAmplify, 0.001f, 2.0f, "%.3f");

        ImGui::Separator();

        ImGui::Checkbox("Velocity arrows", &prop.displayArrow);

        if (prop.displayArrow) {
            ImGui::SliderFloat("Arrow amplify", &prop.arrowAmplify, 0.001f, 10.0f, "%.3f");
            ImGui::SliderInt("Arrow step", &prop.arrowStepSize, 2, 100);
        }

        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader("Compute"))
    {
        ImGui::SliderInt("Threads X", &prop.threadsBlockSizeX, 1, 64);
        ImGui::SliderInt("Threads Y", &prop.threadsBlockSizeY, 1, 64);

        ImGui::Separator();

        ImGui::SliderInt("Jacobi iteration count", &prop.jacobiIterationCount, 1, 1000);
        ImGui::SliderInt("Velocity accuracy", &prop.vAccuracy, -10, 2, "1E%+d");
        ImGui::SliderInt("Color accuracy", &prop.cAccuracy, -10, 2, "1E%+d");
        ImGui::SliderInt("Pressure accuracy", &prop.pAccuracy, -10, 2, "1E%+d");

        ImGui::Checkbox("Use fast Jacobi", &prop.fastJacobi);
        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader("Fluid"))
    {
        ImGui::SliderFloat("V-diffuse", &prop.velocityDiffuseCoef, 0.1f, 10.0f, "%.2f");
        ImGui::SliderFloat("C-diffuse", &prop.colorDiffuseCoef, 0.1f, 30.0f, "%.2f");
        ImGui::SliderFloat("Density", &prop.densityCoef, 0.1f, 10.0f, "%.2f");
        ImGui::SliderFloat("Vorticity", &prop.vorticityForceCoef, 0.0f, 20.0f, "%.2f");

        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader("Flow sources"))
    {
        for (int n = 0; n < sim.getFlowSource().size(); n++)
        {
            sprintf_s(flowSourceList[n], 4, "%d", n + 1);
        }


        if (ImGui::BeginCombo("Source ID", currentSource))
        {
            for (int n = 0; n < sim.getFlowSource().size(); n++)
            {
                bool is_selected = (currentSource == flowSourceList[n]);
                if (ImGui::Selectable(flowSourceList[n], is_selected)) {
                    currentSource = flowSourceList[n];
                    flowIdx = n;
                }
                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }
            ImGui::EndCombo();
        }

        if (ImGui::Button("Add")) {
            if (sim.getFlowSource().size() < 8)
            {
                flowIdx = sim.getFlowSource().size();
                currentSource = flowSourceList[flowIdx];
                sim.getFlowSource().push_back(FlowSource());
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Delete")) {
            if (sim.getFlowSource().size() > 1)
            {
                sim.getFlowSource().erase(sim.getFlowSource().begin() + flowIdx, sim.getFlowSource().begin() + flowIdx + 1);

                if (flowIdx >= sim.getFlowSource().size())
                    flowIdx = sim.getFlowSource().size() - 1;

                currentSource = flowSourceList[flowIdx];
            }
        }


        ImGui::SliderFloat("Pos X##", &sim.getFlowSource()[flowIdx].x, -0.5f, 1.5f);
        ImGui::SliderFloat("Pos Y##", &sim.getFlowSource()[flowIdx].y, -0.5f, 1.5f);

        ImGui::SliderFloat("Size", &sim.getFlowSource()[flowIdx].size, 0.01f, 1000.0f, "%.3f");
        ImGui::SliderAngle("Angle##", &sim.getFlowSource()[flowIdx].angle);
        ImGui::SliderFloat("strength", &sim.getFlowSource()[flowIdx].strength, 0.0f, 50.0f, "%.2f");


        ImGui::Checkbox("Dinamic color", &sim.getFlowSource()[flowIdx].isRainbow);

        if (sim.getFlowSource()[flowIdx].isRainbow)
        {
            ImGui::SliderFloat("Speed", &sim.getFlowSource()[flowIdx].rainbowSpeed, 0.0f, 100.0f, "%.2f");
        }
        else
        {
            ImGui::ColorEdit3("Color", &sim.getFlowSource()[flowIdx].color.R);
        }

        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader("Solid objects"))
    {
        ImGui::InputText("Path to .png", path, 1023);
        if (ImGui::Button("Add solid object")) {
            sim.loadSolidObject(path);
            memcpy(currentSolidObj, path, 1024 * sizeof(char));
            memset(path, 0, 1024 * sizeof(char));

        }

        if (sim.isSolidObject()) {
            ImGui::Text("Object: %s", currentSolidObj);

            ImGui::SliderFloat("Pos X", &sim.getSolidObject().x, -0.5f, 1.5f);
            ImGui::SliderFloat("Pos Y", &sim.getSolidObject().y, -0.5f, 1.5f);

            ImGui::SliderFloat("Scale##", &sim.getSolidObject().scale, 0.01f, 10.0f, "%.2f");
            ImGui::SliderAngle("Angle", &sim.getSolidObject().angle);

        }

        ImGui::Spacing();
    }

    if (ImGui::CollapsingHeader("Environment"))
    {
        ImGui::SliderFloat("Gravity", &prop.gravity, 0.0f, 0.5f);

        ImGui::Checkbox("Edges", &prop.solidEdges);
        ImGui::Spacing();
    }

    ImGui::End();

    return pixels;
}

void PropertiesPanel::draw()
{
    txtr->update((Uint8*) pixels);
    mainWindow.draw(canvas);
}