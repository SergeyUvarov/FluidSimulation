#include "simulationCore.h"

CudaFluidSim::CudaFluidSim() {
    restart();
    solidObj.clear();

    flowSources.push_back(FlowSource());
    //solidObj.load("test.png");
}

CudaFluidSim::~CudaFluidSim() {
    field1.release();
    field2.release();
    solidObj.clear();
    cudaFree(pixelsField);
    cudaFree(solidGrid);
}

void CudaFluidSim::setRunStatus(bool status) {
    pause = !status;
}

bool CudaFluidSim::isRun() const {
    return !pause;
}

void CudaFluidSim::restart() {
    field1.transform(simProp);
    field2.transform(simProp);
    cudaFree(pixelsField);
    cudaMalloc(&pixelsField, simProp.width * simProp.height * sizeof(Uint32));
    cudaMalloc(&solidGrid, simProp.width * simProp.height * sizeof(bool));

    float2* vel = new float2[simProp.width * simProp.height];
    for (int i = 0; i < simProp.width * simProp.height; i++) {
        vel[i].x = (static_cast <float>(rand()) / static_cast <float>(RAND_MAX) - 0.5f) * 5.0f;
        vel[i].y = (static_cast <float>(rand()) / static_cast <float>(RAND_MAX) - 0.5f) * 5.0f;
    }
    field1.setVelocity(vel);
    delete[] vel;

    cudaMemset(field1.color, 0, simProp.width * simProp.height * sizeof(Color3f));
    cudaMemset(field2.color, 0, simProp.width * simProp.height * sizeof(Color3f));
    cudaMemset(field1.pressure, 0, simProp.width * simProp.height * sizeof(float));
    cudaMemset(field2.pressure, 0, simProp.width * simProp.height * sizeof(float));
    cudaMemset(solidGrid, 0, simProp.width * simProp.height * sizeof(bool));


    elapsedTime = 0.0f;
}

void CudaFluidSim::setProperties(const Properties& newProp) {
    simProp = newProp;
}

void CudaFluidSim::loadSolidObject(const std::string& filePath)
{
    solidObj.load(filePath);
}

bool CudaFluidSim::isSolidObject() const
{
    return (solidObj.width > 0) && (solidObj.height > 0);
}

SolidObject& CudaFluidSim::getSolidObject()
{
    return solidObj;
}

std::vector<FlowSource>& CudaFluidSim::getFlowSource()
{
    return flowSources;
}

float CudaFluidSim::getElapsedTime() const
{
    return elapsedTime;
}

Properties& CudaFluidSim::getProperties()
{
    return simProp;
}

float CudaFluidSim::step(Uint32* pixels, float dt) {

    dim3 threadsPerBlock(simProp.threadsBlockSizeX, simProp.threadsBlockSizeY);
    dim3 numBlocks((simProp.width + simProp.threadsBlockSizeX - 1) / threadsPerBlock.x, (simProp.height + simProp.threadsBlockSizeY - 1) / threadsPerBlock.y);

    field1.grid = solidGrid;
    field2.grid = solidGrid;

    if (!pause) {
        if (simProp.dt > 0.0f) dt = simProp.dt;

        clearField << <numBlocks, threadsPerBlock >> > (field1, simProp); // field1 <- field1

        advect << <numBlocks, threadsPerBlock >> > (field2, field1, simProp, dt); // field2 <- field1

        computeCurl << <numBlocks, threadsPerBlock >> > (field2, simProp); // field2 <- field2

        vorticity << <numBlocks, threadsPerBlock >> > (field1, field2, simProp, dt); // field1 <- field2

        FlowSource* cudaFlowSources;
        cudaMalloc(&cudaFlowSources, flowSources.size() * sizeof(FlowSource));
        cudaMemcpy(cudaFlowSources, &(*flowSources.begin()), flowSources.size() * sizeof(FlowSource), cudaMemcpyHostToDevice);
        applyInteractive << <numBlocks, threadsPerBlock >> > (pixelsField, field1, cudaFlowSources, flowSources.size(), simProp, dt, elapsedTime); // field1 <- field1
        cudaFree(cudaFlowSources);

        float alpha_v = simProp.velocityDiffuseCoef / dt;
        float beta_v = 1.0f / (4.0f + alpha_v);
        float alpha_c = simProp.colorDiffuseCoef / dt;
        float beta_c = 1.0f / (4.0f + alpha_c);

        for (int itr = 0; itr < simProp.jacobiIterationCount; itr++) {
            if (!simProp.fastJacobi) std::swap(field1, field2);
            jacobi << <numBlocks, threadsPerBlock >> > (field1, field2, simProp, dt,
                alpha_v, beta_v, alpha_c, beta_c, itr); // field1 <- field1

            if (simProp.fastJacobi) break;
        }

        computePressure << <numBlocks, threadsPerBlock >> > (field2, field1, simProp); // field2 <- field1
    }
    else std::swap(field1, field2); // field2 <- field1

    draw << <numBlocks, threadsPerBlock >> > (pixelsField, field2, simProp); // field2 <- field2

    processingSolidObjects << <numBlocks, threadsPerBlock >> > (pixelsField, field2, simProp, solidObj, dt, solidGrid); // field2 <- field2

    std::swap(field1, field2); // field1 <- field2

    cudaMemcpy(pixels, pixelsField, simProp.width * simProp.height * sizeof(Uint32), cudaMemcpyDeviceToHost);

    if (!pause) elapsedTime += dt;
    return elapsedTime;
}