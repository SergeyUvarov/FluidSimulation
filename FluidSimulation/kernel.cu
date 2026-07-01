#include "kernel.h"


CudaFluidSim::CudaFluidSim()
{
    field1.init();
    field2.init();

    cudaMalloc(&pixelField, WIDTH * HEIGHT * sizeof(Uint32));
    cudaMalloc(&curlField, WIDTH * HEIGHT * sizeof(float));

    float2* initField = new float2[WIDTH * HEIGHT];

    for (int x = 0; x < WIDTH; x++)
        for (int y = 0; y < HEIGHT; y++)
    {
            GET(initField, x, y).x = (rand() % 20001 - 10000) / 10000.0f;
            GET(initField, x, y).y = (rand() % 20001 - 10000) / 10000.0f;
    }

    field1.initVelocity(initField);

    delete[] initField;
}


CudaFluidSim::~CudaFluidSim()
{
    cudaFree(pixelField);
    cudaFree(curlField);

    field1.release();
    field2.release();
}


__global__ void jacobi(Field field, Field oldField, float dt,
    float alpha_v, float beta_v, float alpha_c, float beta_c)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * WIDTH + x;

    float2 v, v_pX, v_nX, v_pY, v_nY;
    Color3f c, c_pX, c_nX, c_pY, c_nY;
    float p_pX, p_nX, p_pY, p_nY;

    if (x == 0 || y == 0 || x >= WIDTH - 1 || y >= HEIGHT - 1)
        return;

    for (int itr = 0; itr < JACOBI_ITERATION_COUNT; itr++) {

        Color3f* tmpcolor = oldField.color;
        float2* tmpvelocity = oldField.velocity;
        float* tmppressure = oldField.pressure;

        oldField.color = field.color;
        oldField.velocity = field.velocity;
        oldField.pressure = field.pressure;

        field.color = tmpcolor;
        field.velocity = tmpvelocity;
        field.pressure = tmppressure;


        v = oldField.velocity[i];
        v_pX = GET(oldField.velocity, x + 1, y);
        v_nX = GET(oldField.velocity, x - 1, y);
        v_pY = GET(oldField.velocity, x, y + 1);
        v_nY = GET(oldField.velocity, x, y - 1);

        c = oldField.color[i];
        c_pX = GET(oldField.color, x + 1, y);
        c_nX = GET(oldField.color, x - 1, y);
        c_pY = GET(oldField.color, x, y + 1);
        c_nY = GET(oldField.color, x, y - 1);

        p_pX = GET(oldField.pressure, x + 1, y);
        p_nX = GET(oldField.pressure, x - 1, y);
        p_pY = GET(oldField.pressure, x, y + 1);
        p_nY = GET(oldField.pressure, x, y - 1);

        field.velocity[i] = make_float2(
            (v_pX.x + v_nX.x + v_pY.x + v_nY.x + alpha_v * v.x) * beta_v,
            (v_pX.y + v_nX.y + v_pY.y + v_nY.y + alpha_v * v.y) * beta_v
        );

        field.color[i] = (c_pX + c_nX + c_pY + c_nY + c * alpha_c) * beta_c;

        float B = (v_pX.x - v_nX.x + v_pY.y - v_nY.y);

        field.pressure[i] = (p_pX + p_nX + p_pY + p_nY - 0.5f * B) * 0.25f;

        //__syncthreads();
    }

}


__global__ void advect(Field field, Field oldField, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * WIDTH + x;

    if (x == 0 || y == 0 || x >= WIDTH - 1 || y >= HEIGHT - 1)
        return;

    float2 v = oldField.velocity[i];
    float2 v_pX = GET(oldField.velocity, x + 1, y);
    float2 v_nX = GET(oldField.velocity, x - 1, y);
    float2 v_pY = GET(oldField.velocity, x, y + 1);
    float2 v_nY = GET(oldField.velocity, x, y - 1);

    float gradX_x = v_pX.x - v_nX.x;
    float gradX_y = v_pY.x - v_nY.x;
    float gradY_x = v_pX.y - v_nX.y;
    float gradY_y = v_pY.y - v_nY.y;

    field.velocity[i] = make_float2(
        v.x - dt * (v.x * gradX_x + v.y * gradX_y) * 0.5f,
        v.y - dt * (v.x * gradY_x + v.y * gradY_y) * 0.5f
    );

    Color3f c = oldField.color[i];
    Color3f c_pX = GET(oldField.color, x + 1, y);
    Color3f c_nX = GET(oldField.color, x - 1, y);
    Color3f c_pY = GET(oldField.color, x, y + 1);
    Color3f c_nY = GET(oldField.color, x, y - 1);


    float2 gradR = make_float2(
        c_pX.R - c_nX.R,
        c_pY.R - c_nY.R
    );
    float2 gradG = make_float2(
        c_pX.G - c_nX.G,
        c_pY.G - c_nY.G
    );
    float2 gradB = make_float2(
        c_pX.B - c_nX.B,
        c_pY.B - c_nY.B
    );

    field.color[i].R = c.R - dt * (v.x * gradR.x + v.y * gradR.y) * 0.5f;
    field.color[i].G = c.G - dt * (v.x * gradG.x + v.y * gradG.y) * 0.5f;
    field.color[i].B = c.B - dt * (v.x * gradB.x + v.y * gradB.y) * 0.5f;

    field.pressure[i] = oldField.pressure[i];
}


__global__ void computeCurl(float* curlField, Field oldField)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * WIDTH + x;

    if (x == 0 || y == 0 || x >= WIDTH - 1 || y >= HEIGHT - 1)
        return;

    float v_pX = GET(oldField.velocity, x + 1, y).y;
    float v_nX = GET(oldField.velocity, x - 1, y).y;
    float v_pY = GET(oldField.velocity, x, y + 1).x;
    float v_nY = GET(oldField.velocity, x, y - 1).x;

    curlField[i] = (v_pX - v_nX - v_pY + v_nY) * 0.5f;
}


__global__ void vorticity(Field field, Field oldField, float* curlField, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * WIDTH + x;

    if (x == 0 || y == 0 || x >= WIDTH - 1 || y >= HEIGHT - 1)
        return;

    float4 crl = make_float4(
        GET(curlField, x + 1, y),
        GET(curlField, x - 1, y),
        GET(curlField, x, y + 1),
        GET(curlField, x, y - 1)
    );

    float2 n_vec = make_float2(
        abs(crl.x) - abs(crl.y),
        abs(crl.z) - abs(crl.w)
    );

    float nLen = sqrtf(n_vec.x * n_vec.x + n_vec.y * n_vec.y) + 1e-5f;

    n_vec = make_float2(n_vec.x / nLen * curlField[i], n_vec.y / nLen * curlField[i]);

    field.velocity[i] = make_float2(
        oldField.velocity[i].x + dt * V_VORTICITY * n_vec.y,
        oldField.velocity[i].y + dt * V_VORTICITY * (-n_vec.x)
    );

    field.color[i] = oldField.color[i];
    field.pressure[i] = oldField.pressure[i];
}


__global__ void computePressure(Field field, Field oldField)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * WIDTH + x;

    if (x == 0 || y == 0 || x >= WIDTH - 1 || y >= HEIGHT - 1)
        return;

    float4 prss = make_float4(
        GET(oldField.pressure, x + 1, y),
        GET(oldField.pressure, x - 1, y),
        GET(oldField.pressure, x, y + 1),
        GET(oldField.pressure, x, y - 1)
    );

    field.velocity[i] = make_float2(
        oldField.velocity[i].x - (prss.x - prss.y) * (0.5f / V_DENSITY),
        oldField.velocity[i].y - (prss.z - prss.w) * (0.5f / V_DENSITY)
    );

    field.color[i] = oldField.color[i];
    field.pressure[i] = oldField.pressure[i];
}


__global__ void clearField(Field field, float* curlField)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * WIDTH + x;


    if (x >= WIDTH || y >= HEIGHT)
        return;

    field.pressure[i] = 0.0f;
    if (x == 0 || y == 0 || x == WIDTH - 1 || y == HEIGHT - 1) {
        //field.color[i] = { 0, 0, 0 };
        field.velocity[i] = make_float2(0.0f, 0.0f);
        curlField[i] = 0;
    }
}


__global__ void applyInteractive(Uint32* pixels, Field field, float dt, float t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * WIDTH + x;

    if (x == 0 || y == 0 || x >= WIDTH - 1 || y >= HEIGHT - 1)
        return;

    if (y >= HEIGHT/2 - 30 && y <= HEIGHT / 2 + 30 && (x == WIDTH/2 - 400 || x == WIDTH / 2 + 400)) {
        //field.velocity[i].y += -dt * 150.0f;

        if (x > WIDTH / 2) {
            field.color[i].R = 100.0f * abs(sin(0.1 * t)) * dt;
            field.color[i].G = 100.0f * abs(cos(0.1 * t)) * dt;

            field.velocity[i].x += -dt * 300.0f;
        }
        else {
            field.color[i].B = 100.0f * abs(sin(0.1 * t)) * dt;
            field.color[i].R = 100.0f * abs(cos(0.1 * t)) * dt;

            field.velocity[i].x += dt * 300.0f;
        }

    }


    float dx = x - WIDTH / 2 - 50 * sin(0.25*t);
    float dy = y - HEIGHT / 2 - 100 * cos(0.05*t);
    if (dx*dx + dy*dy < 500) {
        float len = sqrt(dx * dx + dy * dy) + 1e-5f;
        float vec = sqrt(field.velocity[i].x * field.velocity[i].x + field.velocity[i].y * field.velocity[i].y);
        field.velocity[i].x = vec * dx / len;
        field.velocity[i].y = vec * dy / len;
        GET(field.color, x + int(2*dx / len), y + int(2 * dy / len)) = GET(field.color, x + int(2 * dx / len), y + int(2 * dy / len))*0.96f + field.color[i];
        field.color[i] = { 0.0f, 0.0f, 0.0f };
        pixels[i] = sRGB(100, 100, 50);
    }

    dx = x - WIDTH / 2 + 50 * sin(0.25 * t);
    dy = y - HEIGHT / 2 + 100 * cos(0.05 * t);
    if (dx * dx + dy * dy < 500) {
        float len = sqrt(dx * dx + dy * dy) + 1e-5f;
        float vec = sqrt(field.velocity[i].x * field.velocity[i].x + field.velocity[i].y * field.velocity[i].y);
        field.velocity[i].x = vec * dx / len;
        field.velocity[i].y = vec * dy / len;
        GET(field.color, x + int(2 * dx / len), y + int(2 * dy / len)) = GET(field.color, x + int(2 * dx / len), y + int(2 * dy / len))*0.96f + field.color[i];
        field.color[i] = { 0.0f, 0.0f, 0.0f };
        pixels[i] = sRGB(100, 100, 50);
    }

    /*if (int(t) % 3 == 0)
    {
        float dx = x - WIDTH / 2;
        float dy = y - HEIGHT / 2;

        if (dx * dx + dy * dy <= 100) {

            float len = sqrt(dx * dx + dy * dy) + 1e-5f;

            field.velocity[i].x += 120.0f * sin(t) * dt;
            field.velocity[i].y += 120.0f * cos(t) * dt;
            field.color[i].R += 10.0f * abs(sin(-0.05 * t + dt)) * dt;
            field.color[i].G += 10.0f * abs(cos(0.2 * t - dt)) * dt;
            field.color[i].B += 10.0f * abs(sin(-0.3 * t + dt*t)) * dt;
        }
    }*/
}


__global__ void draw(Uint32* pixels, Field field1, float dt, float t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * WIDTH + x;

    if (x >= WIDTH || y >= HEIGHT)
        return;

    pixels[i] = field1.color[i].toRGBA();
}


__host__ void CudaFluidSim::step(Uint32* pixels, float dt)
{
    if (dt > 1.0f / 40.0f)
        dt = 1.0f / 40.0f;

    //dt = 1.0f / 60.0f;


    static float t = 0; t += dt;

    dim3 threadsPerBlock(THREAD_BLOCK_SIZE_X, THREAD_BLOCK_SIZE_Y);
    dim3 numBlocks((WIDTH+ THREAD_BLOCK_SIZE_X-1) / threadsPerBlock.x, (HEIGHT+ THREAD_BLOCK_SIZE_Y-1) / threadsPerBlock.y);

    clearField << <numBlocks, threadsPerBlock >> > (field1, curlField);
   
    advect << <numBlocks, threadsPerBlock >> > (field2, field1, dt); // field1 <- field2

    computeCurl << <numBlocks, threadsPerBlock >> > (curlField, field2);

    vorticity << <numBlocks, threadsPerBlock >> > (field1, field2, curlField, dt); // field2 <- field1

    float alpha_v = 1.0f / (V_DIFFUSE * dt);
    float beta_v = 1.0f / (4.0f + alpha_v);
    float alpha_c = 1.0f / (V_DIFFUSE * dt);
    float beta_c = 1.0f / (4.0f + alpha_v);

    jacobi << <numBlocks, threadsPerBlock >> > (field1, field2, dt,
        alpha_v, beta_v, alpha_c, beta_c);

    computePressure << <numBlocks, threadsPerBlock >> > (field2, field1); // field2 <- field1

    draw << <numBlocks, threadsPerBlock >> > (pixelField, field2, dt, t);

    applyInteractive << <numBlocks, threadsPerBlock >> > (pixelField, field2, dt, t);

    std::swap(field1, field2);

    cudaMemcpy(pixels, pixelField, WIDTH * HEIGHT * sizeof(Uint32), cudaMemcpyDeviceToHost);
}