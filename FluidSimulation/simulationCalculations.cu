#include "simulationCore.h"

__global__ void jacobi(FluidField field, FluidField oldField, Properties prop, float dt,
    float alpha_v, float beta_v, float alpha_c, float beta_c, int gItr)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * prop.width + x;

    if (x >= prop.width || y >= prop.height) return;

    float2 v, v_pX, v_nX, v_pY, v_nY;
    Color3f c, c_pX, c_nX, c_pY, c_nY;
    float p_pX, p_nX, p_pY, p_nY, B,
        pAccuracy, vAccuracy, cAccuracy;


    pAccuracy = powf(10.0f, prop.vAccuracy);
    vAccuracy = powf(10.0f, prop.pAccuracy);
    cAccuracy = powf(10.0f, prop.cAccuracy);

    for (int itr = 0; itr < prop.jacobiIterationCount; itr++) {
        if (prop.fastJacobi) {
            FluidField tmpField = field;
            field = oldField;
            oldField = tmpField;
        }


        bool computVelocity = (itr == 0 && prop.fastJacobi) || (gItr == 0 && !prop.fastJacobi) || abs(field.velocity[i].x - oldField.velocity[i].x) > abs(vAccuracy * field.velocity[i].x) ||
            abs(field.velocity[i].y - oldField.velocity[i].y) > abs(vAccuracy * field.velocity[i].y);
        bool computPressure = (itr == 0 && prop.fastJacobi) || (gItr == 0 && !prop.fastJacobi) || abs(field.pressure[i] - oldField.pressure[i]) > abs(pAccuracy * field.pressure[i]);

        if (computPressure || computVelocity) {
            v = oldField.velocity[i];
            v_pX = oldField.getVelocity(x, y, 1, 0);
            v_nX = oldField.getVelocity(x, y, -1, 0);
            v_pY = oldField.getVelocity(x, y, 0, 1);
            v_nY = oldField.getVelocity(x, y, 0, -1);
        }

        if (computVelocity) {
            field.velocity[i] = make_float2(
                (v_pX.x + v_nX.x + v_pY.x + v_nY.x + alpha_v * v.x) * beta_v,
                (v_pX.y + v_nX.y + v_pY.y + v_nY.y + alpha_v * v.y) * beta_v
            );
        }
        else field.velocity[i] = oldField.velocity[i];

        if (computPressure) {
            B = (v_pX.x - v_nX.x + v_pY.y - v_nY.y);

            p_pX = oldField.getPressure(x, y, 1, 0);
            p_nX = oldField.getPressure(x, y, -1, 0);
            p_pY = oldField.getPressure(x, y, 0, 1);
            p_nY = oldField.getPressure(x, y, 0, -1);

            field.pressure[i] = (p_pX + p_nX + p_pY + p_nY - 0.5f * B) * 0.25f;
        }
        else field.pressure[i] = oldField.pressure[i];


        if ((itr == 0 && prop.fastJacobi) || (gItr == 0 && !prop.fastJacobi) || (abs(field.color[i].R - oldField.color[i].R) > abs(cAccuracy * field.color[i].R) ||
            abs(field.color[i].G - oldField.color[i].G) > abs(cAccuracy * field.color[i].G) ||
            abs(field.color[i].B - oldField.color[i].B) > abs(cAccuracy * field.color[i].B))) 
        {
            c = oldField.color[i];
            c_pX = oldField.getColor(x, y, 1, 0);
            c_nX = oldField.getColor(x, y, -1, 0);
            c_pY = oldField.getColor(x, y, 0, 1);
            c_nY = oldField.getColor(x, y, 0, -1);
            field.color[i] = (c_pX + c_nX + c_pY + c_nY + c * alpha_c) * beta_c;
        }
        else field.color[i] = oldField.color[i];

        if (!prop.fastJacobi) break;
    }
}


__global__ void advect(FluidField field, FluidField oldField, Properties prop, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * prop.width + x;

    if (x >= prop.width || y >= prop.height) return;

    float2 v = oldField.velocity[i];
    float2 v_pX = oldField.getVelocity(x, y, 1, 0);
    float2 v_nX = oldField.getVelocity(x, y, -1, 0);
    float2 v_pY = oldField.getVelocity(x, y, 0, 1);
    float2 v_nY = oldField.getVelocity(x, y, 0, -1);

    float gradX_x = v_pX.x - v_nX.x;
    float gradX_y = v_pY.x - v_nY.x;
    float gradY_x = v_pX.y - v_nX.y;
    float gradY_y = v_pY.y - v_nY.y;

    field.velocity[i] = make_float2(
        v.x - dt * (v.x * gradX_x + v.y * gradX_y) * 0.5f,
        v.y - dt * (v.x * gradY_x + v.y * gradY_y) * 0.5f
    );

    Color3f c = oldField.color[i];
    Color3f c_pX = oldField.getColor(x, y, 1, 0);
    Color3f c_nX = oldField.getColor(x, y, -1, 0);
    Color3f c_pY = oldField.getColor(x, y, 0, 1);
    Color3f c_nY = oldField.getColor(x, y, 0, -1);


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


__global__ void computeCurl(FluidField field, Properties prop)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * prop.width + x;

    if (x >= prop.width || y >= prop.height) return;

    float v_pX = field.getVelocity(x, y, 1, 0).y;
    float v_nX = field.getVelocity(x, y, -1, 0).y;
    float v_pY = field.getVelocity(x, y, 0, 1).x;
    float v_nY = field.getVelocity(x, y, 0, -1).x;

    field.curl[i] = (v_pX - v_nX - v_pY + v_nY) * 0.5f;
}


__global__ void vorticity(FluidField field, FluidField oldField, Properties prop, float dt)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * prop.width + x;

    if (x >= prop.width || y >= prop.height) return;

    float4 crl = make_float4(
        oldField.getCurl(x + 1, y),
        oldField.getCurl(x - 1, y),
        oldField.getCurl(x, y + 1),
        oldField.getCurl(x, y - 1)
    );

    float2 n_vec = make_float2(
        abs(crl.x) - abs(crl.y),
        abs(crl.z) - abs(crl.w)
    );

    float nLen = sqrtf(n_vec.x * n_vec.x + n_vec.y * n_vec.y) + 1e-5f;

    n_vec = make_float2(n_vec.x / nLen * oldField.curl[i], n_vec.y / nLen * oldField.curl[i]);

    field.velocity[i] = make_float2(
        oldField.velocity[i].x + dt * prop.vorticityForceCoef * n_vec.y,
        oldField.velocity[i].y + dt * prop.vorticityForceCoef * (-n_vec.x)
    );

    field.color[i] = oldField.color[i];
    field.pressure[i] = oldField.pressure[i];
}


__global__ void computePressure(FluidField field, FluidField oldField, Properties prop)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * prop.width + x;

    if (x >= prop.width || y >= prop.height) return;

    float4 prss = make_float4(
        oldField.getPressure(x, y, 1, 0),
        oldField.getPressure(x, y, -1, 0),
        oldField.getPressure(x, y, 0, 1),
        oldField.getPressure(x, y, 0, -1)
    );

    field.velocity[i] = make_float2(
        oldField.velocity[i].x - (prss.x - prss.y) * prop.densityCoef,
        oldField.velocity[i].y - (prss.z - prss.w) * prop.densityCoef
    );

    field.color[i] = oldField.color[i];
    field.pressure[i] = oldField.pressure[i];
}


__global__ void clearField(FluidField field, Properties prop)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * prop.width + x;

    if (x >= prop.width || y >= prop.height) return;

    field.pressure[i] = 0.0f;
    if (field.grid[i]) {
        field.velocity[i] = make_float2(0.0f, 0.0f);
        field.color[i] = { 0.0f, 0.0f, 0.0f };
    }
}


__global__ void applyInteractive(Uint32* pixels, FluidField field, FlowSource* flowSources, int sourcesCount, Properties prop, float dt, float t)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * prop.width + x;

    if (x >= prop.width || y >= prop.height) return;

    field.velocity[i].y += prop.gravity * dt;

    for (int idx = 0; idx < sourcesCount; ++idx)
    {
        float dx1 = (x - flowSources[idx].x * prop.width) / prop.scale;
        float dy1 = -(y - flowSources[idx].y * prop.height) / prop.scale;

        int dx = dx1 * cos(flowSources[idx].angle) + dy1 * sin(flowSources[idx].angle);
        int dy = -dx1 * sin(flowSources[idx].angle) + dy1 * cos(flowSources[idx].angle);

        if (abs(dx) < 2 && abs(dy) < flowSources[idx].size) {

            if (flowSources[idx].strength == 0.0f) break;

            if (flowSources[idx].isRainbow) {
                field.color[i].R = 1.0f * abs(sin(0.01f * t * flowSources[idx].rainbowSpeed + flowSources[idx].x));
                field.color[i].G = 1.0f * abs(cos(0.02f * t * flowSources[idx].rainbowSpeed + flowSources[idx].y));
                field.color[i].B = 1.0f * abs(sin(0.012f * t * flowSources[idx].rainbowSpeed - flowSources[idx].x * flowSources[idx].y));
            }
            else {
                field.color[i] = flowSources[idx].color;
            }

            field.velocity[i].x = flowSources[idx].strength * cos(flowSources[idx].angle);
            field.velocity[i].y = -flowSources[idx].strength * sin(flowSources[idx].angle);
        }
    }
}


__global__ void processingSolidObjects(Uint32* pixels, FluidField field, Properties prop, SolidObject obj, float dt, bool* solidGrid)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * prop.width + x;

    if (x >= prop.width || y >= prop.height) return;
    solidGrid[i] = false;

    if(prop.solidEdges && (x == 0 || y == 0 || x == prop.width-1 || y == prop.height-1) ) solidGrid[i] = true;


    float dx1 = (x - obj.x * prop.width) / obj.scale / prop.scale;
    float dy1 = (y + obj.y * prop.height) / obj.scale / prop.scale;

    int dx = dx1 * cos(obj.angle) - dy1 * sin(obj.angle);
    int dy = dx1 * sin(obj.angle) + dy1 * cos(obj.angle);

    if (dx < 0 || dy < 0 || dx >= obj.width || dy >= obj.height) {
        return;
    }

    int idx = dy * obj.width + dx;
    if (obj.grid[idx]) {
        solidGrid[i] = true;
        pixels[i] = obj.texture[idx];
    }
}


__global__ void draw(Uint32* pixels, FluidField field, Properties prop)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int i = y * prop.width + x;

    if (x >= prop.width || y >= prop.height) return;

    if (prop.displayMode == 0) {
        pixels[i] = (field.color[i] * prop.colorAmplify).toRGBA();
    } else if(prop.displayMode == 1) {
        float p = 255 / 2 + 255 / 2 * LIM(field.pressure[i] * prop.colorAmplify);
        pixels[i] = sRGB(p, p, p);
    } else if (prop.displayMode == 2) {
        float p = 255 * LIM(sqrtf(field.velocity[i].x * field.velocity[i].x + field.velocity[i].y * field.velocity[i].y) * prop.colorAmplify);
        pixels[i] = sRGB(p, p, p);
    }
    
    if (prop.displayArrow && x % prop.arrowStepSize == 0 && (y + (x % (prop.arrowStepSize *2) == 0)* prop.arrowStepSize /2 ) % prop.arrowStepSize == 0) {
        int idx = i;

        float sx = x, sy = y;

        float vel = hypotf(field.velocity[idx].x, field.velocity[idx].y);
        float dx = field.velocity[idx].x / vel;
        float dy = field.velocity[idx].y / vel;
        for (int j = 0; j < prop.arrowStepSize / 1.6f * LIM(vel * prop.arrowAmplify) && sx > 0 && sy > 0 && sx < prop.width && sy < prop.height; j++) {
            idx = round(sy) * prop.width + round(sx);
            
            int clr = 200 * LIM(vel * prop.arrowAmplify);
            pixels[idx] = sRGB(clr, clr, clr);

            if(j == 0) pixels[idx] = sRGB(255, 0, 0);
            sx += dx; sy += dy;
        }
    }
}