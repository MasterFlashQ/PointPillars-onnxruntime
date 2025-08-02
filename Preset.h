#pragma once
#ifndef PRESET_H_
#define PRESET_H_

namespace PreSet
{
    int line_set[][2] = {
    0,1,1,2,2,3,3,0,4,5,5,6,6,7,7,4,0,4,1,5,2,6,3,7
    };
    float color_set[][3] = {
        1.0f,0.0f,0.0f,
        0.0f,1.0f,0.0f,
        0.0f,0.0f,1.0f
    };

    float anchor_trans[][3] = {
        -0.5f,-0.5f,0.0f,
        -0.5f,-0.5f,1.0f,
        -0.5f,0.5f,1.0f,
        -0.5f,0.5f,0.0f,
        0.5f,-0.5f,0.0f,
        0.5f,-0.5f,1.0f,
        0.5f,0.5f,1.0f,
        0.5f,0.5f,0.0f
    };
}

#endif // !VISUALIZATION_H_
