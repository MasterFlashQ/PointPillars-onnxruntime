#pragma once
#ifndef POINTPILLARS_H_
#define POINTPILLARS_H_

#include <onnxruntime_cxx_api.h>
#include <cuda_runtime.h>

#include "common_trt.h"
#include "io_trt.hpp"
#include "voxelization_trt.h"
#include "utils.h"
#include "nms_trt.h"

class PointPillars
{
public:
	PointPillars(std::string model_path);
    ~PointPillars()
    {
        session_options.release();
        session_.release();
    }
	std::vector<Box3dfull> inference(std::vector<Point> &points);

private:
    //[in] points
    //[out] Null
    void PreProcess(std::vector<Point>& points_in);
    void PostProcess(std::vector<float>* output, std::vector<Box3dfull>& bboxes);

    std::vector<float> voxel_size = { 0.16, 0.16, 6 };
    //std::vector<float> coors_range = { 0, -39.68, -3, 69.12, 39.68, 1 };
    std::vector<float> coors_range = { -69.12, -39.68, -3, 69.12, 39.68, 3 };
    int max_points = 32;
    int max_voxels = 40000;
    int NDim = 3;

    int* d_num_points_per_voxel = nullptr;
    float* d_voxels = nullptr;
    int voxel_num;
    int* d_coors_padded = nullptr;

    int num_class = 3, num_box = 100;
    Ort::SessionOptions session_options;
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "PointPillars");
    Ort::Session session_{ nullptr };
    std::array<int64_t, 3> input_pillars_shapeinfo;
    std::array<int64_t, 2> input_coors_shapeinfo;
    std::array<int64_t, 1> input_npoints_shapeinfo;
    Ort::MemoryInfo allocator_info{nullptr};
    std::vector<int64_t> input_pillars_dims;
    std::vector<int64_t> input_coors_dims;
    Ort::AllocatorWithDefaultOptions allocator;

    float nms_thr = 0.01, score_thr = 0.1;
    int max_num = 50;
};

#endif // !POINTPILLARS_H_
