#include "PointPillars.h"

PointPillars::PointPillars(std::string model_path)
{
    std::wstring modelPath = std::wstring(model_path.begin(), model_path.end());
    
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
    session_= Ort::Session(env, modelPath.c_str(), session_options);
    allocator_info= Ort::MemoryInfo("Cuda", OrtAllocatorType::OrtArenaAllocator, 0, OrtMemTypeDefault);

    Ort::TypeInfo input_pillars_typeinfo = session_.GetInputTypeInfo(0);
    auto input_pillars_tensorinfo = input_pillars_typeinfo.GetTensorTypeAndShapeInfo();
    input_pillars_dims = input_pillars_tensorinfo.GetShape();
    Ort::TypeInfo input_coors_typeinfo = session_.GetInputTypeInfo(1);
    auto input_coors_tensorinfo = input_coors_typeinfo.GetTensorTypeAndShapeInfo();
    input_coors_dims = input_coors_tensorinfo.GetShape();
}

void PointPillars::PreProcess(std::vector<Point>& points)
{
    cudaMalloc((void**)&d_num_points_per_voxel, max_voxels * sizeof(int));
    cudaMemset(d_num_points_per_voxel, 0, max_voxels * sizeof(int));
    cudaMalloc((void**)&d_voxels, max_voxels * max_points * sizeof(Point));
    cudaMemset(d_voxels, 0.f, max_voxels * max_points * sizeof(Point));
    int* d_coors = nullptr;
    cudaMalloc((void**)&d_coors, max_voxels * NDim * sizeof(int));
    cudaMemset(d_coors, 0, max_voxels * NDim * sizeof(int));

    voxel_num = voxelizeGpu(points, voxel_size, coors_range, max_points, max_voxels, d_voxels, d_coors, d_num_points_per_voxel, NDim);
    cudaMalloc((void**)&d_coors_padded, voxel_num * (NDim + 1) * sizeof(int));
    cudaMemset(d_coors_padded, 0, voxel_num * (NDim + 1) * sizeof(int));
    padCoorsGPU(d_coors, d_coors_padded, voxel_num);
    cudaFree(d_coors);
}

std::vector<Box3dfull> PointPillars::inference(std::vector<Point>& points_in)
{
    std::vector<Point> points;
    pointCloudFiler(points_in, points);
    PreProcess(points);

    std::vector<float> *output=new std::vector<float>(num_box * (7 + num_class + 1));
    
    input_pillars_shapeinfo = { voxel_num,input_pillars_dims[1],input_pillars_dims[2] };
    input_coors_shapeinfo = { voxel_num,input_coors_dims[1] };
    input_npoints_shapeinfo = { voxel_num };

    Ort::Value input_tensors[] = { Ort::Value::CreateTensor<float>(allocator_info, d_voxels, voxel_num * 32 * 4, input_pillars_shapeinfo.data(), input_pillars_shapeinfo.size()),
                                   Ort::Value::CreateTensor<int>(allocator_info, d_coors_padded, voxel_num * 4, input_coors_shapeinfo.data(), input_coors_shapeinfo.size()),
                                   Ort::Value::CreateTensor<int>(allocator_info, d_num_points_per_voxel, voxel_num, input_npoints_shapeinfo.data(), input_npoints_shapeinfo.size()) };
    std::vector<Ort::Value> ort_outputs;

    auto input_node_name0 = session_.GetInputName(0, allocator);
    auto input_node_name1 = session_.GetInputName(1, allocator);
    auto input_node_name2 = session_.GetInputName(2, allocator);
    auto output_node_name = session_.GetOutputName(0, allocator);
    const std::array<const char*, 3> inputNames = { input_node_name0,input_node_name1,input_node_name2 };
    const std::array<const char*, 1> outNames = { output_node_name };

    ort_outputs = session_.Run(Ort::RunOptions{ nullptr }, inputNames.data(), input_tensors, 3, outNames.data(), outNames.size());

    cudaFree(d_num_points_per_voxel);
    cudaFree(d_voxels);
    cudaFree(d_coors_padded);

    const float* pdata = ort_outputs[0].GetTensorMutableData<float>();
    memcpy(output->data(), pdata, output->size() * sizeof(float));

    std::vector<Box3dfull> bboxes;
    PostProcess(output, bboxes);

    delete output;

    return bboxes;
}

void PointPillars::PostProcess(std::vector<float>* output, std::vector<Box3dfull>& bboxes)
{
    std::vector<Box2d> bboxes_2d;
    std::vector<Box3d> bboxes_3d;
    std::vector<std::vector<float>> scores_list;
    std::vector<float> direction_list;
    decodeDetResults(*output, num_class, bboxes_2d, bboxes_3d, scores_list, direction_list);

    std::vector<Box3dfull> bboxes_3d_nms;
    for (int i = 0; i < num_class; i++) {
        std::vector<int> score_filter_inds;
        std::vector<float> scores;
        filterByScores(i, scores_list, score_thr, score_filter_inds, scores);
        std::vector<Box2d> bboxes_2d_filtered;
        std::vector<Box3d> bboxes_3d_filtered;
        std::vector<float> direction_filtered;
        obtainBoxByInds(score_filter_inds, bboxes_2d, bboxes_2d_filtered, bboxes_3d, bboxes_3d_filtered,
            direction_list, direction_filtered);

        std::vector<int> nms_filter_inds;
        nms(bboxes_2d_filtered, scores, nms_thr, nms_filter_inds);

        for (const auto ind : nms_filter_inds) {
            Box3dfull box3d_full;
            box3d_full.x = bboxes_3d_filtered[ind].x;
            box3d_full.y = bboxes_3d_filtered[ind].y;
            box3d_full.z = bboxes_3d_filtered[ind].z;
            box3d_full.w = bboxes_3d_filtered[ind].w;
            box3d_full.l = bboxes_3d_filtered[ind].l;
            box3d_full.h = bboxes_3d_filtered[ind].h;
            float limited_theta = limitPeriod(bboxes_3d_filtered[ind].theta);
            box3d_full.theta = (1.f - direction_filtered[ind]) * M_PI + limited_theta;
            box3d_full.score = scores[ind];
            box3d_full.label = i;
            bboxes_3d_nms.push_back(box3d_full);
        }
    }
    getTopkBoxes(bboxes_3d_nms, max_num, bboxes);
}