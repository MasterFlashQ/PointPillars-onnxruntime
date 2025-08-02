#ifndef IO_TRT_HPP
#define IO_TRT_HPP
#include <vector>
#include <fstream>
#include <iostream>
#include <cstring>
#include <iomanip>

#include "common_trt.h"

inline void pointCloudFiler(std::vector<Point>& points, std::vector<Point>& new_points) {
    const float data_range[] = { -69.12, -39.68, -3, 69.12, 39.68, 3 };
    for (const auto& point : points) {
        if (point.x > data_range[0] && point.x < data_range[3]
            && point.y > data_range[1] && point.y < data_range[4]
            && point.z > data_range[2] && point.z < data_range[5]) {
            new_points.emplace_back(point);
        }
    }

    // // 输出数据（测试用）
    // for (auto point : new_points) {
    //     std::cout << point.x << ", " << point.y << ", " << point.z << " " << point.feature << std::endl;
    // }
    // std::cout << std::endl;
    // std::cout << "points size: " << new_points.size() << std::endl;
}

inline bool readPoints(std::string file_path, std::vector<Point>& points) {
    // 打开文件
    std::ifstream file(file_path, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件" << std::endl;
        return 0;
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    // 分配内存
    std::vector<char> buffer(size);

    // 读取数据
    file.read(buffer.data(), size);

    // 转换数据类型（如果需要）
    std::vector<float> data(size / sizeof(float));
    std::memcpy(data.data(), buffer.data(), size);

    // 关闭文件
    file.close();

    points.reserve(data.size() / 4); // 每个Point由4个float组成

    for (size_t i = 0; i < data.size(); i += 4) {
        Point p = { data[i], data[i + 1], data[i + 2], data[i + 3] };
        points.push_back(p);
    }

    return 1;
}

inline void writeFile(std::vector<Box3dfull>& bboxes, const std::string file_path) {
    std::ofstream outfile(file_path);

    // 检查文件是否成功打开
    if (!outfile.is_open()) {
        std::cerr << "Failed to open file " << file_path << " for writing." << std::endl;
        return;
    }

    outfile << std::fixed << std::setprecision(4);
    // 遍历 bboxes 中的每个元素，并写入文件
    for (const auto& bbox_3d : bboxes) {
        outfile << bbox_3d.x << " " << bbox_3d.y << " " << bbox_3d.z << " "
            << bbox_3d.w << " " << bbox_3d.l << " " << bbox_3d.h << " "
            << bbox_3d.theta << " " << bbox_3d.score << " " << bbox_3d.label << std::endl;
    }

    // 关闭文件
    outfile.close();
}
#endif // IO_TRT_HPP