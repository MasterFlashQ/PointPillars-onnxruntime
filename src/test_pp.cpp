#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <queue>
#include <fstream>
#include <filesystem>

#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/transforms.h>
#include <pcl/common/io.h>

#include "PointPillars.h"
#include "Preset.h"

namespace fs = std::filesystem;

std::queue<std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, std::vector<Box3dfull>*>> queue_points_box;
std::mutex lock;

void computeBoxCorner(Box3dfull& box, pcl::PointCloud<pcl::PointXYZ>::Ptr box_corners)
{
    /*
           ^ z   x            6 ------ 5
           |   /             / |     / |
           |  /             2 -|---- 1 |
    y      | /              |  |     | |
    <------|o               | 7 -----| 4
                            |/   o   |/
                            3 ------ 0
    x: front, y: left, z: top
    */


    for (int i = 0; i < 8; i++)
    {
        pcl::PointXYZ box_corner(PreSet::anchor_trans[i][0] * box.w, PreSet::anchor_trans[i][1] * box.l, PreSet::anchor_trans[i][2] * box.h);
        box_corners->push_back(box_corner);
    }
    float rotate_sin = sin(box.theta);
    float rotate_cos = cos(box.theta);
    Eigen::Matrix4f rotate_matrix;
    rotate_matrix << rotate_cos, rotate_sin, 0.0f, box.x,
        -rotate_sin, rotate_cos, 0.0f, box.y,
        0.0f, 0.0f, 1.0f, box.z,
        0.0f, 0.0f, 0.0f, 1.0f;

    pcl::transformPointCloud(*box_corners, *box_corners, rotate_matrix);
}

void getPCD()
{
    std::string file_dir = "G:/kitti/dataset/sequences/00/velodyne";
    std::vector<std::string> file_list;
    for (const auto& entry : fs::directory_iterator(file_dir))
    {
        if (!entry.is_regular_file())
            continue;

        std::string filename = entry.path().filename().string();
        file_list.push_back(filename);
    }

    std::sort(file_list.begin(), file_list.end());

    PointPillars* pp = new PointPillars("F:/PointPillars-feature-deployment/pretrained/model.onnx");

    for (auto file_name : file_list)
    {
        std::string file_path = file_dir + "/" + file_name;

        std::vector<Point> points_ori;
        bool read_data_ok = readPoints(file_path, points_ori);
        if (!read_data_ok) continue;
        std::vector<Point> points;
        pointCloudFiler(points_ori, points);

        std::vector<Box3dfull> *bboxes=new std::vector<Box3dfull>;

        *bboxes=pp->inference(points);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        for (int i = 0; i < (points).size(); i++)
        {
            pcl::PointXYZI p(points[i].x, points[i].y, points[i].z, points[i].feature);
            cloud->push_back(p);
        }

        std::pair<pcl::PointCloud<pcl::PointXYZI>::Ptr, std::vector<Box3dfull>*> data_result(cloud, bboxes);

        std::lock_guard<std::mutex> thread_lock(lock);
        queue_points_box.push(data_result);

    }
}

void visualizeBox()
{
    int tag = 0;
    boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer->setBackgroundColor(255, 255, 255);
    viewer->setWindowName("Detection result");

    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    cloud->push_back(pcl::PointXYZI(0, 0, 0, 0));
    viewer->addPointCloud<pcl::PointXYZI>(cloud, "cloud");

    std::vector<Box3dfull>* bboxes;

    while (true)
    {
        std::lock_guard<std::mutex> thread_lock(lock);
        if (queue_points_box.empty())
            continue;
        viewer->removeAllShapes();

        cloud = queue_points_box.front().first;
        bboxes = queue_points_box.front().second;
        queue_points_box.pop();

        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> color(cloud, "intensity");

        viewer->updatePointCloud<pcl::PointXYZI>(cloud, color, "cloud");

        for (int i = 0; i < (*bboxes).size(); i++)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr box_corner(new pcl::PointCloud<pcl::PointXYZ>);
            computeBoxCorner((*bboxes)[i], box_corner);

            for (int j = 0; j < 12; j++)
            {
                std::string line_ID = std::to_string(tag) +"-" + std::to_string(i) + "-" + std::to_string(j);
                viewer->addLine(box_corner->points[PreSet::line_set[j][0]], box_corner->points[PreSet::line_set[j][1]], line_ID);
                viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                    PreSet::color_set[(*bboxes)[i].label][0], PreSet::color_set[(*bboxes)[i].label][1], PreSet::color_set[(*bboxes)[i].label][2], line_ID);
            }
        }
        tag++;

        delete bboxes;
        viewer->spinOnce();
    }
}

int main()
{
    std::thread(getPCD).detach();
    std::thread(visualizeBox).detach();
    while (true)
    {

    }
}