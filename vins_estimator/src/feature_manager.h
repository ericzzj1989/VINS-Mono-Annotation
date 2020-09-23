/**
* feature_manager.h主要三个类：
* FeatureManager管理所有特征点，通过list容器存储特征点属性
* FeaturePerId指的是某feature_id下的所有FeaturePerFrame。常用feature_id和观测第一帧start_frame、最后一帧endFrame()
* FeaturePerFrame指的是每帧基本的数据：特征点[x,y,z,u,v,vx,vy]和td IMU与cam同步时间差
* 
* 三者串联最好的例子是：从f_manager到it_per_id再到底层的it_per_frame，就可以得到基本数据point了
* for (auto &it_per_id : f_manager.feature)
* {
*   ......
*   for (auto &it_per_frame : it_per_id.feature_per_frame)
*   {
*     Vector3d pts_j = it_per_frame.point;// 归一化相机坐标系下3D路标点坐标
*   }
* }
*/


#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
using namespace std;

#include <eigen3/Eigen/Dense>
using namespace Eigen;

#include <ros/console.h>
#include <ros/assert.h>

#include "parameters.h"

/**
 * 类FeaturePerFrame表示每帧图像的基本数据：特征点[x, y, z, u, v, vx, vy]和IMU与cam同步时间差td
 * 意义：每个路标点在一张图像中的信息
*/
class FeaturePerFrame
{
  public:
    // 类FeaturePerFrame的构造函数，_point为每帧的特征点[x,y,z,u,v,vx,vy]，td为IMU和cam同步时间差
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td)
    {
        point.x() = _point(0);
        point.y() = _point(1);
        point.z() = _point(2);
        uv.x() = _point(3);
        uv.y() = _point(4);
        velocity.x() = _point(5); 
        velocity.y() = _point(6); 
        cur_td = td;
    }
    double cur_td;
    Vector3d point; // 特征点在归一化相机坐标系的3D坐标
    Vector2d uv;
    Vector2d velocity;
    double z; // 特征点的深度
    bool is_used; // 是否被用了
    double parallax; // 视差
    MatrixXd A; // 变换矩阵
    VectorXd b;
    double dep_gradient; // 没有被用到，可以删除
};

/**
 * 类FeaturePerId表示某个特征点feature_id下的所有FeaturePerFrame。常用feature_id和观测第一帧start_frame、最后一帧endFrame()
 * 意义：每个路标点由多个连续的图像观测到
*/
class FeaturePerId
{
  public:
    const int feature_id; // 特征点ID索引
    int start_frame; // 首次被观测到时，该帧的索引
    vector<FeaturePerFrame> feature_per_frame; // 能够观测到某个特征点的所有帧，该特征点能够被哪些帧共视，size()表示共视帧的数量

    int used_num; // 该特征出现的次数，等价于vector容器feature_per_frame的个数size()
    bool is_outlier; // 是否为外点
    bool is_margin; // 是否边缘化
    double estimated_depth; // 估计的逆深度
    int solve_flag; // 0 haven't solve yet; 1 solve succ; 2 solve fail; 求解器

    Vector3d gt_p; // 没有被用到，可以删除

    // 类FeaturePerId的构造函数，_feature_id为该特征点ID索引，_start_frame为首次观测到该特征点的帧索引
    FeaturePerId(int _feature_id, int _start_frame)
        : feature_id(_feature_id), start_frame(_start_frame),
          used_num(0), estimated_depth(-1.0), solve_flag(0) 
    {
    }

    int endFrame(); // 返回最后一次观测到这个特征点的帧索引ID
};

/**
 * 类FeatureManager管理所有的特征点，通过list容器feature存储特征点的属性
 * 意义：滑窗内所有的路标点
*/
class FeatureManager
{
  public:
    FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td);
    // void debugShow(); 调试代码，没有用到
    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    //void updateDepth(const VectorXd &x);
    void setDepth(const VectorXd &x);
    void removeFailures();
    void clearDepth(const VectorXd &x);
    VectorXd getDepthVector();
    void triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]);
    void removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P);
    void removeBack();
    void removeFront(int frame_count);
    // void removeOutlier(); 没有用到
    list<FeaturePerId> feature; // 特征管理器类主要指的是这个list容器，非常重要，通过FeatureManager中的这个list容器可以得到滑动窗口内所有的角点信息
    int last_track_num;

  private:
    double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif