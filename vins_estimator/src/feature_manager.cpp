#include "feature_manager.h"

/**
 * @brief           获取最后一次观测到这个特征点的帧索引ID
 * @Description     通过某个特征点首次被观测到的帧索引start_frame和该特征点共视帧的数量feature_per_frame.size()，得到最后一次观测到这个特征点的帧索引ID
 * @param[in]       void
 * @return          int 观测到该特征点的最后一帧ID
*/
int FeaturePerId::endFrame()
{
    return start_frame + feature_per_frame.size() - 1;
}

/**
 * @brief           类FeatureManager的构造函数
 * @Description     通过参数_Rs[]设置IMU到世界坐标系的旋转矩阵Rs，同时将外参旋转矩阵ric[0]初始化为单位矩阵
 * @param[in]       _ric[] 外参旋转矩阵
 * @return          void
*/
FeatureManager::FeatureManager(Matrix3d _Rs[])
    : Rs(_Rs)
{
    for (int i = 0; i < NUM_OF_CAM; i++) // NUM_OF_CAM = 1
        ric[i].setIdentity(); // 通过EIGEN的setIdentity()将外参旋转矩阵ric[0]设置为单位矩阵
}

/**
 * @brief           设置外参的旋转矩阵
 * @Description     通过参数_ric[0]设置外参旋转矩阵ric[0]
 * @param[in]       _ric[] 外参旋转矩阵
 * @return          void
*/
void FeatureManager::setRic(Matrix3d _ric[])
{
    for (int i = 0; i < NUM_OF_CAM; i++) // NUM_OF_CAM = 1
    {
        ric[i] = _ric[i];
    }
}

/**
 * @brief           清除滑动窗口内所有特征点信息
 * @Description     使用clear()清除list容器feature存储的所有特征点信息，使得feature.size() = 0
 * @param[in]       void
 * @return          void
*/
void FeatureManager::clearState()
{
    feature.clear();
}

/**
 * @brief           滑动窗口中被跟踪特征点的总数
 * @Description     该特征点被两帧以上观测到了，且第一次观测到的帧数不是在最后面
 * @param[in]       void
 * @return          int cnt 滑动窗口中被跟踪特征点的总数目
*/
int FeatureManager::getFeatureCount()
{
    int cnt = 0;
    for (auto &it : feature) // 遍历特征点
    {

        it.used_num = it.feature_per_frame.size(); // 该特征点被观测到的所有帧的数量

        // 如果该特征点被至少两帧观测到了并且第一次观测到的帧索引start_frame不是在最后
        if (it.used_num >= 2 && it.start_frame < WINDOW_SIZE - 2)
        {
            cnt++; // 该特征点是有效的，滑动窗口中被跟踪特征点总数增加1
        }
    }
    return cnt;
}

/**
 * 对应论文解读：
 * 为什么要检查视差？
 * VINS中为了控制优化计算量，只对当前帧之前某一部分帧进行优化，而不是全部历史帧，局部优化帧数量的大小就是滑动窗口大小(系统中设置的值为10)
 * 为了维持窗口大小，需要去除旧帧添加新帧，也就是边缘化Marginalization。到底是删去最旧的帧（MARGIN_OLD）还是删去刚刚进来窗口倒数第二帧(MARGIN_SECOND_NEW)，
 * 就需要对当前帧与之前帧进行视差比较，如果是当前帧变化很小，就会删去倒数第二帧，如果变化很大，就删去最旧的帧。
 * 通过检测两帧之间的视差以及特征点数量决定次新帧是否作为关键帧
 * 关键帧选取策略：
 * 1. 当前帧相对最近的关键帧的特征平均视差大于一个阈值就为关键帧（因为视差可以根据平移和旋转共同得到，而纯旋转则导致不能三角化成功，所以这一步需要IMU预积分进行补偿）
 * 2. 当前帧跟踪到的特征点数量小于阈值视为关键帧
 * 对应论文IV-A
*/
/**
 * @brief           视差检查函数，特征点进入时检查视差，判断是否为关键帧
 * @Description     先把特征点从image中放入feature的list容器中，计算每一个点跟踪次数
 *                  和它在次新帧和次次新帧间所有特征点的平均视差，返回是否是关键帧
 * @param[in]       frame_count 当前滑动窗口中的frame个数
 * @param[in]       image 某帧所有特征点，第一个索引是特征点ID feature_id，第二个索引是观测到该特征点的相机帧ID camera_id，其他部分为每帧图像的基本数据xyz_uv_velocity
 * @param[in]       td IMU和cam同步时间差
 * @return          bool true：次新帧是关键帧；false：非关键帧
*/
bool FeatureManager::addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, double td)
{
    ROS_DEBUG("input feature: %d", (int)image.size()); // 输入的特征点数量
    ROS_DEBUG("num of feature: %d", getFeatureCount()); // // 滑动窗口中有效特征点的数量 
    double parallax_sum = 0; // 所有特征点视差总和
    int parallax_num = 0; // 次新帧与次次新帧中的共同特征点的数量
    last_track_num = 0; // 被跟踪点的个数，非新特征点的个数

    // 1. 把image map中的所有特征点放入feature list容器中
    // 遍历图像image中所有的特征点,看该特征点是否在已经记录了特征点的容器feature中，如果没在，则将<FeatureID,Start_frame>存入到Feature列表中；否则统计数目
    for (auto &id_pts : image) // 遍历所有特征点
    {
        // id_pts.second[0].second获取的信息为特征点对应的属性：xyz_uv_velocity << x, y, z, p_u, p_v, velocity_x, velocity_y
        // 这里的0不明白 ？？？
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td); // // _point 每帧的特征点[x,y,z,u,v,vx,vy]和td IMU和cam同步时间差，构造特征点在图像中的数据结构

        // 1.1 迭代器寻找feature list中是否有这feature_id
        int feature_id = id_pts.first; // 获取特征点ID：feature_id
        // 寻找列表中第一个使判别式为true的元素，这里第三个参数使用的是Lambda表达式，参考
        // find_if函数可以参考 https://blog.csdn.net/try_again_later/article/details/104900911 。意为：遍历feature list容器看看之前是否出现过当前的feature_id
        auto it = find_if(feature.begin(), feature.end(), [feature_id](const FeaturePerId &it)
                          {
            return it.feature_id == feature_id;
                          });

        // 1.2 如果没有查到则新建一个，并在feature管理器的list容器最后添加：FeaturePerId、FeaturePerFrame
        if (it == feature.end())
        {
            // 存储特征点格式：首先按照特征点ID，一个一个存储，每个ID会包含其在不同帧上的位置和其他信息
            feature.push_back(FeaturePerId(feature_id, frame_count)); // 特征点ID feature_id，首次观测到特征点的图像帧ID frame_count
            feature.back().feature_per_frame.push_back(f_per_fra); // 添加该特征点在该帧的位置和其他信息
        }
        // 1.3 之前有的话在FeaturePerFrame添加该特征点在该帧的f_per_fra，即位置和其他信息，并对跟踪到的特征点数进行累加
        else if (it->feature_id == feature_id)
        {
            /**
             * 如果找到了相同ID特征点，就在其FeaturePerFrame内增加此特征点在此帧的位置以及其他信息，
             * it的feature_per_frame容器中存放的是该feature能够被哪些帧看到，存放的是在这些帧中该特征点的信息
             * 所以，feature_per_frame.size的大小就表示有多少个帧可以看到该特征点
             * */
            it->feature_per_frame.push_back(f_per_fra);
            last_track_num++; // 表示此帧中有多少个和其他帧中相同的特征点能够被追踪到
        }
    }

    // 2. 窗口内只有一帧或者当前传入图像帧中能跟踪到的特征点数小于20个，则是关键帧
    if (frame_count < 2 || last_track_num < 20)
        return true;

    // 3. 计算每个特征在次新帧和次次新帧中的视差
    for (auto &it_per_id : feature) // 遍历每一个feature
    {
        // 计算能被当前帧和其前两帧共同看到的特征点视差
        // 观测到该特征点的帧的要求：起始帧小于倒数第三帧，终止帧要大于倒数第二帧，保证至少有两帧能观测到
        if (it_per_id.start_frame <= frame_count - 2 &&
            it_per_id.start_frame + int(it_per_id.feature_per_frame.size()) - 1 >= frame_count - 1) // it_per_id.feature_per_frame.size()表示该特征点能够被多少帧共视
        {
            parallax_sum += compensatedParallax2(it_per_id, frame_count); // 计算特征点it_per_id在倒数第二帧和倒数第三帧之间的视差，并求所有视差的累加和
            parallax_num++; // 所有具有视差的特征点个数
        }
    }

    // 4. 第一次加进来的，则是关键帧
    if (parallax_num == 0) // 视差等于零，说明没有共同特征，也就是全新帧
    {
        return true;
    }
    else
    {
        ROS_DEBUG("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        ROS_DEBUG("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        // 5. 平均视差大于阈值的，则是关键帧
        return parallax_sum / parallax_num >= MIN_PARALLAX; // 视差总和除以参与计算(具有)视差的特征点个数，表示每个特征点的平均视差值，MIN_PARALLAX=10.0/460.0，阈值为10个像素
    }
}

#if 0 
调试代码
void FeatureManager::debugShow()
{
    ROS_DEBUG("debug show");
    for (auto &it : feature)
    {
        ROS_ASSERT(it.feature_per_frame.size() != 0);
        ROS_ASSERT(it.start_frame >= 0);
        ROS_ASSERT(it.used_num >= 0);

        ROS_DEBUG("%d,%d,%d ", it.feature_id, it.used_num, it.start_frame);
        int sum = 0;
        for (auto &j : it.feature_per_frame)
        {
            ROS_DEBUG("%d,", int(j.is_used));
            sum += j.is_used;
            printf("(%lf,%lf) ",j.point(0), j.point(1));
        }
        ROS_ASSERT(it.used_num == sum);
    }
}
#endif

/**
 * @brief           计算frame_count_l和frame_count_r两帧匹配(共视)的特征点对，获取特征点的3D坐标
 * @Description     如果某个特征点能被观测到的第一帧和最后一帧之间的范围大，也就是[start_frame,endFrame()]是个大范围，
 *                  而窗口[frame_count_l, frame_count_r]被包含进去了，那么可以直接获取特征点的3D坐标
 * @param[in]       frame_count_l 需要匹配的第一帧图像
 * @param[in]       frame_count_r 需要匹配的第二帧图像
 * @return          frame_count_l和frame_count_r两帧匹配的特征点3D坐标对
*/
vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r)
{
    vector<pair<Vector3d, Vector3d>> corres; // 存储3D点对的容器列表
    for (auto &it : feature) // 通过feature的list容器
    {
        // 要找的特征点的两帧在窗口范围内，可以直接取。窗口为：观测到当前特征点的所有图像帧，也就是[start_frame,endFrame()]
        if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r)
        {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            // 计算传入的frame_count_l和frame_count_r距离最开始一帧的距离，也就是最新两帧的index
            int idx_l = frame_count_l - it.start_frame; // 当前帧-第一次观测到特征点的帧数，也就是当前帧的索引
            int idx_r = frame_count_r - it.start_frame;

            a = it.feature_per_frame[idx_l].point; // 获取该帧上该特征点的3D坐标

            b = it.feature_per_frame[idx_r].point;
            
            corres.push_back(make_pair(a, b)); // 存放每一个特征点it在两个帧中对应的3D点对
        }
    }
    return corres;
}

/**
 * @brief           设置所有有效特征点的逆深度估计值
 * @Description     取有效特征点，也就是该特征点被至少两帧观测到了并且第一次观测到的帧索引不是在最后，设置有效特征点的逆深度值
 * @param[in]       x 所有特征点的深度值向量
 * @return          void
*/
void FeatureManager::setDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature) // 遍历所有特征点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // 该特征点it_per_id被观测到的所有帧的数量

        // 该特征点被至少两帧观测到了并且第一次观测到的帧索引start_frame不是在最后，也就是第一次观测到的帧在滑动窗口内
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2)) 
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index); // 求解逆深度

        if (it_per_id.estimated_depth < 0)
        {
            it_per_id.solve_flag = 2; // 深度小于0则估计失败
        }
        else
            it_per_id.solve_flag = 1; // 深度大于等于0则估计成功
    }
}

/**
 * @brief           删除深度估计失败的特征点信息       
 * @Description     通过判断某特征点估计求解器solve_flag的值，若深度估计失败则删除该特征点的信息
 * @param[in]       void
 * @return          void
*/
void FeatureManager::removeFailures()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) // 遍历滑动窗口内所有的特征点
    {
        it_next++;
        if (it->solve_flag == 2) // 某特征点的深度估计失败（setDepth函数中有失败的处理），即深度值小于0，则删除该特征点
            feature.erase(it); // 从list容器feature中删除该特征点it的信息
    }
}

/**
 * @brief           初始化（还原）有效特征点的逆深度值     
 * @Description     取有效特征点，也就是该特征点被至少两帧观测到了并且第一次观测到的帧索引不是在最后，还原有效特征点的逆深度值为-1
 * @param[in]       x 所有特征点的深度值向量，这里所有元素都为-1
 * @return          void
*/
void FeatureManager::clearDepth(const VectorXd &x)
{
    int feature_index = -1;
    for (auto &it_per_id : feature) // 遍历所有特征点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // 该特征点it_per_id被观测到的所有帧的数量

        // 该特征点被至少两帧观测到了并且第一次观测到的帧索引start_frame不是在最后，也就是第一次观测到的帧在滑动窗口内
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index); // 通过传入的深度值向量对有效特征点的逆深度赋值
    }
}

/**
 * @brief           获取所有有效特征点的深度值
 * @Description     取有效特征点it_per_id，也就是该特征点被至少两帧观测到了并且第一次观测到的帧索引不是在最后，获取有效特征点的逆深度估计值，并转换为深度值对深度值向量dep_vec赋值
 * @param[in]       void
 * @return          dep_vec 深度值向量
*/
VectorXd FeatureManager::getDepthVector()
{
    VectorXd dep_vec(getFeatureCount()); // 定义深度向量，维度为滑动窗口中被跟踪特征点的总数，通过getFeatureCount()得到
    int feature_index = -1;
    for (auto &it_per_id : feature) // 遍历所有特征点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // 该特征点it_per_id被观测到的所有帧的数量

        // 该特征点被至少两帧观测到了并且第一次观测到的帧索引start_frame不是在最后，也就是第一次观测到的帧在滑动窗口内
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
#if 1
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth; // 获取深度向量，1除以逆深度估计值
#else
        dep_vec(++feature_index) = it_per_id->estimated_depth;
#endif
    }
    return dep_vec;
}

/**
 * @brief           对滑动窗口中所有帧中depth未知的特征点进行三角化     
 * @Description     https://blog.csdn.net/u012101603/article/details/79714332
 * @param[in]       Ps[] IMU到世界坐标系的平移量
 * @param[in]       tic[] 相机和IMU外参的平移分量t^b_c
 * @param[in]       ric[] 相机和IMU外参的旋转分量R^b_c  
 * @return          void
*/
void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[])
{
    for (auto &it_per_id : feature) // 遍历所有特征点
    {
        it_per_id.used_num = it_per_id.feature_per_frame.size(); // 该特征点被观测到的所有帧的数量

        // 该特征点被至少两帧观测到了并且第一次观测到的帧索引start_frame不是在最后，也就是第一次观测到的帧在滑动窗口内，此为有效特征点
        if (!(it_per_id.used_num >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        // 该特征点估计的逆深度需要小于0，测试结果为-1
        if (it_per_id.estimated_depth > 0)
            continue;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1; // 第一次观测到该特征点的帧以及它的前一帧

        ROS_ASSERT(NUM_OF_CAM == 1); // 相机个数需要为1，也就是单目，否则断言判断为假，终止程序执行
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame.size(), 4); // 定义三角化最小二乘问题中的系统矩阵A，维度为(2×帧数)*4
        int svd_idx = 0;

        //R0 t0为第i帧相机坐标系到世界坐标系的变换矩阵Rwc，T^w_ci = (T^w_bi) * (T^bi_ci)
        // T^w_ci = [R^w_ci  t^w_ci]
        //          [  0        1  ]
        // T^w_bi = [R^w_bi  t^w_bi]
        //          [  0        1  ]
        // T^bi_ci = [R^bi_ci  t^bi_ci] = [R^b_c  t^b_c] 外参
        //           [   0        1   ]   [  0     1   ]
        // T^w_ci = (T^w_bi) * (T^bi_ci) = [(R^w_bi)*(R^b_c)  (t^w_bi)+(R^w_bi)*t^b_c] = [R^w_ci  t^w_ci]
        //                                 [        0                       1        ]   [  0        1  ]
        Eigen::Matrix<double, 3, 4> P0; // 定义第i帧的投影矩阵P0，维度为3*4
        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0]; // t0为第i帧相机坐标系到世界坐标系的平移分量，t^w_ci = t^w_bi + R^w_bi * t^b_c
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0]; // R0为第i帧相机坐标系到世界坐标系的旋转分量，也就是旋转矩阵 R^w_ci = R^w_bi * R^b_c
        
        // 将第i帧的投影矩阵P0设为  
        // [1 0 0 0]
        // [0 1 0 0]
        // [0 0 1 0]
        P0.leftCols<3>() = Eigen::Matrix3d::Identity();
        P0.rightCols<1>() = Eigen::Vector3d::Zero();

        for (auto &it_per_frame : it_per_id.feature_per_frame) // 遍历某特征点共视的帧
        {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0]; // t0为第j帧相机坐标系到世界坐标系的平移分量，t^w_cj = t^w_bj + R^w_bj * t^b_c
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0]; // R0为第j帧相机坐标系到世界坐标系的旋转分量，也就是旋转矩阵 R^w_cj = R^w_bj * R^b_c
            Eigen::Vector3d t = R0.transpose() * (t1 - t0); // t为第j帧相机坐标系到第i帧相机坐标系变换的平移分量，也就是两帧之间的平移 t^ci_cj = R^ci_w * (t^w_cj - t^w_ci) = (R^w_ci)^T * (t^w_cj - t^w_ci)
            Eigen::Matrix3d R = R0.transpose() * R1; // R为第j帧相机坐标系到第i帧相机坐标系变换的旋转分量，也就是两帧之间的旋转矩阵 R^ci_cj = R^ci_w * R^w_cj = (R^w_ci)^T * R^w_cj
            Eigen::Matrix<double, 3, 4> P; // 定义第j帧的投影矩阵P，维度为3*4

            // 将第j帧的投影矩阵P设为[R^cj_ci | t^cj_ci]
            P.leftCols<3>() = R.transpose(); // 第i帧相机坐标系到第j帧相机坐标系的旋转矩阵 R^cj_ci = (R^ci_cj)^T
            P.rightCols<1>() = -R.transpose() * t; // 第i帧相机坐标系到第j帧相机坐标系的平移 t^cj_ci = -(R^ci_cj)^T * t^ci_cj
            Eigen::Vector3d f = it_per_frame.point.normalized(); // 特征点在归一化相机坐标系上的坐标

            // 每个特征点对A矩阵贡献两个方程
            // P = [P1 P2 P3]^T 行向量形式
            // AX = 0  A = [A(2*i) A(2*i+1) A(2*i+2) A(2*i+3) ...]^T
            // A(2*i) = x(i) * P3 - z(i) * P1
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            // A(2*i+1) = y(i) * P3 - z(i) * P2
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        ROS_ASSERT(svd_idx == svd_A.rows());
        // 对A进行SVD分解得到其最小奇异值对应的单位奇异向量(x,y,z,w)
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3]; // 深度为z/w

        it_per_id.estimated_depth = svd_method; // 深度值赋值

        if (it_per_id.estimated_depth < 0.1)
        {
            it_per_id.estimated_depth = INIT_DEPTH; // INIT_DEPTH = 5.0
        }

    }
}

#if 0
void FeatureManager::removeOutlier()
{
    ROS_BREAK();
    int i = -1;
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next)
    {
        it_next++;
        i += it->used_num != 0;
        if (it->used_num != 0 && it->is_outlier == true)
        {
            feature.erase(it);
        }
    }
}
#endif

/**
 * @brief           对应论文VI-D，边缘化最老帧时处理深度值  
 * @Description     边缘化最老帧时，处理特征点保存的帧号，将帧号向前滑动，同时将起始帧是最老帧的特征点的深度值进行转移，转移到下一帧中
 * @param[in]       marg_R 需要被边缘化帧的旋转矩阵，即 R^w_ci
 * @param[in]       marg_P 需要被边缘化帧的平移量，即 t^w_ci
 * @param[in]       new_R 需要被边缘化帧的下一帧的旋转矩阵，即 R^w_cj
 * @param[in]       new_P 需要被边缘化帧的下一帧的平移量，t^w_cj
 * @return          void
*/
void FeatureManager::removeBackShiftDepth(Eigen::Matrix3d marg_R, Eigen::Vector3d marg_P, Eigen::Matrix3d new_R, Eigen::Vector3d new_P)
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) // 遍历滑动窗口内所有的特征点
    {
        it_next++;

        // 特征点起始帧如果不是最老的帧，也就是不为零，则将起始帧帧号减一
        if (it->start_frame != 0)
            it->start_frame--;
        else
        {
            // 特征点起始帧是最老的帧
            Eigen::Vector3d uv_i = it->feature_per_frame[0].point; // 该特征点在最老帧归一化相机坐标系下的3D坐标
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // 移除掉该特征点的最老帧
            // 该特征点只在最老帧被观测，也就是该特征点被观测到的帧数量为1或者0，则直接移除该特征点
            if (it->feature_per_frame.size() < 2)
            {
                feature.erase(it); // 删除特征点
                continue;
            }
            // 除了最老帧可以观测到该特征点，还有其他帧也观测到了该特征点
            else
            {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth; // pts_i 为该特征点在最老帧相机坐标系下的3D坐标 p_ci，也就是归一化坐标乘以深度
                Eigen::Vector3d w_pts_i = marg_R * pts_i + marg_P; // w_pts_i 为该特征点在世界坐标系下的3D坐标 p_w，即 p_w = (R^w_ci) * p_ci + t^w_ci
                Eigen::Vector3d pts_j = new_R.transpose() * (w_pts_i - new_P); // pts_j 为该特征点在最老帧的下一帧相机坐标系下的3D坐标，p_w = (R^w_cj) * p_cj + t^w_cj -> p_cj = (R^w_cj)^T * (p_w - (t^w_cj))
                double dep_j = pts_j(2); // 该特征点在最老帧的下一帧相机坐标系下的深度
                if (dep_j > 0)
                    it->estimated_depth = dep_j; // 该特征点的深度
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
    }
}

/**
 * @brief           对应论文VI-D，边缘化最老帧时处理特征点所保存的帧号
 * @Description     边缘化最老帧时，直接将特征点所保存的帧号向前滑动，也就是对特征点在最老帧的信息进行移除处理
 * @return          void
*/
void FeatureManager::removeBack()
{
    for (auto it = feature.begin(), it_next = feature.begin();
         it != feature.end(); it = it_next) // 遍历滑动窗口内所有的特征点
    {
        it_next++;

        // 特征点起始帧如果不是最老的帧，也就是不为零，则将起始帧帧号减一
        if (it->start_frame != 0)
            it->start_frame--;

        // 如果start_frame为0则直接移除feature_per_frame中第0帧对应的FeaturePerFrame
        else
        {
            it->feature_per_frame.erase(it->feature_per_frame.begin()); // 移除feature_per_frame的第0帧FeaturePerFrame

            // 如果feature_per_frame为空，也就是该特征点只在最老帧被观测，则直接删除特征点
            if (it->feature_per_frame.size() == 0)
                feature.erase(it); // 删除特征点
        }
    }
}

/**
 * @brief           对应论文VI-D，边缘化次新帧处理特征点所保存的帧号  
 * @Description     边缘化次新帧时，对特征点在次新帧的信息进行移除处理
 * @param[in]       frame_count 当前滑动窗口中的frame个数
 * @return          void
*/
void FeatureManager::removeFront(int frame_count)
{
    for (auto it = feature.begin(), it_next = feature.begin(); 
         it != feature.end(); it = it_next) // 遍历滑动窗口内所有的特征点
    {
        it_next++;

        // 特征点起始帧如果是最新帧，则将起始帧帧号减一
        if (it->start_frame == frame_count)
        {
            it->start_frame--;
        }
        else
        {
            int j = WINDOW_SIZE - 1 - it->start_frame;

            // 如果次新帧之前已经跟踪结束则什么都不做
            if (it->endFrame() < frame_count - 1)
                continue;

            //如果次新帧仍被跟踪，则删除feature_per_frame中次新帧对应的FeaturePerFrame
            it->feature_per_frame.erase(it->feature_per_frame.begin() + j);

            // 如果feature_per_frame为空，也就是该特征点只在次新帧被观测，则直接删除特征点
            if (it->feature_per_frame.size() == 0)
                feature.erase(it);
        }
    }
}

/**
 * @brief           计算某个特征点it_per_id在次新帧(倒数第二帧)和次次新帧(倒数第三帧)之间的视差
 * @Description     判断观测到该特征点的frame中倒数第二帧和倒数第三帧的共视关系 
 *                  实际是求取该特征点在两帧的归一化平面上的坐标点的距离
 * @param[in]       it_per_id 从特征点list上取下来的一个feature
 * @param[in]       frame_count 当前滑动窗口中的frame个数
 * @return          double ans：两帧中该特征的坐标点距离
*/
double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count)
{
    //check the second last frame is keyframe or not
    //parallax betwwen seconde last frame and third last frame
    // feature_per_frame[]表示包含这个特征的关键帧的管理器
    const FeaturePerFrame &frame_i = it_per_id.feature_per_frame[frame_count - 2 - it_per_id.start_frame]; // 窗口内倒数第三帧i
    const FeaturePerFrame &frame_j = it_per_id.feature_per_frame[frame_count - 1 - it_per_id.start_frame]; // 窗口内倒数第二帧j

    double ans = 0;
    Vector3d p_j = frame_j.point; // 倒数第二帧j的3D路标点坐标

    // 因为特征点都是归一化之后的点，所以深度都为1，这里没有对倒数第二帧j的3D路标点去除深度
    double u_j = p_j(0);
    double v_j = p_j(1);

    Vector3d p_i = frame_i.point; // 倒数第三帧i的3D路标点坐标
    Vector3d p_i_comp;

    p_i_comp = p_i;

    // p_i(2)就是深度值，这里是归一化坐标所以z为1，也就是对倒数第三帧i的3D路标点去除深度，和j的操作不一样
    double dep_i = p_i(2);
    double u_i = p_i(0) / dep_i;
    double v_i = p_i(1) / dep_i;
    double du = u_i - u_j, dv = v_i - v_j;

    // 和 p_i 一样的操作
    double dep_i_comp = p_i_comp(2);
    double u_i_comp = p_i_comp(0) / dep_i_comp;
    double v_i_comp = p_i_comp(1) / dep_i_comp;
    double du_comp = u_i_comp - u_j, dv_comp = v_i_comp - v_j; // 和du的计算是一样的，结果也应该是一样的

    // 视差距离计算，这里min中的两个平方距离的计算应该是一样的，因为p_i_comp和p_i是一样的
    // sqrt的结果为非负，不需要和0做max的比较
    ans = max(ans, sqrt(min(du * du + dv * dv, du_comp * du_comp + dv_comp * dv_comp)));

    return ans;
}