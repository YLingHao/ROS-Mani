#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <std_msgs/Float64.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Pose2D.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>
#include <pcl/visualization/cloud_viewer.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>
#include <tf_conversions/tf_eigen.h>
#include <wpb_mani_behaviors/Coord.h>
#include <map>
#include <vector>
#include <string>
#include <dirent.h>
#include <sys/stat.h>
#include <cctype>
#include <locale>
#include <codecvt>

using namespace cv;
using namespace std;
using namespace ros;

// 函数声明
void BoxCoordinateReceived(const wpb_mani_behaviors::Coord::ConstPtr &msg);
void GrabResultArrived(const std_msgs::String::ConstPtr &msg);
void ArmVelocityCallback(const geometry_msgs::Twist::ConstPtr &msg);
void setGripperState(double pos);

// 点云处理相关函数声明
void HandlePointCloudData(const sensor_msgs::PointCloud2::ConstPtr& msg);
void FindObjectsInCloud();

// 抓取控制相关函数声明
void ExecuteGrabSequence();
void ExtendArmToTarget();
void AdjustGripper(double pos);
void SendMotionCommand(double vx, double vy, double wz);
void AlignWithTarget();
void PerformGrabAction();
void RetractArmToHome();

// 图像验证相关函数声明
bool waitForFreshImage();
bool confirmBoxPickup(cv::Mat& img);

// 任务状态机
enum OperationPhase {
    STARTUP,                        // 启动
    NAVIGATE_TO_POINT1,             // 导航到点1
    NAVIGATE_TO_POINT2,             // 导航到点2
    SHIFT_LEFT_INITIAL,             // 初始左移
    CHECK_RED_BOX_DARKNESS,         // 检查红盒暗区
    SHIFT_RIGHT_POST_DETECT,        // 检测后右移
    VERIFY_RED_BOX_AFTER_SHIFT,     // 移动后验证红盒
    ADVANCE_TO_TOUCH,               // 前进接触
    PERFORM_TOUCH_ACTION,           // 执行接触动作
    NAVIGATE_TO_POINT3,             // 导航到点3
    SCAN_FOR_OBJECTS,               // 扫描物体
    MOVE_TO_GRAB_POSITION,          // 移动到抓取位置
    INITIATE_GRAB_PROCESS,          // 启动抓取流程
    NAVIGATE_TO_POINT4,             // 导航到点4
    IDENTIFY_COLOR_OBJECTS,         // 识别颜色物体
    APPROACH_COLOR_TARGET,          // 接近颜色目标
    MOVE_TOWARD_TARGET,             // 向目标移动
    EXECUTE_PLACEMENT,              // 执行放置
    NAVIGATE_TO_POINT5,             // 导航到点5
    RECOGNIZE_NUMERALS,             // 识别数字
    PERFORM_ROTATION,               // 执行旋转
    DETECT_SPECIAL_MARKERS,         // 检测特殊标记
    RETURN_TO_START,                // 返回起点
    MISSION_COMPLETE                // 任务完成
};

// 全局变量
OperationPhase currentPhase = STARTUP;
Publisher navPointPublisher, armController, gripperController, motionPublisher;
Subscriber imageSubscriber, navigationSubscriber;
string objectColor;
string matchedColor;        // 存储匹配的颜色
string darkBlockOrientation; // 存储暗块方向
bool initialDetectionDone = false, objectDetectionDone = false, numberDetectionDone = false, markerDetectionDone = false;
int collectedCount = 0;
const int MAX_OBJECTS = 3;
std_msgs::String navigationMessage;
string detectionSide = "";

// 图像相关全局变量
cv::Mat currentFrame;
bool newImageAvailable = false;
ros::Time lastImageTime;
int detectedNumber = 0;

// 物体检测相关全局变量
bool receivedObjectCoordinates = false; // 是否收到物体坐标
int foundObjectsCount = 0; // 发现的物体数量
float primaryObjectX = 0.0; // 主要物体X坐标
float primaryObjectY = 0.0; // 主要物体Y坐标
float primaryObjectZ = 0.0; // 主要物体Z坐标

// 机械臂模块相关全局变量
bool armSystemActive = false; // 机械臂系统激活标志
bool grabOperationSuccess = false; // 抓取操作成功标志
bool grabInProgressFlag = false; // 抓取进行中标志
bool coordinatesTransmitted = false; // 坐标已发送标志
bool resultNotificationReceived = false; // 结果通知收到标志
int grabAttemptCounter = 0; // 抓取尝试计数器
geometry_msgs::Pose grabTargetMessage; // 抓取目标消息
ros::Publisher grabTargetPublisher; // 抓取目标发布器
ros::Publisher surfaceHeightPublisher; // 表面高度发布器

// 机械臂模块启动相关发布器
ros::Publisher objectScanPublisher; // 物体扫描发布器

// 点云处理相关全局变量
pcl::PointCloud<pcl::PointXYZ>::Ptr rawCloudData; // 原始点云数据
pcl::PointCloud<pcl::PointXYZ>::Ptr processedCloud; // 处理后的点云
ros::Subscriber cloudDataSubscriber; // 点云数据订阅器
bool cloudDataReceived = false; // 点云数据接收标志
float detectedSurfaceHeight = 0.0; // 检测到的表面高度
vector<vector<float>> locatedObjects; // 定位的物体列表
ros::Publisher objectsLocationPublisher; // 物体位置发布器
ros::Publisher visualMarkersPublisher; // 可视化标记发布器

// 抓取控制相关全局变量
enum GrabSequence {
    PHASE_ALIGNMENT,      // 对齐阶段
    PHASE_EXTENSION,      // 伸展阶段
    PHASE_APPROACH,       // 接近阶段
    PHASE_GRAB_ACTION,    // 抓取动作阶段
    PHASE_RETRACTION      // 收回阶段
};
GrabSequence currentGrabPhase = PHASE_ALIGNMENT; // 当前抓取阶段
bool grabSequenceActive = false; // 抓取序列激活标志
bool grabTimeExceeded = false; // 抓取超时标志
ros::Time grabSequenceStart; // 抓取序列开始时间
float targetGrabPositionX = 0.0; // 目标抓取位置X
float targetGrabPositionY = 0.0; // 目标抓取位置Y
float targetGrabPositionZ = 0.0; // 目标抓取位置Z
int grabRetryCounter = 0; // 抓取重试计数器
const int MAX_GRAB_ATTEMPTS = 3; // 最大抓取尝试次数

// 物体信息结构
struct ObjectData {
    string color;
    int pixelDisplacement;
    int centerX;
    int x;        // 中心X坐标
    int y;        // 中心Y坐标
    int width;    // 宽度
    int height;   // 高度
    double area;  // 面积
};

// 图像回调函数
void imageDataCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        currentFrame = cv_ptr->image;
        newImageAvailable = true;
        lastImageTime = ros::Time::now();
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge异常: %s", e.what());
        return;
    }
}

// 红盒底部暗区检测函数
bool analyzeRedBoxDarkRegion(Mat& frame, double& dark_ratio) {
    if (frame.empty()) {
        ROS_ERROR("[红盒分析] 图像为空");
        return false;
    }
    
    // 转换为HSV颜色空间分析红盒
    Mat hsvFrame;
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV);
    
    // 红色范围（HSV空间）
    Scalar redLower1 = Scalar(0, 50, 50);
    Scalar redUpper1 = Scalar(10, 255, 255);
    Scalar redLower2 = Scalar(170, 50, 50);
    Scalar redUpper2 = Scalar(180, 255, 255);
    
    Mat maskA, maskB, redMask;
    inRange(hsvFrame, redLower1, redUpper1, maskA);
    inRange(hsvFrame, redLower2, redUpper2, maskB);
    redMask = maskA | maskB;
    
    // 形态学操作
    Mat kernelElement = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(redMask, redMask, MORPH_CLOSE, kernelElement);
    
    // 查找红盒轮廓
    vector<vector<Point>> contourList;
    findContours(redMask, contourList, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contourList.empty()) {
        ROS_WARN("[红盒分析] 未检测到红盒");
        return false;
    }
    
    // 查找最大红盒
    double maxArea = 0;
    Rect redBoxRect;
    for (const auto& contour : contourList) {
        Rect rect = boundingRect(contour);
        double area = rect.area();
        if (area > maxArea && area > 1000) { // 面积阈值1000像素
            maxArea = area;
            redBoxRect = rect;
        }
    }
    
    if (maxArea == 0) {
        ROS_WARN("[红盒分析] 未找到有效红盒");
        return false;
    }
    
    // 提取红盒下半部分
    int bottomHeight = redBoxRect.height / 2;
    Rect bottomRegion(redBoxRect.x, redBoxRect.y + redBoxRect.height / 2, 
                      redBoxRect.width, bottomHeight);
    
    if (bottomRegion.width <= 0 || bottomRegion.height <= 0) {
        ROS_WARN("[红盒分析] 底部区域无效");
        return false;
    }
    
    Mat bottomArea = frame(bottomRegion);
    
    // 转换底部区域到HSV进行暗色检测
    Mat hsvBottom;
    cvtColor(bottomArea, hsvBottom, COLOR_BGR2HSV);
    
    // 定义暗色范围（HSV空间）
    Scalar darkLower = Scalar(0, 0, 0);
    Scalar darkUpper = Scalar(180, 255, 50);
    
    // 创建底部区域暗色掩码
    Mat darkMask;
    inRange(hsvBottom, darkLower, darkUpper, darkMask);
    
    // 形态学操作去除噪声
    morphologyEx(darkMask, darkMask, MORPH_CLOSE, kernelElement);
    
    // 计算底部区域暗色面积
    double darkArea = countNonZero(darkMask);
    double redBoxHalfArea = redBoxRect.area() / 2.0; // 与红盒一半面积比较
    
    if (redBoxHalfArea == 0) {
        ROS_WARN("[红盒分析] 红盒半面积为零");
        return false;
    }
    
    dark_ratio = darkArea / redBoxHalfArea;
    
    ROS_INFO("[红盒分析] 红盒面积: %.0f, 红盒半面积: %.0f, 暗色面积: %.0f, 暗区比例: %.2f", 
             redBoxRect.area(), redBoxHalfArea, darkArea, dark_ratio);
    
    return true;
}

// 数字识别函数
bool identifyNumberInRedBox(Mat& frame, int& number) {
    if (frame.empty()) {
        ROS_ERROR("[数字识别] 图像为空");
        return false;
    }
    
    // 转换为HSV颜色空间检测红盒
    Mat hsvFrame;
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV);
    
    // 红色范围（HSV空间）
    Scalar redLower1 = Scalar(0, 50, 50);
    Scalar redUpper1 = Scalar(10, 255, 255);
    Scalar redLower2 = Scalar(170, 50, 50);
    Scalar redUpper2 = Scalar(180, 255, 255);
    
    Mat maskA, maskB, redMask;
    inRange(hsvFrame, redLower1, redUpper1, maskA);
    inRange(hsvFrame, redLower2, redUpper2, maskB);
    redMask = maskA | maskB;
    
    // 形态学操作
    Mat kernelElement = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(redMask, redMask, MORPH_CLOSE, kernelElement);
    
    // 查找红盒轮廓
    vector<vector<Point>> contourList;
    findContours(redMask, contourList, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contourList.empty()) {
        ROS_WARN("[数字识别] 未检测到红盒");
        return false;
    }
    
    // 查找最大红盒
    double maxArea = 0;
    Rect redBoxRect;
    for (const auto& contour : contourList) {
        Rect rect = boundingRect(contour);
        double area = rect.area();
        if (area > maxArea && area > 1000) { // 面积阈值1000像素
            maxArea = area;
            redBoxRect = rect;
        }
    }
    
    if (maxArea == 0) {
        ROS_WARN("[数字识别] 未找到有效红盒");
        return false;
    }
    
    // 提取红盒内部区域（缩小10%避免边框干扰）
    int marginX = redBoxRect.width * 0.1;
    int marginY = redBoxRect.height * 0.1;
    Rect innerRegion(redBoxRect.x + marginX, redBoxRect.y + marginY, 
                     redBoxRect.width - 2 * marginX, redBoxRect.height - 2 * marginY);
    
    if (innerRegion.width <= 0 || innerRegion.height <= 0) {
        ROS_WARN("[数字识别] 内部区域无效");
        return false;
    }
    
    // 修复：将变量名改为numberRegion避免冲突
    Mat numberRegion = frame(innerRegion);
    
    // 转换为灰度图
    Mat grayImage;
    cvtColor(numberRegion, grayImage, COLOR_BGR2GRAY);
    
    // 二值化
    Mat binaryImage;
    threshold(grayImage, binaryImage, 128, 255, THRESH_BINARY);
    
    // 查找数字轮廓
    vector<vector<Point>> numberContours;
    findContours(binaryImage, numberContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (numberContours.empty()) {
        ROS_WARN("[数字识别] 未检测到数字轮廓");
        return false;
    }
    
    // 查找最大数字轮廓
    Rect numberRect;
    for (const auto& contour : numberContours) {
        Rect rect = boundingRect(contour);
        if (rect.area() > numberRect.area()) {
            numberRect = rect;
        }
    }
    
    // 根据面积比（数字面积/红盒面积）判断数字
    double redBoxArea = redBoxRect.area();
    // 修复：将变量名改为numberAreaValue避免冲突
    double numberAreaValue = numberRect.area();
    double areaRatio = numberAreaValue / redBoxArea;
    ROS_INFO("[数字识别] 红盒面积: %.0f, 数字面积: %.0f, 面积比: %.2f", 
             redBoxArea, numberAreaValue, areaRatio);
    
    if (areaRatio < 0.3) {
        number = 1;
        ROS_INFO("[数字识别] 识别结果: 数字1 (面积比 < 30%%)");
    } else {
        number = 2;
        ROS_INFO("[数字识别] 识别结果: 数字2 (面积比 >= 30%%)");
    }
    
    return true;
}

// 物体和机器人坐标同步相关
static ros::Subscriber objectRobotSyncSubscriber; // 物体机器人同步订阅器
static ros::Subscriber robotPositionSubscriber;    // 机器人位置订阅器
static double robotPosX = 0.0;       // 机器人X坐标
static double robotPosY = 0.0;       // 机器人Y坐标
static double robotOrientation = 0.0;    // 机器人朝向角

// 缺失的全局变量
double currentYPosition = 0.0;
int rotationAngle = 0;

// TF坐标变换相关
tf::TransformListener* coordinateTransformer = nullptr;

// 图像缓存和验证相关

// 删除定时器相关全局变量

// 机械臂预设姿态
const vector<double> ARM_HOME_POSITION = {0, -1.57, 1.35, 0.24};
const vector<double> ARM_TOUCH_POSITION = {0, 0.9, -0.4, -0.2};
const vector<double> ARM_GRAB_READY = {0, -0.7, 0.5, -0.3};
const vector<double> ARM_DOWN_POSITION = {0, -0.7, 0.5, -0.3};
const vector<double> ARM_PLACE_POSITION = {0, 1.57, -0.94, -0.59};
const double GRIPPER_OPEN_STATE = 1.0;
const double GRIPPER_CLOSE_STATE = 0.035;

// 侧向移动
void performLateralMovement(const string& dir, double duration = 2.0, double speed = 1.0, bool checkObjectDetection = false) {
    geometry_msgs::Twist motionCommand;
    motionCommand.linear.x = 0;
    motionCommand.linear.y = (dir == "left") ? speed : -speed;
    motionCommand.angular.z = 0;

    ROS_INFO("[移动控制] 向%s侧向移动%.2f秒", dir.c_str(), duration);
    ros::Time beginTime = ros::Time::now();
    bool objectDetected = false;
    
    while ((ros::Time::now() - beginTime).toSec() < duration && ros::ok()) {
        motionPublisher.publish(motionCommand);
        
        // 如果启用了物体检测中断，检查是否检测到物体
        if (checkObjectDetection) {
            // 检查是否有有效的物体坐标
            if (receivedObjectCoordinates && foundObjectsCount > 0) {
                ROS_INFO("[移动控制] 检测到物体，立即停止移动");
                objectDetected = true;
                break;
            }
        }
        
        ros::Duration(0.01).sleep();
    }
    
    motionCommand.linear.y = 0;
    motionPublisher.publish(motionCommand);
    ros::Duration(0.3).sleep();
    
    if (objectDetected) {
        ROS_INFO("[移动控制] 侧向移动被中断，检测到物体");
    }
}

// 前进移动
void moveForward(double duration = 2.0, double speed = 1.2) {
    geometry_msgs::Twist motionCommand;
    motionCommand.linear.x = speed;
    motionCommand.linear.y = 0;
    motionCommand.angular.z = 0;

    ROS_INFO("[移动控制] 前进%.2f秒（约%.2f米）", duration, speed * duration);
    ros::Time beginTime = ros::Time::now();
    while ((ros::Time::now() - beginTime).toSec() < duration && ros::ok()) {
        motionPublisher.publish(motionCommand);
        ros::Duration(0.01).sleep();
    }
    motionCommand.linear.x = 0;
    motionPublisher.publish(motionCommand);
    ros::Duration(0.3).sleep();
}

// 后退移动函数
void moveBackward(double duration = 1.0, double speed = 1.0) {
    geometry_msgs::Twist motionCommand;
    motionCommand.linear.x = -speed;
    motionCommand.linear.y = 0;
    motionCommand.angular.z = 0;

    ROS_INFO("[移动控制] 后退%.2f秒", duration);
    ros::Time beginTime = ros::Time::now();
    while ((ros::Time::now() - beginTime).toSec() < duration && ros::ok()) {
        motionPublisher.publish(motionCommand);
        ros::Duration(0.01).sleep();
    }
    motionCommand.linear.x = 0;
    motionPublisher.publish(motionCommand);
    ros::Duration(0.3).sleep();
}

// 机械臂控制
void sendArmCommand(const vector<double>& angles) {
    sensor_msgs::JointState armMessage;
    armMessage.name = {"joint1", "joint2", "joint3", "joint4"};
    armMessage.position = angles;
    // 使用机械臂模块推荐的关节速度参数
    armMessage.velocity = {0.35, 0.35, 0.35, 0.35}; // 基础速度0.35
    armMessage.velocity[1] = 1.05; // 关节2速度调整为1.05
    armMessage.velocity[3] = 0.5;  // 关节4速度调整为0.5
    ROS_INFO("[机械臂控制] 发送姿态: 关节1=%.2f, 关节2=%.2f, 关节3=%.2f, 关节4=%.2f",
             angles[0], angles[1], angles[2], angles[3]);
    for (int i = 0; i < 3; i++) {
        armController.publish(armMessage);
        ros::Duration(0.1).sleep();
    }
    ros::Duration(0.5).sleep(); // 减少等待时间
}

// 同时控制机械臂和夹爪的函数

// 坐标转换函数：从图像坐标到世界坐标
bool convertImageToWorld(int imgX, int imgY, double& worldX, double& worldY) {
    if (!coordinateTransformer) {
        ROS_ERROR("[坐标转换] 坐标转换器未初始化");
        return false;
    }
    
    // 假设相机内参
    double focalX = 525.0;  // 焦距X
    double focalY = 525.0;  // 焦距Y
    double centerX = 320.0;  // 中心X
    double centerY = 240.0;  // 中心Y
    
    // 假设物体距离相机的距离
    double objectDistance = 0.5;  // 米
    
    // 计算归一化图像坐标
    double normX = (imgX - centerX) / focalX;
    double normY = (imgY - centerY) / focalY;
    
    // 计算相机坐标系中的3D坐标
    double camX = normX * objectDistance;
    double camY = normY * objectDistance;
    double camZ = objectDistance;
    
    // 转换到base_footprint坐标系
    geometry_msgs::PointStamped cameraPoint;
    cameraPoint.header.frame_id = "camera_link";
    cameraPoint.header.stamp = ros::Time(0);
    cameraPoint.point.x = camX;
    cameraPoint.point.y = camY;
    cameraPoint.point.z = camZ;
    
    geometry_msgs::PointStamped basePoint;
    try {
        coordinateTransformer->transformPoint("base_footprint", cameraPoint, basePoint);
        worldX = basePoint.point.x;
        worldY = basePoint.point.y;
        ROS_INFO("[坐标转换] 图像坐标(%d,%d) -> 世界坐标(%.3f,%.3f)", imgX, imgY, worldX, worldY);
        return true;
    } catch (tf::TransformException& ex) {
        ROS_ERROR("[坐标转换] TF变换失败: %s", ex.what());
        return false;
    }
}

// 获取当前机器人位置
bool getCurrentRobotPosition(double& posX, double& posY, double& heading) {
    if (!coordinateTransformer) {
        ROS_ERROR("[位置获取] 坐标转换器未初始化");
        return false;
    }
    
    tf::StampedTransform transform;
    try {
        coordinateTransformer->lookupTransform("map", "base_footprint", ros::Time(0), transform);
        posX = transform.getOrigin().x();
        posY = transform.getOrigin().y();
        heading = tf::getYaw(transform.getRotation());
        ROS_INFO("[位置获取] 机器人位置: (%.3f, %.3f), 朝向: %.3f", posX, posY, heading);
        return true;
    } catch (tf::TransformException& ex) {
        ROS_ERROR("[位置获取] 获取机器人位置失败: %s", ex.what());
        return false;
    }
}

// 计算夹爪与物体的对齐误差
bool calculatePositionError(double objWorldX, double objWorldY, double& errorX, double& errorY) {
    double robotX, robotY, robotHeading;
    if (!getCurrentRobotPosition(robotX, robotY, robotHeading)) {
        return false;
    }
    
    // 计算相对位置
    double relX = objWorldX - robotX;
    double relY = objWorldY - robotY;
    
    // 转换到机器人坐标系
    errorX = relX * cos(robotHeading) + relY * sin(robotHeading);
    errorY = -relX * sin(robotHeading) + relY * cos(robotHeading);
    
    ROS_INFO("[位置误差] 相对误差: x=%.3fm, y=%.3fm", errorX, errorY);
    return true;
}

// 存储识别数字的全局变量

// 导航回调函数
void navigationCallback(const std_msgs::String::ConstPtr& msg) {
    if (msg->data != "done") return;
    
    switch (currentPhase) {
        case NAVIGATE_TO_POINT1:
            ROS_INFO("[导航] 到达点1 → 前往点2");
            navigationMessage.data = "2";
            navPointPublisher.publish(navigationMessage);
            currentPhase = NAVIGATE_TO_POINT2;
            break;
        case NAVIGATE_TO_POINT2:
            ROS_INFO("[导航] 到达点2 → 开始新任务流程");
            currentPhase = SHIFT_LEFT_INITIAL;
            break;
        case NAVIGATE_TO_POINT3:
            ROS_INFO("[导航] 到达点3 → 前进到抓取位置");
            currentPhase = MOVE_TO_GRAB_POSITION;
            break;
        case NAVIGATE_TO_POINT4:
            ROS_INFO("[导航] 到达点4 → 前进到放置位置");
            currentPhase = MOVE_TOWARD_TARGET;
            break;
        case NAVIGATE_TO_POINT5:
            ROS_INFO("[导航] 到达点5 → 开始数字识别");
            currentPhase = RECOGNIZE_NUMERALS;
            break;
        case RETURN_TO_START:
            ROS_INFO("[导航] 返回起点 → 任务完成");
            currentPhase = MISSION_COMPLETE;
            break;
        default:
            break;
    }
}

// 主函数
int main(int argc, char** argv) {
    // 设置字符编码确保中文日志显示正确
    setlocale(LC_ALL, "");
    setlocale(LC_CTYPE, "zh_CN.UTF-8");
    
    init(argc, argv, "navigation_node");
    NodeHandle nodeHandler;

    // 设置导航速度参数
    nodeHandler.setParam("/move_base/DWAPlannerROS/max_vel_x", 0.8);
    nodeHandler.setParam("/move_base/DWAPlannerROS/max_vel_trans", 0.8);
    nodeHandler.setParam("/move_base/WpbhLocalPlanner/max_vel_trans", 0.8);
    nodeHandler.setParam("/move_base/TebLocalPlannerROS/max_vel_x", 0.8);
    
    nodeHandler.setParam("/move_base/DWAPlannerROS/max_vel_theta", 1.5);
    nodeHandler.setParam("/move_base/TebLocalPlannerROS/max_vel_theta", 1.5);
    
    nodeHandler.setParam("/move_base/DWAPlannerROS/acc_lim_x", 2.5);
    nodeHandler.setParam("/move_base/DWAPlannerROS/acc_lim_theta", 3.0);
    nodeHandler.setParam("/move_base/TebLocalPlannerROS/acc_lim_x", 2.5);
    nodeHandler.setParam("/move_base/TebLocalPlannerROS/acc_lim_theta", 3.0);
    
    nodeHandler.setParam("/move_base/DWAPlannerROS/xy_goal_tolerance", 0.15);
    nodeHandler.setParam("/move_base/DWAPlannerROS/yaw_goal_tolerance", 0.2);
    nodeHandler.setParam("/move_base/TebLocalPlannerROS/xy_goal_tolerance", 0.15);
    nodeHandler.setParam("/move_base/TebLocalPlannerROS/yaw_goal_tolerance", 0.2);
    
    ROS_INFO("[系统] 导航参数设置完成");

    // 初始化坐标转换器
    coordinateTransformer = new tf::TransformListener();
    ROS_INFO("[系统] 坐标转换器初始化完成");

    navPointPublisher = nodeHandler.advertise<std_msgs::String>("/waterplus/navi_waypoint", 10);
    armController = nodeHandler.advertise<sensor_msgs::JointState>("/wpb_mani/joint_ctrl", 10);
    gripperController = nodeHandler.advertise<std_msgs::Float64>("/wpb_mani/gripper_position_controller/command", 10);
    motionPublisher = nodeHandler.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    imageSubscriber = nodeHandler.subscribe("/depth/image_color", 1, imageDataCallback);
    navigationSubscriber = nodeHandler.subscribe("/waterplus/navi_result", 10, navigationCallback);
    
    // 机械臂抓取相关发布器和订阅器
    surfaceHeightPublisher = nodeHandler.advertise<std_msgs::Float64>("/wpb_mani/plane_height", 10);
    grabTargetPublisher = nodeHandler.advertise<geometry_msgs::Pose>("/wpb_mani/grab_box", 10);
    objectScanPublisher = nodeHandler.advertise<std_msgs::String>("/wpb_mani/box_detect", 10);
    ros::Subscriber boxCoordinateSub = nodeHandler.subscribe("/wpb_mani/boxes_3d", 10, BoxCoordinateReceived);
    ros::Subscriber resultSub = nodeHandler.subscribe("/wpb_mani/grab_result", 10, GrabResultArrived);
    
    // 频率控制机制
    ros::Subscriber armCmdSub = nodeHandler.subscribe("/wpb_mani/cmd_vel", 10, ArmVelocityCallback);
    
    // 点云处理相关发布器和订阅器
    cloudDataSubscriber = nodeHandler.subscribe("/kinect2/sd/points", 1, HandlePointCloudData);
    objectsLocationPublisher = nodeHandler.advertise<wpb_mani_behaviors::Coord>("/wpb_mani/boxes_3d", 10);
    visualMarkersPublisher = nodeHandler.advertise<visualization_msgs::MarkerArray>("/wpb_mani/boxes_markers", 10);
    
    // 初始化点云
    rawCloudData.reset(new pcl::PointCloud<pcl::PointXYZ>);
    processedCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    
    ROS_INFO("[集成功能] 点云处理初始化完成");

    ROS_INFO("[系统] 初始化完成");

    ROS_INFO("[初始化] 机械臂归位，夹爪打开");
    sendArmCommand(ARM_HOME_POSITION);
    setGripperState(GRIPPER_OPEN_STATE);
    

    // 按点1-2-3-4-5-1顺序执行任务
    ROS_INFO("[系统] 任务模式启动");
    
    navigationMessage.data = "1";
    navPointPublisher.publish(navigationMessage);
    currentPhase = NAVIGATE_TO_POINT1;

    Rate controlRate(10);
    while (ok() && currentPhase != MISSION_COMPLETE) {
        switch (currentPhase) {
            case SHIFT_LEFT_INITIAL: {
                ROS_INFO("[任务] 点2: 首先左移1秒");
                
                performLateralMovement("left", 1.0, 0.6);
                
                ROS_INFO("[左移] 左移完成，开始红盒暗区检测");
                currentPhase = CHECK_RED_BOX_DARKNESS;
                break;
            }

            case CHECK_RED_BOX_DARKNESS: {
                ROS_INFO("[任务] 点2: 检测红盒暗区比例");
                
                if (newImageAvailable) {
                    double darkRatio;
                    
                    if (analyzeRedBoxDarkRegion(currentFrame, darkRatio)) {
                        ROS_INFO("[红盒暗区] 检测成功，暗区比例: %.2f", darkRatio);
                        
                        if (darkRatio < 0.15) {
                            ROS_INFO("[红盒暗区] 暗区比例<15%，向右移动");
                            currentPhase = SHIFT_RIGHT_POST_DETECT;
                        } else {
                            ROS_INFO("[红盒暗区] 暗区比例>=15%，前进接触");
                            currentPhase = ADVANCE_TO_TOUCH;
                        }
                    } else {
                        ROS_WARN("[红盒暗区] 检测失败，重试中...");
                    }
                } else {
                    ROS_WARN("[红盒暗区] 等待图像数据...");
                }
                
                break;
            }

            case SHIFT_RIGHT_POST_DETECT: {
                ROS_INFO("[任务] 点2: 右移2秒");
                
                performLateralMovement("right", 2.0, 0.6);
                
                ROS_INFO("[右移] 右移完成，再次检测暗区比例");
                currentPhase = VERIFY_RED_BOX_AFTER_SHIFT;
                break;
            }

            case VERIFY_RED_BOX_AFTER_SHIFT: {
                ROS_INFO("[任务] 点2: 右移后再次检测暗区比例");
                
                if (newImageAvailable) {
                    double darkRatioAfter;
                    
                    if (analyzeRedBoxDarkRegion(currentFrame, darkRatioAfter)) {
                        ROS_INFO("[右移后红盒暗区] 检测成功，暗区比例: %.2f", darkRatioAfter);
                        
                        if (darkRatioAfter < 0.15) {
                            ROS_INFO("[右移后红盒暗区] 暗区比例<15%，前往点3");
                            navigationMessage.data = "3";
                            navPointPublisher.publish(navigationMessage);
                            currentPhase = NAVIGATE_TO_POINT3;
                        } else {
                            ROS_INFO("[右移后红盒暗区] 暗区比例>=15%，前进接触");
                            currentPhase = ADVANCE_TO_TOUCH;
                        }
                    } else {
                        ROS_WARN("[右移后红盒暗区] 检测失败，前往点3");
                        navigationMessage.data = "3";
                        navPointPublisher.publish(navigationMessage);
                        currentPhase = NAVIGATE_TO_POINT3;
                    }
                } else {
                    ROS_WARN("[右移后红盒暗区] 等待图像数据，前往点3");
                    navigationMessage.data = "3";
                    navPointPublisher.publish(navigationMessage);
                    currentPhase = NAVIGATE_TO_POINT3;
                }
                
                break;
            }

            case ADVANCE_TO_TOUCH: {
                ROS_INFO("[任务] 点2: 前进1秒");
                
                moveForward(1.0, 0.75);
                ROS_INFO("[前进] 前进到目标前方");
                
                ROS_INFO("[接触] 开始接触动作");
                sendArmCommand(ARM_TOUCH_POSITION);
                ros::Duration(1.5).sleep();
                sendArmCommand(ARM_HOME_POSITION);
                
                ROS_INFO("[导航] 点2接触完成，前往点3");
                navigationMessage.data = "3";
                navPointPublisher.publish(navigationMessage);
                currentPhase = NAVIGATE_TO_POINT3;
                break;
            }

            case MOVE_TO_GRAB_POSITION: {
                ROS_INFO("[任务] 点3前进1秒");
                moveForward(1.0, 0.28);
                ROS_INFO("[前进] 前进到抓取位置");
                
                currentPhase = INITIATE_GRAB_PROCESS;
                break;
            }

            case INITIATE_GRAB_PROCESS: {
                ROS_INFO("[抓取] 开始抓取物体（第%d个物体，第%d次尝试）", collectedCount + 1, grabAttemptCounter + 1);
                
                ROS_INFO("[抓取] 前进完成，启动机械臂模块");
                
                grabOperationSuccess = false;
                grabInProgressFlag = false;
                coordinatesTransmitted = false;
                
                armSystemActive = true;
                
                std_msgs::Float64 surfaceHeightMsg;
                surfaceHeightMsg.data = 0.22;
                surfaceHeightPublisher.publish(surfaceHeightMsg);
                ROS_INFO("[抓取] 发布表面高度: %.2f米", surfaceHeightMsg.data);
                
                ros::Duration(0.5).sleep();
                
                ros::Time grabStart = ros::Time::now();
                while (ros::ok() && !grabOperationSuccess) {
                    ros::Duration(0.033).sleep();
                    spinOnce();
                    
                    if (resultNotificationReceived) {
                        if (grabOperationSuccess) {
                            ROS_INFO("[抓取] 收到抓取结果，抓取成功");
                            break;
                        } else {
                            ROS_INFO("[抓取] 收到抓取结果，抓取失败，继续监控");
                            resultNotificationReceived = false;
                        }
                    }
                }
                
                if (grabOperationSuccess) {
                    ROS_INFO("[抓取] 物体抓取成功!");
                    
                    armSystemActive = false;
                    
                    sendArmCommand(ARM_HOME_POSITION);
                    ros::Duration(0.8).sleep();
                    
                    ROS_INFO("[抓取] 抓取成功，前往点4放置物体");
                    navigationMessage.data = "4";
                    navPointPublisher.publish(navigationMessage);
                    currentPhase = NAVIGATE_TO_POINT4;
                    
                    grabOperationSuccess = false;
                    grabInProgressFlag = false;
                    coordinatesTransmitted = false;
                    resultNotificationReceived = false;
                    
                    armSystemActive = false;
                    
                    ROS_INFO("[系统重置] 抓取成功，系统状态重置完成");
                } else {
                    ROS_WARN("[抓取] 抓取失败，重试");
                    grabAttemptCounter++;
                    
                    grabOperationSuccess = false;
                    grabInProgressFlag = false;
                    coordinatesTransmitted = false;
                    resultNotificationReceived = false;
                    
                    armSystemActive = false;
                    
                    sendArmCommand(ARM_HOME_POSITION);
                    
                    ROS_INFO("[系统重置] 抓取失败，系统状态重置完成");
                    
                    if (grabAttemptCounter >= 3) {
                        ROS_WARN("[抓取] 抓取尝试次数超过限制，返回物体检测阶段");
                        currentPhase = SCAN_FOR_OBJECTS;
                        grabAttemptCounter = 0;
                    } else {
                        currentPhase = INITIATE_GRAB_PROCESS;
                    }
                }

                break;
            }

            case IDENTIFY_COLOR_OBJECTS: {
                ROS_INFO("[状态] 识别颜色物体");
                
                ROS_INFO("[颜色匹配] 当前夹爪中物体颜色: %s，直接进行放置操作", objectColor.c_str());
                
                currentPhase = APPROACH_COLOR_TARGET;
                ROS_INFO("[状态转换] 识别颜色物体 -> 接近颜色目标");
                break;
            }

            case APPROACH_COLOR_TARGET: {
                ROS_INFO("[侧移] 跳过侧移，直接前进到颜色目标前方");
                currentPhase = MOVE_TOWARD_TARGET;
                break;
            }

            case MOVE_TOWARD_TARGET: {
                ROS_INFO("[前进] 前进到%s颜色目标前方", matchedColor.c_str());
                
                moveForward(1.0, 0.4);
                
                ROS_INFO("[前进] 到达%s颜色目标前方，准备放置物体", matchedColor.c_str());
                currentPhase = EXECUTE_PLACEMENT;
                break;
            }

            case EXECUTE_PLACEMENT: {
                ROS_INFO("[放置过程] 开始放置物体（第%d个物体）", collectedCount);
                
                ROS_INFO("[放置过程] 伸展机械臂到放置位置");
                sendArmCommand(ARM_PLACE_POSITION);
                ros::Duration(0.2).sleep();
                
                ROS_INFO("[放置过程] 释放夹爪放置物体");
                setGripperState(GRIPPER_OPEN_STATE);
                ros::Duration(0.5).sleep();
                
                ROS_INFO("[放置过程] 收回机械臂");
                sendArmCommand(ARM_HOME_POSITION);
                ros::Duration(0.3).sleep();
                
                ROS_INFO("[放置过程] 物体放置完成");
                
                collectedCount++;
                
                if (collectedCount >= 3) {
                    ROS_INFO("[任务] 所有3个物体已放置，前往点5");
                    navigationMessage.data = "5";
                    navPointPublisher.publish(navigationMessage);
                    currentPhase = NAVIGATE_TO_POINT5;
                } else {
                    ROS_INFO("[任务] 还剩%d个物体要抓取，返回点3继续抓取", 3 - collectedCount);
                    navigationMessage.data = "3";
                    navPointPublisher.publish(navigationMessage);
                    currentPhase = NAVIGATE_TO_POINT3;
                }
                break;
            }

            case RECOGNIZE_NUMERALS: {
                ROS_INFO("[数字识别] 开始检测红盒中的数字");
                
                if (!newImageAvailable) {
                    ROS_WARN("[数字识别] 等待图像数据...");
                    break;
                }
                
                int detectedNum;
                if (identifyNumberInRedBox(currentFrame, detectedNum)) {
                    ROS_INFO("[数字识别] 检测到数字: %d", detectedNum);
                    detectedNumber = detectedNum;
                    newImageAvailable = false;
                    currentPhase = PERFORM_ROTATION;
                } else {
                    ROS_WARN("[数字识别] 未检测到有效数字，继续检测");
                }
                break;
            }

            case PERFORM_ROTATION: {
                ROS_INFO("[旋转任务] 开始原地旋转，旋转次数: %d", detectedNumber);
                
                for (int i = 0; i < detectedNumber; i++) {
                    ROS_INFO("[旋转任务] 第%d/%d次旋转", i + 1, detectedNumber);
                    
                    geometry_msgs::Twist rotationCommand;
                    rotationCommand.angular.z = 1.0;
                    motionPublisher.publish(rotationCommand);
                    ros::Duration(6.28).sleep();
                    
                    rotationCommand.angular.z = 0;
                    motionPublisher.publish(rotationCommand);
                    ros::Duration(0.5).sleep();
                }
                
                ROS_INFO("[旋转任务] 旋转完成，检查任务是否完成");
                
                if (collectedCount >= 3) {
                    ROS_INFO("[任务完成] 完成%d个物体，任务完成，返回起点", collectedCount);
                    navigationMessage.data = "1";
                    navPointPublisher.publish(navigationMessage);
                    currentPhase = RETURN_TO_START;
                } else {
                    ROS_INFO("[任务继续] 完成%d个物体，继续任务，返回起点", collectedCount);
                    navigationMessage.data = "1";
                    navPointPublisher.publish(navigationMessage);
                    currentPhase = NAVIGATE_TO_POINT1;
                }
                
                numberDetectionDone = false;
                detectedNumber = 0;
                break;
            }

            case DETECT_SPECIAL_MARKERS:
                if (markerDetectionDone) {
                    currentPhase = NAVIGATE_TO_POINT3;
                    markerDetectionDone = false;
                }
                break;

            default:
                break;
        }
        
        spinOnce();
        controlRate.sleep();
    }

    return 0;
}

// 物体坐标回调函数
void BoxCoordinateReceived(const wpb_mani_behaviors::Coord::ConstPtr& msg) {
    if (msg->x.size() > 0) {
        primaryObjectX = msg->x[0];
        primaryObjectY = msg->y[0];
        primaryObjectZ = msg->z[0];
        foundObjectsCount = msg->x.size();
        receivedObjectCoordinates = true;
        
        if (armSystemActive) {
            grabTargetMessage.position.x = msg->x[0]; 
            grabTargetMessage.position.y = msg->y[0]; 
            grabTargetMessage.position.z = msg->z[0]; 
            grabTargetPublisher.publish(grabTargetMessage); 
            ROS_INFO("[物体坐标] 发布抓取命令到机械臂模块");
        }
        
        ROS_INFO("[物体坐标] 收到%d个物体坐标，第一个物体位置: x=%.3f, y=%.3f, z=%.3f", 
                 foundObjectsCount, primaryObjectX, primaryObjectY, primaryObjectZ);
    } else {
        receivedObjectCoordinates = false;
        ROS_WARN("[物体坐标] 未检测到物体");
    }
}

// 抓取结果回调函数
void GrabResultArrived(const std_msgs::String::ConstPtr& msg) {
    ROS_INFO("[抓取结果] 收到抓取结果: %s", msg->data.c_str());
    
    if (msg->data == "success" || msg->data == "done") {
        grabOperationSuccess = true;
        resultNotificationReceived = true;
        ROS_INFO("[抓取结果] 抓取成功!");
    } else if (msg->data == "failed") {
        grabOperationSuccess = false;
        resultNotificationReceived = true;
        ROS_WARN("[抓取结果] 抓取失败!");
    } else {
        ROS_WARN("[抓取结果] 未知抓取结果: %s", msg->data.c_str());
    }
}

// 机械臂速度回调函数
void ArmVelocityCallback(const geometry_msgs::Twist::ConstPtr& msg) {
    if (armSystemActive) {
        motionPublisher.publish(*msg);
        ROS_INFO("[机械臂速度] 转发速度命令: vx=%.2f, vy=%.2f, wz=%.2f", 
                 msg->linear.x, msg->linear.y, msg->angular.z);
    }
}

// 设置夹爪状态函数
void setGripperState(double pos) {
    ROS_INFO("[夹爪控制] 设置夹爪位置: %f", pos);
    
    std_msgs::Float64 gripperMsg;
    gripperMsg.data = pos;
    
    for (int i = 0; i < 3; i++) {
        gripperController.publish(gripperMsg);
        ros::Duration(0.1).sleep();
    }
    
    ROS_INFO("[夹爪控制] 夹爪控制命令已发送，位置: %f", pos);
}

// 点云处理回调函数
void HandlePointCloudData(const sensor_msgs::PointCloud2::ConstPtr& msg) {
    pcl::fromROSMsg(*msg, *rawCloudData);
    cloudDataReceived = true;
    
    FindObjectsInCloud();
}

// 物体检测函数
void FindObjectsInCloud() {
    if (!cloudDataReceived || rawCloudData->empty()) {
        ROS_WARN("[物体检测] 点云数据为空");
        return;
    }
    
    ROS_INFO("[物体检测] 开始点云处理，原始点云大小: %zu", rawCloudData->size());
    
    // 点云滤波
    pcl::VoxelGrid<pcl::PointXYZ> voxelFilter;
    voxelFilter.setInputCloud(rawCloudData);
    voxelFilter.setLeafSize(0.01f, 0.01f, 0.01f);
    voxelFilter.filter(*processedCloud);
    
    ROS_INFO("[物体检测] 体素滤波后点云大小: %zu", processedCloud->size());
    
    // 平面检测
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    pcl::SACSegmentation<pcl::PointXYZ> segmenter;
    
    segmenter.setOptimizeCoefficients(true);
    segmenter.setModelType(pcl::SACMODEL_PLANE);
    segmenter.setMethodType(pcl::SAC_RANSAC);
    segmenter.setDistanceThreshold(0.01);
    segmenter.setInputCloud(processedCloud);
    segmenter.segment(*inliers, *coefficients);
    
    if (inliers->indices.empty()) {
        ROS_WARN("[物体检测] 未检测到平面");
        return;
    }
    
    detectedSurfaceHeight = coefficients->values[3];
    ROS_INFO("[物体检测] 检测到平面，高度: %.3f", detectedSurfaceHeight);
    
    // 提取平面上的物体
    pcl::ExtractIndices<pcl::PointXYZ> extractor;
    extractor.setInputCloud(processedCloud);
    extractor.setIndices(inliers);
    extractor.setNegative(true);
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr objectsCloud(new pcl::PointCloud<pcl::PointXYZ>);
    extractor.filter(*objectsCloud);
    
    ROS_INFO("[物体检测] 平面上物体点云大小: %zu", objectsCloud->size());
    
    // 欧几里得聚类
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(objectsCloud);
    
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> clusterExtractor;
    clusterExtractor.setClusterTolerance(0.02);
    clusterExtractor.setMinClusterSize(100);
    clusterExtractor.setMaxClusterSize(25000);
    clusterExtractor.setSearchMethod(tree);
    clusterExtractor.setInputCloud(objectsCloud);
    clusterExtractor.extract(clusterIndices);
    
    ROS_INFO("[物体检测] 检测到%zu个聚类", clusterIndices.size());
    
    // 提取物体坐标
    locatedObjects.clear();
    for (size_t i = 0; i < clusterIndices.size(); i++) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr clusterCloud(new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto& idx : clusterIndices[i].indices) {
            clusterCloud->points.push_back(objectsCloud->points[idx]);
        }
        
        Eigen::Vector4f centroid;
        pcl::compute3DCentroid(*clusterCloud, centroid);
        
        pcl::PointXYZ minPoint, maxPoint;
        pcl::getMinMax3D(*clusterCloud, minPoint, maxPoint);
        
        float width = maxPoint.x - minPoint.x;
        float depth = maxPoint.y - minPoint.y;
        float height = maxPoint.z - minPoint.z;
        
        if (width > 0.03 && depth > 0.03 && height > 0.03) {
            vector<float> objectInfo = {centroid[0], centroid[1], centroid[2], width, depth, height};
            locatedObjects.push_back(objectInfo);
            
            ROS_INFO("[物体检测] 物体%zu: 位置(%.3f,%.3f,%.3f), 尺寸(%.3f,%.3f,%.3f)", 
                     i+1, centroid[0], centroid[1], centroid[2], width, depth, height);
        }
    }
    
    // 发布物体坐标
    if (!locatedObjects.empty()) {
        wpb_mani_behaviors::Coord objectsMsg;
        for (size_t i = 0; i < locatedObjects.size(); i++) {
            objectsMsg.name.push_back("object_" + to_string(i));
            objectsMsg.x.push_back(locatedObjects[i][0]);
            objectsMsg.y.push_back(locatedObjects[i][1]);
            objectsMsg.z.push_back(locatedObjects[i][2]);
            objectsMsg.probability.push_back(1.0);
        }
        
        objectsLocationPublisher.publish(objectsMsg);
        receivedObjectCoordinates = true;
        foundObjectsCount = locatedObjects.size();
        
        if (foundObjectsCount > 0) {
            primaryObjectX = locatedObjects[0][0];
            primaryObjectY = locatedObjects[0][1];
            primaryObjectZ = locatedObjects[0][2];
        }
        
        ROS_INFO("[物体检测] 发布%zu个物体坐标", locatedObjects.size());
    }
}

// 瞄准目标
void AlignWithTarget() {
    ROS_INFO("[抓取控制] 瞄准物体");
    
    double angle = atan2(targetGrabPositionY, targetGrabPositionX);
    
    SendMotionCommand(0.0, 0.0, angle);
    ros::Duration(1.0).sleep();
    
    currentGrabPhase = PHASE_EXTENSION;
}

// 伸出手臂
void ExtendArmToTarget() {
    ROS_INFO("[抓取控制] 伸出手臂");
    
    sendArmCommand(ARM_GRAB_READY);
    ros::Duration(1.0).sleep();
    
    currentGrabPhase = PHASE_APPROACH;
}

// 抓取物体
void PerformGrabAction() {
    ROS_INFO("[抓取控制] 抓取物体");
    
    AdjustGripper(GRIPPER_CLOSE_STATE);
    ros::Duration(1.0).sleep();
    
    if (grabOperationSuccess) {
        ROS_INFO("[抓取控制] 抓取成功");
        currentGrabPhase = PHASE_RETRACTION;
    } else {
        ROS_WARN("[抓取控制] 抓取失败，重试");
        grabRetryCounter++;
        if (grabRetryCounter >= MAX_GRAB_ATTEMPTS) {
            ROS_ERROR("[抓取控制] 抓取失败次数过多，中止");
            grabSequenceActive = false;
        }
    }
}

// 收回
void RetractArmToHome() {
    ROS_INFO("[抓取控制] 收回");
    
    sendArmCommand(ARM_HOME_POSITION);
    ros::Duration(1.0).sleep();
    
    sendArmCommand(ARM_GRAB_READY);
    ros::Duration(1.0).sleep();
    
    currentGrabPhase = PHASE_ALIGNMENT;
    grabRetryCounter = 0;
    
    ROS_INFO("[抓取控制] 抓取过程完成");
}

// 机械臂夹爪控制
void AdjustGripper(double pos) {
    ROS_INFO("[夹爪调整] 设置夹爪位置: %.3f", pos);
    setGripperState(pos);
}

// 速度控制
void SendMotionCommand(double vx, double vy, double wz) {
    geometry_msgs::Twist motionCmd;
    motionCmd.linear.x = vx;
    motionCmd.linear.y = vy;
    motionCmd.angular.z = wz;
    motionPublisher.publish(motionCmd);
    
    ROS_INFO("[运动控制] 发布速度命令: vx=%.2f, vy=%.2f, wz=%.2f", vx, vy, wz);
}

// 等待新图像函数
bool waitForFreshImage() {
    ros::Time startTime = ros::Time::now();
    while (ros::ok()) {
        if (newImageAvailable) {
            return true;
        }
        
        ros::Duration(0.05).sleep();
        spinOnce();
        
        ros::Duration elapsed = ros::Time::now() - startTime;
        if (elapsed.toSec() > 5.0) {
            ROS_WARN("[图像等待] 等待新图像超时（5秒）");
            return false;
        }
    }
    return false;
}

// 验证物体抓取函数
bool confirmBoxPickup(cv::Mat& img) {
    if (img.empty()) {
        ROS_WARN("[抓取验证] 图像为空，无法验证抓取结果");
        return false;
    }
    
    cv::Mat hsvImage;
    cv::cvtColor(img, hsvImage, cv::COLOR_BGR2HSV);
    
    cv::Scalar lower, upper;
    
    if (objectColor == "red") {
        lower = cv::Scalar(0, 50, 50);
        upper = cv::Scalar(10, 255, 255);
    } else if (objectColor == "green") {
        lower = cv::Scalar(35, 50, 50);
        upper = cv::Scalar(85, 255, 255);
    } else if (objectColor == "blue") {
        lower = cv::Scalar(100, 50, 50);
        upper = cv::Scalar(130, 255, 255);
    } else {
        lower = cv::Scalar(0, 50, 50);
        upper = cv::Scalar(10, 255, 255);
    }
    
    cv::Mat colorMask;
    cv::inRange(hsvImage, lower, upper, colorMask);
    
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(colorMask, colorMask, cv::MORPH_CLOSE, kernel);
    
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(colorMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) {
        ROS_INFO("[抓取验证] 未检测到%s物体轮廓，抓取验证失败", objectColor.c_str());
        return false;
    }
    
    double maxArea = 0;
    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > maxArea) {
            maxArea = area;
        }
    }
    
    if (maxArea > 500) {
        ROS_INFO("[抓取验证] 检测到%s物体（面积%.2f），抓取验证失败", objectColor.c_str(), maxArea);
    } else {
        ROS_INFO("[抓取验证] 未检测到%s物体（最大面积%.2f），抓取验证成功", objectColor.c_str(), maxArea);
        return true;
    }
}
