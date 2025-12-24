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
#include <climits>

using namespace cv;
using namespace std;
using namespace ros;

// 新增：函数声明
void RobotPoseCB(const geometry_msgs::Pose2D::ConstPtr &msg);
void BoxAndRobotCoordSyncCB(const wpb_mani_behaviors::Coord::ConstPtr &msg);

// 任务状态机
enum TaskStep {
    INIT,
    GOTO_WP1,
    GOTO_WP2,
    DETECT_QD,
    MOVE_TO_QD_SIDE,
    MOVE_TO_QD_FRONT,
    KNOCK_QD,
    GOTO_WP3,
    DETECT_BOX,
    MOVE_TO_TABLE_FRONT,
    GRAB_BOX,
    GOTO_WP4,
    DETECT_COLOR_BOX,  // 新增：检测颜色方框
    MOVE_TO_COLOR_BOX, // 新增：横向移动到对应颜色方框前
    MOVE_FORWARD_TO_BOX, // 新增：前进到方框前
    PLACE_BOX,
    GOTO_WP5,
    DETECT_NUM,
    ROTATE,
    GOTO_WP1_FINAL,
    DETECT_XN,
    DONE
};

// 全局变量
TaskStep current_step = INIT;
Publisher waypoint_pub, mani_pub, gripper_pub, cmd_vel_pub;
Subscriber img_sub, navi_sub;
string box_color;
string selected_color;  // 用于存储匹配的方框颜色
bool qd_ok = false, box_ok = false, num_ok = false, xn_ok = false;
int box_count = 0;
const int MAX_BOX = 3;
std_msgs::String nav_msg;
string qd_side = "";



// 新增：盒子检测相关全局变量
bool box_coord_received = false; // 标记是否接收到盒子坐标
int box_num = 0; // 检测到的盒子数量
float box_track_x = 0.0; // 存储第一个盒子的x坐标
float box_track_y = 0.0; // 存储第一个盒子的y坐标
float box_track_z = 0.0; // 存储第一个盒子的z坐标

// 新增：盒子信息结构体
struct BoxInfo {
    string color;
    int pixel_offset;
    int center_x;
    int x;      // 中心x坐标
    int y;      // 中心y坐标
    int width;  // 宽度
    int height; // 高度
    double area; // 面积
};

// 新增函数声明
vector<string> detect_all_color_boxes(const Mat& img);
string select_best_match_box(const string& current_box_color, const vector<string>& detected_colors, const Mat& img);
bool wait_for_new_image();
bool verify_box_capture(const Mat& img);
bool has_valid_shape(const vector<vector<Point>>& contours);
void send_mani_with_gripper(const vector<double>& angles, double gripper_angle);
void reset_y_monitoring(); // 重置y值监控系统
bool detect_gripper_by_image_features(const Mat& img, Rect& gripper_roi); // 机械爪图像特征检测

// 新增：可视化函数声明
void visualize_detection_regions(const Mat& img, const Rect& gripper_roi, const string& detection_method);
void visualize_gripper_color_detection(const Mat& img, const Rect& gripper_roi, const string& target_color);
void visualize_smart_color_detection(const Mat& img, const Rect& gripper_roi, const string& detected_color, 
                                   bool red_touches_edge, bool yellow_touches_edge, bool blue_touches_edge);

// 新增：新的盒子检测和选择函数
vector<BoxInfo> detect_all_color_boxes_with_offset(const Mat& img);
BoxInfo select_box_by_pixel_offset(const vector<BoxInfo>& detected_boxes);

// 新增：航点3颜色像素块检测函数
vector<BoxInfo> detect_color_pixels_on_desk(const Mat& img);
string select_direction_by_horizontal_offset(const vector<BoxInfo>& detected_boxes);

// 新增：辅助函数声明
void publish_zero_velocity_for_duration(double duration);
void publish_zero_velocity();
void extend_manipulator();

// 新增：颜色分类函数声明
string classify_color(double red, double green, double blue);

// 新增：缺失的函数声明
int calculate_box_pixel_offset(const Mat& img, const string& color);
string select_direction_by_color_regions(const vector<BoxInfo>& color_regions);
void execute_initial_positioning(const string& direction);

// 新增：频率降低机制函数声明
void wpbManiCmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg);
void publishReducedFrequencyCmd();

// wpb_mani抓取相关
static ros::Publisher plane_height_pub;
static ros::Publisher grab_box_pub;
static geometry_msgs::Pose grab_box_msg;
bool grab_in_progress = false;
bool grab_success = false;
int grab_attempt_count = 0; // 抓取尝试次数
bool grab_coord_sent = false; // 是否已经发送过抓取坐标
bool grab_result_received = false; // 新增：是否接收到抓取结果回调

// 新增：盒子和小车坐标同步输出相关
static ros::Subscriber box_and_robot_sub; // 新增：盒子和小车坐标同步订阅器
static ros::Subscriber robot_pose_sub;    // 新增：小车位置订阅器
static double robot_x = 0.0;               // 新增：小车x坐标
static double robot_y = 0.0;               // 新增：小车y坐标
static double robot_theta = 0.0;          // 新增：小车航向角

// y值监控相关变量
double last_y_coord = 0.0; // 上一次的y坐标值
int y_large_change_count = 0; // 连续y值差大于0.1的计数
int y_small_change_count = 0; // 连续y值差小于0.01的计数
bool y_small_change_triggered = false; // 连续5次y值小变化是否已经触发（每次抓取只触发一次）
bool wpb_mani_active = false; // wpb_mani模块是否活跃

// TF坐标转换相关
tf::TransformListener* tf_listener = nullptr;

// 新增：图像缓存和验证相关
Mat current_image;
bool has_new_image = false;
ros::Time last_image_time;

// 新增：当前y坐标值
double current_y = 0.0;

// 角色模板颜色特征
Scalar enemy_target_color_avg;
Scalar friendly_target_color_avg;
bool has_enemy_target_template = false;
bool has_friendly_target_template = false;

// 计时器相关全局变量
ros::Time start_time;

// 盒子抓取检测相关全局变量
int rotation_angle = 0;                // 当前旋转角度（正数右转，负数左转）
bool grab_detection_enabled = false;   // 抓取检测是否启用
bool y_monitoring_enabled = true;      // y值监控是否启用
bool y_monitor_first_detection = true; // y坐标监控：是否是第一次检测
bool y_coord_exceed_threshold = false; // y坐标监控：是否超过阈值

// 新增：智能y值监控系统
float y_prev_value = 0.0;              // 前一个y坐标值
bool y_monitoring_active = false;      // y值监控是否激活
bool y_large_change_detected = false;   // 是否检测到大变化
bool y_small_change_detected = false;   // 是否检测到小变化

// 新增：频率降低机制
geometry_msgs::Twist last_wpb_mani_cmd; // 存储来自wpb_mani模块的最新指令
bool has_new_wpb_mani_cmd = false;      // 是否有新的wpb_mani指令
ros::Time last_cmd_time;                // 上次发布指令的时间
const double TARGET_FREQUENCY = 5.0;    // 目标频率5Hz
const double TARGET_INTERVAL = 1.0 / TARGET_FREQUENCY; // 目标间隔0.2秒

// 新增：改进的y值监控系统
int y_large_change_counter = 0;        // 累积两次>0.1的计数器
int y_small_change_counter = 0;        // 连续8次<0.01的计数器

// 新增：异步机械臂控制类（参考cruise_node实现）
class AsyncManipulator {
private:
    bool action_completed_;
    ros::Time action_start_time_;
    vector<double> target_angles_;
    
public:
    AsyncManipulator() : action_completed_(true) {}
    
    void send_mani_async(const vector<double>& angles) {
        target_angles_ = angles;
        action_completed_ = false;
        action_start_time_ = ros::Time::now();
        
        // 发送机械臂控制命令
        sensor_msgs::JointState msg;
        msg.name = {"joint1", "joint2", "joint3", "joint4"};
        msg.position = angles;
        msg.velocity = {10, 10, 10, 10};
        ROS_INFO("[异步机械臂] 发送姿态：joint1=%.2f, joint2=%.2f, joint3=%.2f, joint4=%.2f",
                 angles[0], angles[1], angles[2], angles[3]);
        
        for (int i = 0; i < 3; i++) {
            mani_pub.publish(msg);
            Duration(0.1).sleep();
        }
        
        ROS_INFO("[异步机械臂] 动作已发送，开始异步等待");
    }
    
    bool wait_for_completion(double timeout = 1.5) {
        if (action_completed_) {
            return true;
        }
        
        ros::Duration elapsed = ros::Time::now() - action_start_time_;
        
        if (elapsed.toSec() > timeout) {
            ROS_INFO("[异步机械臂] 动作超时，用时%.2f秒", elapsed.toSec());
            action_completed_ = true;
            return true;  // 超时也认为完成
        }
        
        // 优化：基于经验时间判断动作完成
        // 机械臂动作通常在0.8-1.2秒内完成，不再固定等待2.0秒
        if (elapsed.toSec() > 0.8) {
            action_completed_ = true;
            ROS_INFO("[异步机械臂] 动作完成，用时%.2f秒", elapsed.toSec());
            return true;
        }
        
        return false;
    }
    
    bool is_action_completed() {
        return action_completed_;
    }
};
int y_fine_tune_counter = 0;           // 连续12次<0.002的计数器
bool y_large_change_triggered = false;  // 累积两次>0.1是否已触发
bool y_fine_tune_triggered = false;     // 连续12次<0.002是否已触发
bool in_fine_tune_phase = false;        // 是否在微调阶段
bool wpb_mani_enabled = true;          // wpb_mani模块是否启用

// 新增：数字识别边框显示控制
bool show_crop_borders = false;        // 是否显示裁剪边框
int frame_counter = 0;                  // 帧计数器
const int MAX_FRAMES = 150;            // 最大显示帧数（约5秒，30fps）

// 新增：颜色方框检测显示控制
bool show_color_box_detection = false;  // 是否显示颜色方框检测区域
int color_box_frame_counter = 0;        // 颜色方框检测帧计数器
const int COLOR_BOX_MAX_FRAMES = 150;   // 颜色方框检测最大显示帧数（约10秒，30fps）

// 新增：航点3颜色像素块检测显示控制
bool show_color_pixels_detection = false;  // 是否显示颜色像素块检测区域
int color_pixels_frame_counter = 0;        // 颜色像素块检测帧计数器
const int COLOR_PIXELS_MAX_FRAMES = 150;   // 颜色像素块检测最大显示帧数（约5秒，30fps）

// 机械臂预设姿态
const vector<double> MANI_UP = {0, -1.57, 1.35, 0.24};
const vector<double> MANI_KNOCK = {0, 0.8, -0.6, -0.4}; // 优化：向上举高0.4m，向前伸出0.8m以上，推板子上半部分
const vector<double> MANI_GRAB = {0, -0.7, 0.5, -0.3};
const vector<double> MANI_DOWN = {0, -0.7, 0.5, -0.3}; // 新增：机械臂向下伸出姿态
const vector<double> MANI_PLACE = {0, -1.0, 0.8, -0.4}; // 改进：使用更伸展的放置姿态，确保盒子能顺利放下
const double GRIPPER_OPEN = 1.0;  // 修正：使用有效范围内的最大张开角度
const double GRIPPER_CLOSE = 0.035;  // 优化：增大闭合角度到0.035，避免夹得太紧导致盒子飞走

// 横向移动
void move_lateral(const string& direction, double duration_sec = 2.0, double speed = 1.0, bool check_box_detected = false) {
    geometry_msgs::Twist twist;
    twist.linear.x = 0;
    twist.linear.y = (direction == "left") ? speed : -speed;
    twist.angular.z = 0;

    ROS_INFO("[移动] 向 %s 横向移动 %.2f 秒", direction.c_str(), duration_sec);
    ros::Time start = ros::Time::now();
    bool box_detected = false;
    
    while ((ros::Time::now() - start).toSec() < duration_sec && ros::ok()) {
        cmd_vel_pub.publish(twist);
        
        // 如果启用了盒子检测中断，检查是否检测到盒子
        if (check_box_detected) {
            // 检查是否有有效的盒子坐标
            if (box_coord_received && box_num > 0) {
                ROS_INFO("[移动] 检测到盒子，立即停止移动");
                box_detected = true;
                break;
            }
        }
        
        ros::Duration(0.01).sleep();
    }
    
    twist.linear.y = 0;
    cmd_vel_pub.publish(twist);
    ros::Duration(0.3).sleep();
    
    if (box_detected) {
        ROS_INFO("[移动] 横向移动中断，检测到盒子");
    }
}

// 向前移动
void move_forward(double duration_sec = 2.0, double speed = 1.2) {
    geometry_msgs::Twist twist;
    twist.linear.x = speed;
    twist.linear.y = 0;
    twist.angular.z = 0;

    ROS_INFO("[移动] 向前移动 %.2f 秒（约 %.2f 米）", duration_sec, speed * duration_sec);
    ros::Time start = ros::Time::now();
    while ((ros::Time::now() - start).toSec() < duration_sec && ros::ok()) {
        cmd_vel_pub.publish(twist);
        ros::Duration(0.01).sleep();
    }
    twist.linear.x = 0;
    cmd_vel_pub.publish(twist);
    ros::Duration(0.3).sleep();
}

// 新增：后退移动函数
void move_backward(double duration_sec = 1.0, double speed = 1.0) {
    geometry_msgs::Twist twist;
    twist.linear.x = -speed;
    twist.linear.y = 0;
    twist.angular.z = 0;

    ROS_INFO("[移动] 向后移动 %.2f 秒", duration_sec);
    ros::Time start = ros::Time::now();
    while ((ros::Time::now() - start).toSec() < duration_sec && ros::ok()) {
        cmd_vel_pub.publish(twist);
        ros::Duration(0.01).sleep();
    }
    twist.linear.x = 0;
    cmd_vel_pub.publish(twist);
    ros::Duration(0.3).sleep();
}

// 机械臂控制
void send_mani(const vector<double>& angles) {
    sensor_msgs::JointState msg;
    msg.name = {"joint1", "joint2", "joint3", "joint4"};
    msg.position = angles;
    msg.velocity = {10, 10, 10, 10};
    ROS_INFO("[机械臂] 发送姿态：joint1=%.2f, joint2=%.2f, joint3=%.2f, joint4=%.2f",
             angles[0], angles[1], angles[2], angles[3]);
    for (int i = 0; i < 3; i++) {
        mani_pub.publish(msg);
        Duration(0.1).sleep();
    }
    Duration(2.0).sleep();
}

// 夹爪控制函数声明
void set_gripper(double angle);

// 新增：同时控制机械臂和手爪的函数（参考wpb_mani实现）
void send_mani_with_gripper(const vector<double>& angles, double gripper_angle) {
    // 关键修复：参考wpb_mani实现，一次性发送机械臂和夹爪的联合控制命令
    sensor_msgs::JointState msg;
    msg.name = {"joint1", "joint2", "joint3", "joint4", "gripper"};
    
    // 修复：使用1和0控制夹爪开关，而不是0.9
    // gripper_angle = 1 表示张开夹爪，gripper_angle = 0 表示闭合夹爪
    double converted_gripper_angle = gripper_angle;
    if (gripper_angle > 0.5) { // 张开状态
        converted_gripper_angle = 1.0; // 夹爪张开角度改为1.0
    } else { // 闭合状态
        converted_gripper_angle = 0.0; // 夹爪闭合角度改为0.0
    }
    
    msg.position = {angles[0], angles[1], angles[2], angles[3], converted_gripper_angle};
    msg.velocity = {10, 10, 10, 10, 12};
    
    ROS_INFO("[机械臂] 发送联合姿态：joint1=%.2f, joint2=%.2f, joint3=%.2f, joint4=%.2f, gripper=%.2f(转换后:%.2f)",
             angles[0], angles[1], angles[2], angles[3], gripper_angle, converted_gripper_angle);
    
    // 参考wpb_mani实现：发送一次命令即可，不需要多次发送
    mani_pub.publish(msg);
    
    // 额外发送夹爪专用命令作为备份（参考wpb_mani的ManiGripper函数）
    set_gripper(converted_gripper_angle);
    
    Duration(1.0).sleep(); // 等待机械臂和夹爪到位
}

void set_gripper(double angle) {
    // 修复：使用1和0控制夹爪开关，而不是0.9
    // angle = 1 表示张开夹爪，angle = 0 表示闭合夹爪
    double converted_angle = angle;
    if (angle > 0.5) { // 张开状态
        converted_angle = 1.0; // 夹爪张开角度改为1.0
    } else { // 闭合状态
        converted_angle = 0.0; // 夹爪闭合角度改为0.0
    }
    
    ROS_INFO("[夹爪] 开始控制夹爪，目标角度：%.2f(转换后:%.2f)", angle, converted_angle);
    
    // 优化：增加命令发送频率，模仿wpb_mani的30Hz控制频率
    int command_count = 0;
    const int COMMAND_FREQUENCY = 30; // 30Hz频率
    const double COMMAND_INTERVAL = 1.0 / COMMAND_FREQUENCY; // 约0.033秒
    const int TOTAL_COMMANDS = 6; // 发送6次命令，约0.2秒
    
    while (command_count < TOTAL_COMMANDS && ros::ok()) {
        // 方法1：发布JointState消息（兼容性）
        sensor_msgs::JointState msg;
        msg.name = {"gripper"};
        msg.position = {converted_angle};
        msg.velocity = {15}; // 夹爪张开速度从1.2改为15
        
        ROS_INFO("[夹爪] 第%d次发送JointState命令：%.2f", command_count + 1, converted_angle);
        mani_pub.publish(msg);
        
        // 方法2：发布Float64消息（直接控制gripper_position_controller）
        std_msgs::Float64 gripper_cmd;
        gripper_cmd.data = converted_angle;
        
        ROS_INFO("[夹爪控制] 第%d次发布到gripper_position_controller：%.2f", command_count + 1, converted_angle);
        gripper_pub.publish(gripper_cmd);
        
        // 按照30Hz频率发送命令
        Duration(COMMAND_INTERVAL).sleep(); // 约0.033秒间隔
        
        command_count++;
    }
    
    ROS_INFO("[夹爪] 夹爪控制命令发送完成，共发送%d次，频率%dHz", command_count, COMMAND_FREQUENCY);
}



// 坐标转换函数：从图像坐标到世界坐标
bool imageToWorldCoordinate(int img_x, int img_y, double& world_x, double& world_y) {
    if (!tf_listener) {
        ROS_ERROR("[坐标转换] TF监听器未初始化");
        return false;
    }
    
    // 假设相机内参（需要根据实际相机标定调整）
    double fx = 525.0;  // 焦距x
    double fy = 525.0;  // 焦距y
    double cx = 320.0;  // 主点x
    double cy = 240.0;  // 主点y
    
    // 假设盒子距离相机的距离（需要根据实际测量调整）
    double box_distance = 0.5;  // 米
    
    // 计算归一化图像坐标
    double normalized_x = (img_x - cx) / fx;
    double normalized_y = (img_y - cy) / fy;
    
    // 计算相机坐标系下的3D坐标
    double camera_x = normalized_x * box_distance;
    double camera_y = normalized_y * box_distance;
    double camera_z = box_distance;
    
    // 转换到base_footprint坐标系
    geometry_msgs::PointStamped camera_point;
    camera_point.header.frame_id = "camera_link";  // 需要根据实际相机坐标系调整
    camera_point.header.stamp = ros::Time(0);
    camera_point.point.x = camera_x;
    camera_point.point.y = camera_y;
    camera_point.point.z = camera_z;
    
    geometry_msgs::PointStamped base_point;
    try {
        tf_listener->transformPoint("base_footprint", camera_point, base_point);
        world_x = base_point.point.x;
        world_y = base_point.point.y;
        ROS_INFO("[坐标转换] 图像坐标(%d,%d) -> 世界坐标(%.3f,%.3f)", img_x, img_y, world_x, world_y);
        return true;
    } catch (tf::TransformException& ex) {
        ROS_ERROR("[坐标转换] TF转换失败: %s", ex.what());
        return false;
    }
}

// 新增：机械爪位置检测函数 - 通过机械臂关节角度计算夹爪在图像中的位置
bool detect_gripper_position(const Mat& img, Rect& gripper_roi) {
    // 获取当前机械臂关节角度（需要订阅机械臂状态）
    // 这里假设机械臂关节角度已知，实际需要从机械臂状态消息获取
    
    // 机械臂参数（需要根据实际机械臂结构调整）
    double joint1_angle = 0.0;  // 基座旋转角度
    double joint2_angle = 0.0;  // 大臂俯仰角度
    double joint3_angle = 0.0;  // 小臂俯仰角度
    double joint4_angle = 0.0;  // 末端执行器角度
    
    // 机械臂连杆长度（单位：米）
    double link1_length = 0.1;   // 基座到关节2
    double link2_length = 0.128; // 关节2到关节3
    double link3_length = 0.124; // 关节3到关节4
    double link4_length = 0.024; // 关节4到夹爪中心
    
    // 计算夹爪在世界坐标系中的位置
    // 使用正向运动学计算夹爪位置
    double gripper_x = link1_length + link2_length * sin(joint2_angle) + 
                      link3_length * sin(joint2_angle + joint3_angle) + 
                      link4_length * sin(joint2_angle + joint3_angle + joint4_angle);
    
    double gripper_y = 0.0;  // 假设机械臂在机器人中心线上
    double gripper_z = link2_length * cos(joint2_angle) + 
                      link3_length * cos(joint2_angle + joint3_angle) + 
                      link4_length * cos(joint2_angle + joint3_angle + joint4_angle);
    
    // 将夹爪世界坐标转换为图像坐标
    // 这里使用简化的投影模型，实际需要相机标定参数
    
    // 相机内参（需要根据实际相机标定调整）
    double fx = 525.0;  // 焦距x
    double fy = 525.0;  // 焦距y
    double cx = 320.0;  // 主点x
    double cy = 240.0;  // 主点y
    
    // 假设相机与夹爪的相对位置（需要根据机器人结构调整）
    double camera_to_gripper_x = 0.1;  // 相机在夹爪前方0.1米
    double camera_to_gripper_y = 0.0;   // 相机与夹爪在同一中心线上
    double camera_to_gripper_z = 0.05;  // 相机在夹爪上方0.05米
    
    // 计算夹爪在相机坐标系中的位置
    double gripper_camera_x = gripper_x - camera_to_gripper_x;
    double gripper_camera_y = gripper_y - camera_to_gripper_y;
    double gripper_camera_z = gripper_z - camera_to_gripper_z;
    
    // 投影到图像平面
    if (gripper_camera_z > 0) {
        int img_x = int((gripper_camera_x / gripper_camera_z) * fx + cx);
        int img_y = int((gripper_camera_y / gripper_camera_z) * fy + cy);
        
        // 定义夹爪检测区域（以夹爪位置为中心）
        int roi_width = 100;   // 检测区域宽度
        int roi_height = 80;   // 检测区域高度
        
        int roi_x = max(0, img_x - roi_width / 2);
        int roi_y = max(0, img_y - roi_height / 2);
        roi_width = min(roi_width, img.cols - roi_x);
        roi_height = min(roi_height, img.rows - roi_y);
        
        gripper_roi = Rect(roi_x, roi_y, roi_width, roi_height);
        
        ROS_INFO("[机械爪定位] 计算夹爪位置: 图像坐标(%d,%d), 检测区域(%d,%d,%d,%d)", 
                 img_x, img_y, roi_x, roi_y, roi_width, roi_height);
        return true;
    }
    
    // 如果无法计算夹爪位置，使用基于图像特征的检测方法
    return detect_gripper_by_image_features(img, gripper_roi);
}

// 新增：基于图像特征的机械爪检测（备选方案）
bool detect_gripper_by_image_features(const Mat& img, Rect& gripper_roi) {
    // 方法1：检测机械臂的黑色轮廓特征
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    // 使用Canny边缘检测
    Mat edges;
    Canny(gray, edges, 50, 150);
    
    // 查找轮廓
    vector<vector<Point>> contours;
    findContours(edges, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // 可视化：创建带检测框的图像
    Mat display_img = img.clone();
    
    // 寻找可能代表机械臂的轮廓
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > 500 && area < 5000) { // 机械臂轮廓面积范围
            Rect bounding_rect = boundingRect(contour);
            
            // 检查轮廓位置（机械臂通常在图像下方）
            if (bounding_rect.y + bounding_rect.height > img.rows * 0.6) {
                // 检查宽高比（机械臂通常较细长）
                double aspect_ratio = (double)bounding_rect.width / bounding_rect.height;
                if (aspect_ratio < 0.5) { // 高度大于宽度
                    // 定义夹爪检测区域（在机械臂轮廓上方）
                    int roi_width = 120;
                    int roi_height = 100;
                    int roi_x = max(0, bounding_rect.x + bounding_rect.width / 2 - roi_width / 2);
                    int roi_y = max(0, bounding_rect.y - roi_height);
                    
                    gripper_roi = Rect(roi_x, roi_y, roi_width, roi_height);
                    gripper_roi = gripper_roi & Rect(0, 0, img.cols, img.rows);
                    
                    // 可视化：绘制机械臂轮廓和夹爪检测区域
                    rectangle(display_img, bounding_rect, Scalar(255, 0, 0), 2); // 蓝色框表示机械臂轮廓
                    rectangle(display_img, gripper_roi, Scalar(0, 255, 0), 2); // 绿色框表示夹爪检测区域
                    putText(display_img, "Arm Contour", Point(bounding_rect.x, bounding_rect.y - 10), 
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
                    putText(display_img, "Gripper ROI", Point(roi_x, roi_y - 10), 
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
                    
                    ROS_INFO("[机械爪定位] 基于图像特征检测到机械臂，夹爪区域(%d,%d,%d,%d)", 
                             roi_x, roi_y, roi_width, roi_height);
                    
                    // 使用可视化函数显示检测区域
                    visualize_detection_regions(img, gripper_roi, "Arm Contour");
                    
                    return true;
                }
            }
        }
    }
    
    // 方法2：使用颜色特征检测黑色机械爪
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // 检测黑色机械爪（低亮度，低饱和度）
    Mat black_mask;
    inRange(hsv, Scalar(0, 0, 0), Scalar(180, 255, 50), black_mask); // 低亮度范围检测黑色
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(black_mask, black_mask, MORPH_CLOSE, kernel);
    morphologyEx(black_mask, black_mask, MORPH_OPEN, kernel);
    
    vector<vector<Point>> black_contours;
    findContours(black_mask, black_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : black_contours) {
        double area = contourArea(contour);
        if (area > 300 && area < 3000) { // 机械爪黑色部件面积范围
            Rect bounding_rect = boundingRect(contour);
            
            // 检查位置（夹爪通常在图像中心偏下）
            if (bounding_rect.y > img.rows * 0.4 && bounding_rect.y < img.rows * 0.8) {
                // 检查宽高比（夹爪通常为矩形）
                double aspect_ratio = (double)bounding_rect.width / bounding_rect.height;
                if (aspect_ratio > 0.3 && aspect_ratio < 3.0) {
                    // 定义夹爪检测区域
                    int roi_width = 100;
                    int roi_height = 80;
                    int roi_x = max(0, bounding_rect.x + bounding_rect.width / 2 - roi_width / 2);
                    int roi_y = max(0, bounding_rect.y + bounding_rect.height / 2 - roi_height / 2);
                    
                    gripper_roi = Rect(roi_x, roi_y, roi_width, roi_height);
                    gripper_roi = gripper_roi & Rect(0, 0, img.cols, img.rows);
                    
                    // 可视化：绘制黑色部件轮廓和夹爪检测区域
                    rectangle(display_img, bounding_rect, Scalar(255, 255, 0), 2); // 青色框表示黑色部件
                    rectangle(display_img, gripper_roi, Scalar(0, 255, 255), 2); // 黄色框表示夹爪检测区域
                    putText(display_img, "Black Component", Point(bounding_rect.x, bounding_rect.y - 10), 
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
                    putText(display_img, "Gripper ROI", Point(roi_x, roi_y - 10), 
                            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
                    
                    ROS_INFO("[机械爪定位] 基于黑色检测到夹爪，区域(%d,%d,%d,%d)", 
                             roi_x, roi_y, roi_width, roi_height);
                    
                    // 使用可视化函数显示检测区域
                    visualize_detection_regions(img, gripper_roi, "Black Component");
                    
                    return true;
                }
            }
        }
    }
    
    // 如果所有方法都失败，使用默认的夹爪区域（图像中心偏下）
    int center_x = img.cols / 2;
    int center_y = img.rows * 2 / 3;  // 中心偏下位置
    int roi_width = 120;
    int roi_height = 100;
    
    int roi_x = max(0, center_x - roi_width / 2);
    int roi_y = max(0, center_y - roi_height / 2);
    roi_width = min(roi_width, img.cols - roi_x);
    roi_height = min(roi_height, img.rows - roi_y);
    
    gripper_roi = Rect(roi_x, roi_y, roi_width, roi_height);
    
    // 可视化：绘制默认夹爪检测区域
    rectangle(display_img, gripper_roi, Scalar(255, 0, 255), 2); // 紫色框表示默认夹爪区域
    putText(display_img, "Default Gripper ROI", Point(roi_x, roi_y - 10), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);
    putText(display_img, "Detection Method: Default", Point(10, img.rows - 20), 
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 255, 255), 2);
    
    ROS_WARN("[机械爪定位] 使用默认夹爪区域(%d,%d,%d,%d)", roi_x, roi_y, roi_width, roi_height);
    
    // 使用可视化函数显示检测区域
    visualize_detection_regions(img, gripper_roi, "Default");
    
    return true;
}

// 获取当前机器人位置
bool getRobotPosition(double& robot_x, double& robot_y, double& robot_yaw) {
    if (!tf_listener) {
        ROS_ERROR("[位置获取] TF监听器未初始化");
        return false;
    }
    
    tf::StampedTransform transform;
    try {
        tf_listener->lookupTransform("map", "base_footprint", ros::Time(0), transform);
        robot_x = transform.getOrigin().x();
        robot_y = transform.getOrigin().y();
        robot_yaw = tf::getYaw(transform.getRotation());
        ROS_INFO("[位置获取] 机器人位置: (%.3f, %.3f), 朝向: %.3f", robot_x, robot_y, robot_yaw);
        return true;
    } catch (tf::TransformException& ex) {
        ROS_ERROR("[位置获取] 获取机器人位置失败: %s", ex.what());
        return false;
    }
}

// 计算爪子与盒子的对齐误差
bool calculateAlignmentError(double box_world_x, double box_world_y, double& error_x, double& error_y) {
    double robot_x, robot_y, robot_yaw;
    if (!getRobotPosition(robot_x, robot_y, robot_yaw)) {
        return false;
    }
    
    // 计算相对位置
    double relative_x = box_world_x - robot_x;
    double relative_y = box_world_y - robot_y;
    
    // 转换到机器人坐标系
    error_x = relative_x * cos(robot_yaw) + relative_y * sin(robot_yaw);
    error_y = -relative_x * sin(robot_yaw) + relative_y * cos(robot_yaw);
    
    ROS_INFO("[对齐误差] 相对误差: x=%.3fm, y=%.3fm", error_x, error_y);
    return true;
}

// 模板匹配
double template_match(const Mat& img, const Mat& template_img) {
    if (img.empty() || template_img.empty()) {
        ROS_WARN("[模板匹配] 输入图像或模板为空");
        return 0.0;
    }
    
    if (template_img.rows > img.rows || template_img.cols > img.cols) {
        ROS_WARN("[模板匹配] 模板尺寸大于图像，无法匹配");
        return 0.0;
    }
    
    Mat result;
    try {
        matchTemplate(img, template_img, result, TM_CCOEFF_NORMED);
        double max_val;
        minMaxLoc(result, nullptr, &max_val);
        return max_val;
    } catch (const cv::Exception& e) {
        ROS_ERROR("[模板匹配] OpenCV异常: %s", e.what());
        return 0.0;
    }
}

Mat qd_template;

// 安全加载QD模板
void load_qd_template() {
    string pkg_path = ros::package::getPath("mobile_pkg");
    string data_dir = pkg_path + "/data";

    DIR* dir = opendir(data_dir.c_str());
    if (!dir) {
        ROS_ERROR("[模板加载] 无法打开 data 目录: %s", data_dir.c_str());
        return;
    }

    vector<string> image_files;
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            string filename = entry->d_name;
            if (filename[0] == '.') continue;
            string ext = filename;
            transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext.find(".jpg") != string::npos ||
                ext.find(".jpeg") != string::npos ||
                ext.find(".png") != string::npos ||
                ext.find(".bmp") != string::npos ||
                ext.find(".tiff") != string::npos) {
                image_files.push_back(data_dir + "/" + filename);
            }
        }
    }
    closedir(dir);

    if (image_files.empty()) {
        ROS_ERROR("[模板加载] data 目录中没有找到任何图片文件！");
        return;
    }

    string qd_path;
    for (const auto& path : image_files) {
        string lower_path = path;
        transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);
        if (lower_path.find("qd") != string::npos || 
            lower_path.find("enemy") != string::npos ||
            lower_path.find("enemy_target") != string::npos) {
            qd_path = path;
            break;
        }
    }
    
    if (qd_path.empty()) {
        qd_path = image_files[0];
        ROS_WARN("[模板加载] 未找到特定QD模板，使用第一个文件: %s", qd_path.c_str());
    }

    ROS_INFO("[模板加载] 尝试加载 QD 模板: %s", qd_path.c_str());
    qd_template = imread(qd_path, IMREAD_COLOR);
    
    if (qd_template.empty()) {
        ROS_ERROR("[模板加载] 无法加载图片: %s", qd_path.c_str());
    } else {
        if (qd_template.cols > 200 || qd_template.rows > 200) {
            resize(qd_template, qd_template, Size(100, 100));
        }
        ROS_INFO("[模板加载] QD 模板加载成功！尺寸: %dx%d", qd_template.cols, qd_template.rows);
    }
}

// 安全提取中心区域颜色
Scalar extract_center_color(const Mat& img, int center_width = 20) {
    if (img.empty()) {
        ROS_WARN("[颜色提取] 输入图像为空");
        return Scalar(0, 0, 0);
    }
    
    center_width = min(center_width, img.cols);
    int left_center_x = max(0, (img.cols - center_width) / 2);
    left_center_x = min(left_center_x, img.cols - center_width);
    
    try {
        Mat center_region = img.colRange(left_center_x, left_center_x + center_width);
        return mean(center_region);
    } catch (const cv::Exception& e) {
        ROS_ERROR("[颜色提取] 提取中心颜色失败: %s", e.what());
        return Scalar(0, 0, 0);
    }
}

// 加载角色模板并提取颜色特征
void load_character_templates() {
    string pkg_path = ros::package::getPath("mobile_pkg");
    string data_dir = pkg_path + "/data";

    DIR* dir = opendir(data_dir.c_str());
    if (!dir) {
        ROS_ERROR("[角色模板] 无法打开 data 目录: %s", data_dir.c_str());
        return;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type != DT_REG) continue;
        string filename = entry->d_name;
        if (filename[0] == '.') continue;

        string lower_name = filename;
        transform(lower_name.begin(), lower_name.end(), lower_name.begin(), ::tolower);

        string full_path = data_dir + "/" + filename;
        Mat img = imread(full_path, IMREAD_COLOR);
        if (img.empty()) {
            ROS_WARN("[角色模板] 无法读取图片: %s", full_path.c_str());
            continue;
        }

        Scalar avg_color = extract_center_color(img);

        if (lower_name.find("enemy") != string::npos ||
             lower_name.find("enemy_target") != string::npos) {
            enemy_target_color_avg = avg_color;
            has_enemy_target_template = true;
            ROS_INFO("[角色模板] 加载敌方目标模板: %s, BGR=(%.1f, %.1f, %.1f)",
                     filename.c_str(), avg_color[0], avg_color[1], avg_color[2]);
        }
        else if (lower_name.find("friendly") != string::npos ||
                 lower_name.find("friendly_target") != string::npos ||
                 lower_name.find("pink") != string::npos) {
            friendly_target_color_avg = avg_color;
            has_friendly_target_template = true;
            ROS_INFO("[角色模板] 加载友方目标模板: %s, BGR=(%.1f, %.1f, %.1f)",
                     filename.c_str(), avg_color[0], avg_color[1], avg_color[2]);
        }
    }
    closedir(dir);

    if (!has_enemy_target_template && !has_friendly_target_template) {
        ROS_WARN("[角色模板] 未加载任何角色模板！将使用默认颜色特征。");
        enemy_target_color_avg = Scalar(30, 30, 30);
        friendly_target_color_avg = Scalar(180, 105, 255);
    }
}

// 安全的肤色检测函数
int safe_skin_detect(const Mat& roi) {
    if (roi.empty()) {
        return 0;
    }
    
    try {
        Mat ycrcb;
        cvtColor(roi, ycrcb, COLOR_BGR2YCrCb);
        Mat skin_mask;
        inRange(ycrcb, Scalar(0, 133, 77), Scalar(255, 173, 127), skin_mask);
        return countNonZero(skin_mask);
    } catch (const cv::Exception& e) {
        ROS_ERROR("[肤色检测] 处理失败: %s", e.what());
        return 0;
    }
}



// wpb_mani盒子坐标回调函数
void BoxCoordCB(const wpb_mani_behaviors::Coord::ConstPtr &msg) 
{ 
    // 获取盒子检测结果并更新全局变量
    box_num = msg->name.size(); 
    box_coord_received = (box_num > 0);
    
    if (box_num > 0) {
        // 更新盒子坐标全局变量（使用第一个检测到的盒子）
        box_track_x = msg->x[0];
        box_track_y = msg->y[0];
        box_track_z = msg->z[0];
        
        // 设置抓取进行中标志
        grab_in_progress = true;
        
        // 每次检测到盒子都发送坐标
        if (wpb_mani_active) {
            grab_box_msg.position.x = msg->x[0]; 
            grab_box_msg.position.y = msg->y[0]; 
            grab_box_msg.position.z = msg->z[0]; 
            grab_box_pub.publish(grab_box_msg); 
        }
        
        // y值监控机制 - 只有在启用时才执行
        double current_y = msg->y[0];
        double y_diff = fabs(current_y - last_y_coord);
        
        if (y_monitoring_enabled) {
            // 1. 监控y值大变化（>0.1）- 仅作用于第二次和第三次抓取
            if (box_count >= 1 && y_diff > 0.1) { // box_count从0开始，1表示第二次抓取，2表示第三次抓取
                y_large_change_count++;
                ROS_INFO("[y值监控] 检测到y值大变化 (%.3f -> %.3f)，差值: %.3f，连续计数: %d", 
                         last_y_coord, current_y, y_diff, y_large_change_count);
                
                if (y_large_change_count >= 2) { // 连续2次y值差>0.1
                    ROS_WARN("[y值监控] 连续2次y值差>0.1，发布0速度并重启wpb_mani模块");
                    
                    // 发布0速度命令
                    geometry_msgs::Twist cmd_vel;
                    cmd_vel.linear.x = 0.0;
                    cmd_vel.linear.y = 0.0;
                    cmd_vel.linear.z = 0.0;
                    cmd_vel.angular.x = 0.0;
                    cmd_vel.angular.y = 0.0;
                    cmd_vel.angular.z = 0.0;
                    cmd_vel_pub.publish(cmd_vel);
                    
                    // 重置y值监控系统
                    reset_y_monitoring();
                    
                    // 重启wpb_mani模块（通过重置状态实现）
                    ROS_INFO("[wpb_mani] 重启抓取任务");
                    grab_success = false;
                    grab_in_progress = false;
                    grab_attempt_count++;
                    grab_coord_sent = false;
                }
            } else {
                y_large_change_count = 0;
            }
            
            // 2. 监控y值小变化（<0.01）- 作用于三次抓取，且每次抓取只触发一次
            if (y_diff < 0.01) {
                y_small_change_count++;
                ROS_INFO("[y值监控] 检测到y值小变化 (%.3f -> %.3f)，差值: %.3f，连续计数: %d", 
                         last_y_coord, current_y, y_diff, y_small_change_count);
                
                if (y_small_change_count >= 7 && !y_small_change_triggered) { // 连续7次y值差<0.01且未触发过
                    ROS_INFO("[y值监控] 连续5次y值差<0.01，触发立即抓取");
                    
                    // 发布0速度命令使小车停下
                    geometry_msgs::Twist cmd_vel;
                    cmd_vel.linear.x = 0.0;
                    cmd_vel.linear.y = 0.0;
                    cmd_vel.linear.z = 0.0;
                    cmd_vel.angular.x = 0.0;
                    cmd_vel.angular.y = 0.0;
                    cmd_vel.angular.z = 0.0;
                    cmd_vel_pub.publish(cmd_vel);
                    
                    // 标记为已触发（每次抓取只触发一次）
                    y_small_change_triggered = true;
                    
                    // 立即发送抓取命令
                    if (wpb_mani_active) {
                        grab_box_msg.position.x = msg->x[0]; 
                        grab_box_msg.position.y = msg->y[0]; 
                        grab_box_msg.position.z = msg->z[0]; 
                        grab_box_pub.publish(grab_box_msg); 
                        ROS_INFO("[BoxCoordCB] 立即发送抓取命令");
                        
                        // 发送抓取命令后立即禁用y值监控和wpb_mani模块
                        y_monitoring_enabled = false;
                        wpb_mani_active = false;
                        ROS_INFO("[y值监控] 发送抓取命令后，禁用y值监控和wpb_mani模块");
                    }
                }
            } else {
                y_small_change_count = 0;
            }
        }
        
        last_y_coord = current_y;
        
        // 输出详细日志
        ROS_INFO("[BoxCoordCB] 检测到盒子: x=%.3f, y=%.3f, z=%.3f", 
                 msg->x[0], msg->y[0], msg->z[0]);
    } else {
        // 未检测到盒子，重置相关标记
        grab_in_progress = false; // 重置抓取进行中标志
        box_coord_received = false; // 重置盒子坐标接收标记
    }
} 

// 重置y值监控系统函数
void reset_y_monitoring() {
    last_y_coord = 0.0;
    y_large_change_count = 0;
    y_small_change_count = 0;
    y_small_change_triggered = false;
    y_monitoring_enabled = true; // 重置时启用y值监控
    ROS_INFO("[y值监控] 重置y值监控系统");
}

// 新增：小车位置坐标回调函数
void RobotPoseCB(const geometry_msgs::Pose2D::ConstPtr &msg) 
{ 
    // 更新小车位置坐标
    robot_x = msg->x;
    robot_y = msg->y;
    robot_theta = msg->theta;
    
    ROS_DEBUG("[RobotPoseCB] 小车位置更新: x=%.3f, y=%.3f, theta=%.3f", 
              robot_x, robot_y, robot_theta);
}

// 新增：盒子和小车坐标同步输出回调函数
void BoxAndRobotCoordSyncCB(const wpb_mani_behaviors::Coord::ConstPtr &msg) 
{ 
    // 检查是否有盒子检测结果
    if (msg->name.size() > 0) {
        // 获取盒子坐标（相对于小车的坐标）
        double box_x = msg->x[0];
        double box_y = msg->y[0];
        double box_z = msg->z[0];
        
        // 输出盒子坐标和小车坐标的同步信息
        ROS_INFO("========================================");
        ROS_INFO("[抓取调试] 盒子相对坐标: x=%.3f, y=%.3f, z=%.3f", box_x, box_y, box_z);
        ROS_INFO("[抓取调试] 小车绝对坐标: x=%.3f, y=%.3f, theta=%.3f", robot_x, robot_y, robot_theta);
        ROS_INFO("[抓取调试] 抓取状态: 进行中=%s, 成功=%s, 尝试次数=%d", 
                 grab_in_progress ? "是" : "否", 
                 grab_success ? "是" : "否", 
                 grab_attempt_count);
        ROS_INFO("========================================");
        
        // 同时输出到ROS_DEBUG用于详细调试
        ROS_DEBUG("[抓取调试详细] 盒子检测数量: %zu", msg->name.size());
        for (size_t i = 0; i < msg->name.size(); i++) {
            ROS_DEBUG("[抓取调试详细] 盒子%d: 名称=%s, x=%.3f, y=%.3f, z=%.3f", 
                     i, msg->name[i].c_str(), msg->x[i], msg->y[i], msg->z[i]);
        }
    } else {
        ROS_INFO("[抓取调试] 未检测到盒子");
    }
}

// wpb_mani抓取结果回调函数
void GrabResultCB(const std_msgs::String::ConstPtr &msg) 
{ 
    // 标记已接收到抓取结果回调
    grab_result_received = true;
    
    if (msg->data == "done") 
    { 
        // 设置抓取结果标志
        grab_success = true;
        
        // 抓取成功后禁用y值监控
        y_monitoring_enabled = false;
        
        ROS_INFO("[GrabResultCB] 抓取成功，禁用y值监控");
    } 
    else if (msg->data == "failed") 
    { 
        // 设置抓取结果标志
        grab_success = false;
        
        // 增加抓取尝试次数
        grab_attempt_count++;
        
        ROS_ERROR("[GrabResultCB] 抓取失败，尝试次数: %d", grab_attempt_count);
        
        // 重置状态，准备重试
        grab_coord_sent = false;
        grab_in_progress = false;
        grab_result_received = false;
        
        // 重置y值监控系统（抓取失败后重置，避免影响下一次抓取）
        reset_y_monitoring();
        
        // 抓取失败后暂停wpb_mani模块0.5秒，然后重新启动
        wpb_mani_active = false;
        ros::Duration(0.5).sleep();
        wpb_mani_active = true;
        
        ROS_INFO("[GrabResultCB] 抓取失败处理完成，重置系统状态");
    }
}

// 改进的QD检测函数
// 新增：检测红色方框并提取内部人物区域
Rect extract_person_region_from_red_box(const Mat& img, const Rect& search_rect) {
    if (img.empty()) return Rect();
    
    Mat roi = img(search_rect);
    Mat hsv;
    cvtColor(roi, hsv, COLOR_BGR2HSV);
    
    // 检测红色方框（两个HSV范围）
    Mat red1, red2, red_mask;
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), red1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), red2);
    red_mask = red1 | red2;
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(red_mask, red_mask, MORPH_CLOSE, kernel);
    
    // 查找红色方框轮廓
    vector<vector<Point>> contours;
    findContours(red_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > 1000) { // 红色方框面积阈值
            Rect bounding_rect = boundingRect(contour);
            
            // 提取方框内部区域（人物区域）
            int margin = 10; // 边缘留白
            Rect person_rect(
                bounding_rect.x + margin,
                bounding_rect.y + margin,
                bounding_rect.width - 2 * margin,
                bounding_rect.height - 2 * margin
            );
            
            // 确保区域在图像范围内
            person_rect = person_rect & Rect(0, 0, roi.cols, roi.rows);
            
            if (person_rect.width > 50 && person_rect.height > 50) {
                // 转换为原图坐标
                Rect global_rect(
                    search_rect.x + person_rect.x,
                    search_rect.y + person_rect.y,
                    person_rect.width,
                    person_rect.height
                );
                ROS_INFO("[红色方框检测] 检测到红色方框，提取人物区域: (%d,%d,%d,%d)", 
                        global_rect.x, global_rect.y, global_rect.width, global_rect.height);
                return global_rect;
            }
        }
    }
    
    // 如果没有检测到红色方框，返回搜索区域中心部分
    Rect center_rect(
        search_rect.x + search_rect.width / 4,
        search_rect.y + search_rect.height / 4,
        search_rect.width / 2,
        search_rect.height / 2
    );
    return center_rect;
}

// 新增：人物中轴线颜色检测函数
string analyze_person_centerline_color(const Mat& person_roi) {
    if (person_roi.empty()) return "unknown";
    
    // 计算人物中轴线（垂直中心线）
    int center_x = person_roi.cols / 2;
    int strip_width = max(5, person_roi.cols / 20); // 中轴线宽度，取人物宽度的5%或最小5像素
    
    // 定义中轴线区域（垂直条带）
    Rect centerline_rect(max(0, center_x - strip_width/2), 0, 
                        min(strip_width, person_roi.cols), person_roi.rows);
    centerline_rect = centerline_rect & Rect(0, 0, person_roi.cols, person_roi.rows);
    
    if (centerline_rect.width <= 0 || centerline_rect.height <= 0) return "unknown";
    
    Mat centerline_roi = person_roi(centerline_rect);
    Scalar avg_color = mean(centerline_roi);
    
    // 基于BGR颜色空间进行颜色分类
    double blue = avg_color[0];
    double green = avg_color[1];
    double red = avg_color[2];
    
    ROS_INFO("[中轴线检测] 中轴线平均颜色: B=%.1f, G=%.1f, R=%.1f", blue, green, red);
    
    // 使用统一的classify_color函数进行颜色分类
    return classify_color(red, green, blue);
}

// 新增：中轴线分段检测函数 - 将人物中轴线分为上中下三段
vector<string> analyze_person_centerline_segments(const Mat& person_roi) {
    vector<string> segment_colors(3, "unknown");
    
    if (person_roi.empty() || person_roi.rows < 30 || person_roi.cols < 10) {
        return segment_colors;
    }
    
    // 计算中轴线宽度（人物宽度的5%，最小5像素）
    int centerline_width = max(5, person_roi.cols / 20);
    int center_x = person_roi.cols / 2;
    
    // 将人物高度分为三段
    int segment_height = person_roi.rows / 3;
    
    // 分析上段（头部区域）
    Rect top_segment(center_x - centerline_width/2, 0, centerline_width, segment_height);
    top_segment = top_segment & Rect(0, 0, person_roi.cols, person_roi.rows);
    if (top_segment.width > 0 && top_segment.height > 0) {
        Mat top_roi = person_roi(top_segment);
        Scalar avg_color = mean(top_roi);
        segment_colors[0] = classify_color(avg_color[2], avg_color[1], avg_color[0]);
    }
    
    // 分析中段（躯干区域）
    Rect middle_segment(center_x - centerline_width/2, segment_height, centerline_width, segment_height);
    middle_segment = middle_segment & Rect(0, 0, person_roi.cols, person_roi.rows);
    if (middle_segment.width > 0 && middle_segment.height > 0) {
        Mat middle_roi = person_roi(middle_segment);
        Scalar avg_color = mean(middle_roi);
        segment_colors[1] = classify_color(avg_color[2], avg_color[1], avg_color[0]);
    }
    
    // 分析下段（腿部区域）
    Rect bottom_segment(center_x - centerline_width/2, 2*segment_height, centerline_width, person_roi.rows - 2*segment_height);
    bottom_segment = bottom_segment & Rect(0, 0, person_roi.cols, person_roi.rows);
    if (bottom_segment.width > 0 && bottom_segment.height > 0) {
        Mat bottom_roi = person_roi(bottom_segment);
        Scalar avg_color = mean(bottom_roi);
        segment_colors[2] = classify_color(avg_color[2], avg_color[1], avg_color[0]);
    }
    
    return segment_colors;
}

// 新增：中轴线颜色分布统计函数
map<string, int> get_centerline_color_distribution(const Mat& person_roi) {
    map<string, int> color_distribution;
    
    if (person_roi.empty() || person_roi.rows < 20 || person_roi.cols < 10) {
        return color_distribution;
    }
    
    // 计算中轴线宽度（人物宽度的5%，最小5像素）
    int centerline_width = max(5, person_roi.cols / 20);
    int center_x = person_roi.cols / 2;
    
    // 定义中轴线区域
    Rect centerline_rect(center_x - centerline_width/2, 0, centerline_width, person_roi.rows);
    centerline_rect = centerline_rect & Rect(0, 0, person_roi.cols, person_roi.rows);
    
    if (centerline_rect.width <= 0 || centerline_rect.height <= 0) {
        return color_distribution;
    }
    
    // 遍历中轴线上的每个像素，统计颜色分布
    Mat centerline_roi = person_roi(centerline_rect);
    
    for (int y = 0; y < centerline_roi.rows; y++) {
        for (int x = 0; x < centerline_roi.cols; x++) {
            Vec3b pixel = centerline_roi.at<Vec3b>(y, x);
            double blue = pixel[0];
            double green = pixel[1];
            double red = pixel[2];
            
            string color = classify_color(red, green, blue);
            color_distribution[color]++;
        }
    }
    
    return color_distribution;
}

// 新增：智能人物服装颜色分析函数（基于中轴线颜色分布）
string analyze_person_clothing_color(const Mat& person_roi) {
    if (person_roi.empty()) return "unknown";
    
    // 获取中轴线颜色分布统计
    map<string, int> color_distribution = get_centerline_color_distribution(person_roi);
    
    // 如果颜色分布为空，返回unknown
    if (color_distribution.empty()) {
        return "unknown";
    }
    
    // 找到分布最多的颜色
    string dominant_color = "unknown";
    int max_count = 0;
    
    for (const auto& pair : color_distribution) {
        if (pair.second > max_count && pair.first != "unknown") {
            max_count = pair.second;
            dominant_color = pair.first;
        }
    }
    
    // 输出颜色分布统计信息
    ROS_INFO("[服装颜色分析] 中轴线颜色分布统计:");
    for (const auto& pair : color_distribution) {
        ROS_INFO("  %s: %d 像素", pair.first.c_str(), pair.second);
    }
    ROS_INFO("[服装颜色分析] 分布最多的颜色: %s (%d 像素)", dominant_color.c_str(), max_count);
    
    // 同时进行中轴线颜色检测（用于对比）
    string centerline_color = analyze_person_centerline_color(person_roi);
    ROS_INFO("[中轴线检测] 中轴线平均颜色: %s", centerline_color.c_str());
    
    return dominant_color;
}

// 新增：改进的模板匹配函数，专注人物特征
double improved_template_match(const Mat& img, const Mat& template_img, const string& clothing_color) {
    if (img.empty() || template_img.empty()) return 0.0;
    
    // 调整图像尺寸以匹配模板
    Mat img_resized, template_resized;
    resize(img, img_resized, Size(100, 100));
    resize(template_img, template_resized, Size(100, 100));
    
    // 转换为灰度图
    Mat img_gray, template_gray;
    cvtColor(img_resized, img_gray, COLOR_BGR2GRAY);
    cvtColor(template_resized, template_gray, COLOR_BGR2GRAY);
    
    // 使用多种匹配方法
    Mat result;
    matchTemplate(img_gray, template_gray, result, TM_CCOEFF_NORMED);
    
    double max_val;
    minMaxLoc(result, NULL, &max_val, NULL, NULL);
    
    // 根据服装颜色调整匹配得分
    double adjusted_score = max_val;
    
    // 敌方特征颜色：提高匹配得分权重
    if (clothing_color == "black" || clothing_color == "dark_gray" || clothing_color == "gray") {
        adjusted_score *= 1.2; // 提高20%权重
        ROS_INFO("[模板匹配] 检测到敌方特征颜色 %s，提高权重20%%", clothing_color.c_str());
    }
    // 友方特征颜色：降低匹配得分权重
    else if (clothing_color == "pink" || clothing_color == "pinkish" || clothing_color == "white" || clothing_color == "light_gray") {
        adjusted_score *= 0.8; // 降低20%权重
        ROS_INFO("[模板匹配] 检测到友方特征颜色 %s，降低权重20%%", clothing_color.c_str());
    }
    // 中性颜色：保持原权重
    else if (clothing_color != "unknown") {
        ROS_INFO("[模板匹配] 检测到中性颜色 %s，保持原权重", clothing_color.c_str());
    }
    
    return min(adjusted_score, 1.0); // 确保不超过1.0
}

void detect_qd(const Mat& img) {
    if (qd_template.empty()) {
        ROS_WARN("[敌方目标识别] 敌方目标模板未加载，跳过识别");
        return;
    }

    if (img.empty() || img.cols < 100 || img.rows < 100) {
        ROS_WARN("[敌方目标识别] 图像尺寸太小或为空: %dx%d", img.cols, img.rows);
        return;
    }

    // 创建可视化图像副本
    Mat display_img = img.clone();
    
    int mid_x = img.cols / 2;
    
    // 定义左右搜索区域
    Rect left_search_rect(0, 0, mid_x, img.rows);
    Rect right_search_rect(mid_x, 0, img.cols - mid_x, img.rows);
    
    // 在敌方目标识别任务时显示检测区域（持续5秒）
    if (current_step == DETECT_QD) {
        // 使用全局变量控制检测区域显示
        static bool show_qd_detection = false;
        static int qd_frame_counter = 0;
        const int QD_MAX_FRAMES = 150; // 约5秒，30fps
        
        // 当开始敌方目标识别时，重置计数器并开始显示检测区域
        if (!show_qd_detection) {
            qd_frame_counter = 0;
            show_qd_detection = true;
            ROS_INFO("[敌方目标识别] 开始显示敌方目标识别检测区域，持续%d帧（约5秒）", QD_MAX_FRAMES);
        }
        
        // 显示检测区域持续150帧（约5秒）
        if (show_qd_detection && qd_frame_counter < QD_MAX_FRAMES) {
            qd_frame_counter++;
            
            // 绘制左右搜索区域
            rectangle(display_img, left_search_rect, Scalar(255, 0, 0), 2); // 蓝色框表示左侧搜索区域
            rectangle(display_img, right_search_rect, Scalar(0, 0, 255), 2); // 红色框表示右侧搜索区域
            
            putText(display_img, "Left Search Area", Point(left_search_rect.x + 10, left_search_rect.y + 30), 
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
            putText(display_img, "Right Search Area", Point(right_search_rect.x + 10, right_search_rect.y + 30), 
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            
            // 显示剩余帧数信息
            string frame_info = "Enemy Detection: " + to_string(QD_MAX_FRAMES - qd_frame_counter) + " frames left";
            putText(display_img, frame_info, Point(10, display_img.rows - 30), 
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
        } else if (show_qd_detection) {
            // 5秒后停止显示检测区域
            show_qd_detection = false;
            ROS_INFO("[敌方目标识别] 敌方目标识别检测区域显示完成，共显示%d帧", QD_MAX_FRAMES);
        }
    }
    
    // 提取红色方框内部的人物区域
    Rect left_person_rect = extract_person_region_from_red_box(img, left_search_rect);
    Rect right_person_rect = extract_person_region_from_red_box(img, right_search_rect);
    
    // 绘制人物检测区域
    if (left_person_rect.area() > 0) {
        rectangle(display_img, left_person_rect, Scalar(0, 255, 255), 2); // 黄色框表示左侧人物区域
        putText(display_img, "Left Person", Point(left_person_rect.x + 10, left_person_rect.y + 30), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
    }
    if (right_person_rect.area() > 0) {
        rectangle(display_img, right_person_rect, Scalar(255, 255, 0), 2); // 青色框表示右侧人物区域
        putText(display_img, "Right Person", Point(right_person_rect.x + 10, right_person_rect.y + 30), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
    }
    
    // 如果检测到红色方框，使用提取的人物区域；否则使用原始搜索区域
    Mat left_roi = left_person_rect.area() > 0 ? img(left_person_rect) : img(left_search_rect);
    Mat right_roi = right_person_rect.area() > 0 ? img(right_person_rect) : img(right_search_rect);

    // 使用智能服装颜色分析
    string left_clothing_color = analyze_person_clothing_color(left_roi);
    string right_clothing_color = analyze_person_clothing_color(right_roi);
    
    ROS_INFO("[服装颜色分析] 左人物服装颜色: %s", left_clothing_color.c_str());
    ROS_INFO("[服装颜色分析] 右人物服装颜色: %s", right_clothing_color.c_str());
    
    // 中轴线检测：分析人物中轴线上的颜色分布
string left_centerline_color = analyze_person_centerline_color(left_roi);
string right_centerline_color = analyze_person_centerline_color(right_roi);

ROS_INFO("[中轴线检测] 左人物中轴线颜色: %s", left_centerline_color.c_str());
ROS_INFO("[中轴线检测] 右人物中轴线颜色: %s", right_centerline_color.c_str());

// 中轴线分段检测：将人物中轴线分为上中下三段，分析各段颜色分布
vector<string> left_segment_colors = analyze_person_centerline_segments(left_roi);
vector<string> right_segment_colors = analyze_person_centerline_segments(right_roi);

// 输出分段检测结果
ROS_INFO("[中轴线分段检测] 左人物: 上段=%s, 中段=%s, 下段=%s", 
         left_segment_colors[0].c_str(), left_segment_colors[1].c_str(), left_segment_colors[2].c_str());
ROS_INFO("[中轴线分段检测] 右人物: 上段=%s, 中段=%s, 下段=%s", 
         right_segment_colors[0].c_str(), right_segment_colors[1].c_str(), right_segment_colors[2].c_str());

// 分析中轴线颜色分布统计
map<string, int> left_color_distribution = get_centerline_color_distribution(left_roi);
map<string, int> right_color_distribution = get_centerline_color_distribution(right_roi);

// 输出颜色分布统计
ROS_INFO("[中轴线颜色分布] 左人物颜色分布:");
for (const auto& pair : left_color_distribution) {
    ROS_INFO("  %s: %d像素", pair.first.c_str(), pair.second);
}
ROS_INFO("[中轴线颜色分布] 右人物颜色分布:");
for (const auto& pair : right_color_distribution) {
    ROS_INFO("  %s: %d像素", pair.first.c_str(), pair.second);
}

    // 使用改进的模板匹配算法（考虑服装颜色）
    double left_score = improved_template_match(left_roi, qd_template, left_clothing_color);
    double right_score = improved_template_match(right_roi, qd_template, right_clothing_color);

    int left_skin = safe_skin_detect(left_roi);
    int right_skin = safe_skin_detect(right_roi);

    Scalar left_avg = extract_center_color(left_roi);
    Scalar right_avg = extract_center_color(right_roi);

    auto color_distance = [](const Scalar& a, const Scalar& b) -> double {
        double dr = a[2] - b[2];
        double dg = a[1] - b[1];
        double db = a[0] - b[0];
        return sqrt(dr*dr + dg*dg + db*db);
    };

    double left_to_enemy = 1e9, left_to_friendly = 1e9;
    double right_to_enemy = 1e9, right_to_friendly = 1e9;

    if (has_enemy_target_template) {
        left_to_enemy = color_distance(left_avg, enemy_target_color_avg);
        right_to_enemy = color_distance(right_avg, enemy_target_color_avg);
    }
    if (has_friendly_target_template) {
        left_to_friendly = color_distance(left_avg, friendly_target_color_avg);
        right_to_friendly = color_distance(right_avg, friendly_target_color_avg);
    }

    // 基于服装颜色的智能权重调整
    double left_weighted = left_score;
    double right_weighted = right_score;
    
    // 敌方特征颜色（黑色、深灰色、灰色）提高权重
    if (left_clothing_color == "black" || left_clothing_color == "dark_gray" || left_clothing_color == "gray") {
        left_weighted += 0.3;
        ROS_INFO("[敌方目标识别] 左侧检测到敌方特征颜色 %s，提高权重0.3", left_clothing_color.c_str());
    }
    if (right_clothing_color == "black" || right_clothing_color == "dark_gray" || right_clothing_color == "gray") {
        right_weighted += 0.3;
        ROS_INFO("[敌方目标识别] 右侧检测到敌方特征颜色 %s，提高权重0.3", right_clothing_color.c_str());
    }
    
    // 友方特征颜色（粉色、浅粉色、白色、浅灰色）降低权重
    if (left_clothing_color == "pink" || left_clothing_color == "pinkish" || left_clothing_color == "white" || left_clothing_color == "light_gray") {
        left_weighted -= 0.2;
        ROS_INFO("[敌方目标识别] 左侧检测到友方特征颜色 %s，降低权重0.2", left_clothing_color.c_str());
    }
    if (right_clothing_color == "pink" || right_clothing_color == "pinkish" || right_clothing_color == "white" || right_clothing_color == "light_gray") {
        right_weighted -= 0.2;
        ROS_INFO("[敌方目标识别] 右侧检测到友方特征颜色 %s，降低权重0.2", right_clothing_color.c_str());
    }

    // 如果没有模板，使用服装颜色分析结果
    if (!has_enemy_target_template && !has_friendly_target_template) {
        if (left_clothing_color == "black" && left_clothing_color != "pink") left_weighted += 0.3;
        if (right_clothing_color == "black" && right_clothing_color != "pink") right_weighted += 0.3;
        if (left_clothing_color == "pink" && left_clothing_color != "black") left_weighted -= 0.2;
        if (right_clothing_color == "pink" && right_clothing_color != "black") right_weighted -= 0.2;
    }

    if (left_skin > 500) left_weighted += 0.1;
    if (right_skin > 500) right_weighted += 0.1;

    ROS_INFO("[敌方目标识别] 左侧: 模板=%.2f, 肤色=%d, 服装颜色=%s, 综合=%.2f",
             left_score, left_skin, left_clothing_color.c_str(), left_weighted);
    ROS_INFO("[敌方目标识别] 右侧: 模板=%.2f, 肤色=%d, 服装颜色=%s, 综合=%.2f",
             right_score, right_skin, right_clothing_color.c_str(), right_weighted);

    // 智能阈值调整
    double threshold = 0.6;
    
    // 如果检测到黑色服装（敌方特征），降低阈值
    if (left_clothing_color == "black" || right_clothing_color == "black") {
        threshold = 0.5;
        ROS_INFO("[敌方目标识别] 检测到黑色服装（敌方特征），降低识别阈值至%.2f", threshold);
    }
    
    // 如果检测到粉色服装（友方特征），提高阈值
    if (left_clothing_color == "pink" || right_clothing_color == "pink") {
        threshold = 0.7;
        ROS_INFO("[敌方目标识别] 检测到粉色服装（友方特征），提高识别阈值至%.2f", threshold);
    }

    // 智能决策 - 改进逻辑：如果一侧明显高于另一侧，即使未达到阈值也优先选择
    if (left_weighted > threshold && left_weighted > right_weighted + 0.1) {
        qd_side = "left";
        ROS_INFO("[敌方目标识别] 检测到敌方目标在左侧（服装颜色: %s）", left_clothing_color.c_str());
    } else if (right_weighted > threshold && right_weighted > left_weighted + 0.1) {
        qd_side = "right";
        ROS_INFO("[敌方目标识别] 检测到敌方目标在右侧（服装颜色: %s）", right_clothing_color.c_str());
    } else if (left_weighted > right_weighted + 0.2) {
        // 如果左侧明显高于右侧（差距>0.2），即使未达到阈值也选择左侧
        qd_side = "left";
        ROS_INFO("[敌方目标识别] 左侧明显高于右侧（差距%.2f），决策为左侧", left_weighted - right_weighted);
    } else if (right_weighted > left_weighted + 0.2) {
        // 如果右侧明显高于左侧（差距>0.2），即使未达到阈值也选择右侧
        qd_side = "right";
        ROS_INFO("[敌方目标识别] 右侧明显高于左侧（差距%.2f），决策为右侧", right_weighted - left_weighted);
    } else {
        // 如果两侧得分确实相近（差距≤0.2），使用服装颜色特征决策
        if (left_clothing_color == "black" && right_clothing_color != "black") {
            qd_side = "left";
            ROS_WARN("[敌方目标识别] 两侧得分相近（差距%.2f），但左侧检测到黑色服装，决策为左侧", abs(left_weighted - right_weighted));
        } else if (right_clothing_color == "black" && left_clothing_color != "black") {
            qd_side = "right";
            ROS_WARN("[敌方目标识别] 两侧得分相近（差距%.2f），但右侧检测到黑色服装，决策为右侧", abs(left_weighted - right_weighted));
        } else {
            // 默认决策
            qd_side = left_weighted > right_weighted ? "left" : "right";
            ROS_WARN("[敌方目标识别] 两侧得分相近（差距%.2f），使用默认决策: %s", abs(left_weighted - right_weighted), qd_side.c_str());
        }
    }

    qd_ok = true;
    ROS_INFO("[敌方目标识别] 检测到敌方目标，位于 %s 侧", qd_side.c_str());
    
    // 绘制检测结果
    string result_text = "Enemy Detected: " + qd_side;
    Scalar result_color = (qd_side == "left") ? Scalar(255, 0, 0) : Scalar(0, 0, 255); // 左侧蓝色，右侧红色
    
    // 在图像顶部显示检测结果
    putText(display_img, result_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, result_color, 2);
    
    // 在检测到的侧边绘制高亮框
    if (qd_side == "left") {
        rectangle(display_img, left_search_rect, Scalar(255, 0, 0), 4); // 加粗蓝色框
        putText(display_img, "ENEMY", Point(left_search_rect.x + 10, left_search_rect.y + 50), 
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 0, 0), 2);
    } else {
        rectangle(display_img, right_search_rect, Scalar(0, 0, 255), 4); // 加粗红色框
        putText(display_img, "ENEMY", Point(right_search_rect.x + 10, right_search_rect.y + 50), 
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
    }
    
    // 显示服装颜色信息
    putText(display_img, "Left Clothing: " + left_clothing_color, Point(10, 60), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    putText(display_img, "Right Clothing: " + right_clothing_color, Point(10, 80), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    
    // 显示得分信息
    putText(display_img, "Left Score: " + to_string(left_weighted).substr(0, 4), Point(10, 100), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    putText(display_img, "Right Score: " + to_string(right_weighted).substr(0, 4), Point(10, 120), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    
    // 更新原图像以显示可视化结果
    const_cast<Mat&>(img) = display_img;
}

// 改进的方块检测函数 - 基于ck.cpp的精确颜色阈值和位置验证
void detect_box(const Mat& img) {
    if (img.empty()) {
        ROS_WARN("[方块识别] 图像为空");
        return;
    }

    // 定义检测区域（图像中心区域）
    int img_center_x = img.cols / 2;
    int img_center_y = img.rows / 2;
    int roi_width = 400;  // 检测区域宽度
    int roi_height = 300; // 检测区域高度
    
    int roi_x = max(0, img_center_x - roi_width / 2);
    int roi_y = max(0, img_center_y - roi_height / 2);
    roi_width = min(roi_width, img.cols - roi_x);
    roi_height = min(roi_height, img.rows - roi_y);
    
    Rect detection_roi(roi_x, roi_y, roi_width, roi_height);
    
    // 调用可视化函数显示检测区域
    visualize_detection_regions(img, detection_roi, "Box Detection");

    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // 基于ck.cpp的精确颜色阈值范围
    Mat red1, red2, red, yellow, blue;
    
    // 红色检测（两个HSV范围）- 调整阈值避免误识别
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), red1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), red2);
    red = red1 | red2;
    
    // 黄色检测
    inRange(hsv, Scalar(20, 100, 100), Scalar(30, 255, 255), yellow);
    
    // 蓝色检测 - 调整阈值避免与红色混淆
    inRange(hsv, Scalar(100, 120, 70), Scalar(130, 255, 255), blue);

    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(red, red, MORPH_CLOSE, kernel);
    morphologyEx(yellow, yellow, MORPH_CLOSE, kernel);
    morphologyEx(blue, blue, MORPH_CLOSE, kernel);

    // 查找轮廓并验证形状和位置
    auto detect_color_with_position = [&img](const Mat& mask, const string& color_name) -> bool {
        vector<vector<Point>> contours;
        findContours(mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area < 300) continue; // 降低面积阈值，提高灵敏度
            
            // 计算轮廓的边界矩形
            RotatedRect rect = minAreaRect(contour);
            float aspect_ratio = max(rect.size.width, rect.size.height) / 
                               min(rect.size.width, rect.size.height);
            
            // 验证形状（方块应该近似正方形）
            if (aspect_ratio > 2.5) continue; // 放宽宽高比限制
            
            // 验证位置（盒子应该在图像中心区域）
            Point2f center = rect.center;
            int img_center_x = img.cols / 2;
            int img_center_y = img.rows / 2;
            
            // 盒子应该在图像中心±200像素范围内
            if (abs(center.x - img_center_x) > 200 || abs(center.y - img_center_y) > 200) {
                continue;
            }
            
            ROS_INFO("[方块识别] 检测到%s方块，面积=%.1f, 宽高比=%.2f, 位置=(%.1f,%.1f)", 
                    color_name.c_str(), area, aspect_ratio, center.x, center.y);
            return true;
        }
        return false;
    };

    string detected_color = "";
    if (detect_color_with_position(red, "红色")) {
        detected_color = "red";
    } else if (detect_color_with_position(yellow, "黄色")) {
        detected_color = "yellow";
    } else if (detect_color_with_position(blue, "蓝色")) {
        detected_color = "blue";
    }

    if (!detected_color.empty()) {
        box_color = detected_color;
        box_ok = true;
        ROS_INFO("[方块识别] 确认检测到%s方块", box_color.c_str());
    } else {
        ROS_WARN("[方块识别] 未检测到有效方块");
    }
}

// 新增：颜色方框检测函数
string detect_color_box(const Mat& img) {
    if (img.empty()) return "";
    
    // 创建可视化图像副本
    Mat display_img = img.clone();
    
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // 检测红色方框（在航点4位置）
    Mat red1, red2, red;
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), red1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), red2);
    red = red1 | red2;
    
    // 检测蓝色方框
    Mat blue;
    inRange(hsv, Scalar(100, 120, 70), Scalar(130, 255, 255), blue);
    
    // 检测黄色方框
    Mat yellow;
    inRange(hsv, Scalar(20, 100, 100), Scalar(30, 255, 255), yellow);
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(red, red, MORPH_CLOSE, kernel);
    morphologyEx(blue, blue, MORPH_CLOSE, kernel);
    morphologyEx(yellow, yellow, MORPH_CLOSE, kernel);
    
    // 计算各颜色区域的面积
    vector<vector<Point>> red_contours, blue_contours, yellow_contours;
    findContours(red, red_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(blue, blue_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(yellow, yellow_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    double red_area = 0, blue_area = 0, yellow_area = 0;
    for (const auto& c : red_contours) {
        red_area += contourArea(c);
    }
    for (const auto& c : blue_contours) {
        blue_area += contourArea(c);
    }
    for (const auto& c : yellow_contours) {
        yellow_area += contourArea(c);
    }
    
    ROS_INFO("[颜色方框检测] 红色面积: %.1f, 蓝色面积: %.1f, 黄色面积: %.1f", 
             red_area, blue_area, yellow_area);
    
    // 根据当前盒子颜色选择对应的方框
    string detected_box = "";
    if (box_color == "red" && red_area > 1000) {
        detected_box = "red";
    } else if (box_color == "blue" && blue_area > 1000) {
        detected_box = "blue";
    } else if (box_color == "yellow" && yellow_area > 1000) {
        detected_box = "yellow";
    }
    
    // 可视化：绘制检测结果
    if (!detected_box.empty()) {
        // 在图像顶部显示检测结果
        string result_text = "Color Box Detected: " + detected_box;
        Scalar result_color;
        if (detected_box == "red") result_color = Scalar(0, 0, 255);
        else if (detected_box == "blue") result_color = Scalar(255, 0, 0);
        else result_color = Scalar(0, 255, 255);
        
        putText(display_img, result_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, result_color, 2);
        
        // 显示各颜色面积信息
        putText(display_img, "Red Area: " + to_string((int)red_area), Point(10, 60), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1);
        putText(display_img, "Blue Area: " + to_string((int)blue_area), Point(10, 80), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 1);
        putText(display_img, "Yellow Area: " + to_string((int)yellow_area), Point(10, 100), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
        
        // 绘制检测到的颜色方框区域
        // 这里可以添加具体的方框位置绘制代码，但需要先找到方框位置
        // 暂时只显示文本信息
    } else {
        // 显示未检测到有效方框
        putText(display_img, "No Color Box Detected", Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(255, 255, 255), 2);
        
        putText(display_img, "Red Area: " + to_string((int)red_area), Point(10, 60), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1);
        putText(display_img, "Blue Area: " + to_string((int)blue_area), Point(10, 80), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 1);
        putText(display_img, "Yellow Area: " + to_string((int)yellow_area), Point(10, 100), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
    }
    
    // 更新原图像以显示可视化结果
    const_cast<Mat&>(img) = display_img;
    
    return detected_box;
}

// 新增：计算方框在图像中的位置
Point get_color_box_position(const Mat& img, const string& color) {
    if (img.empty()) return Point(-1, -1);
    
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    Mat color_mask;
    if (color == "red") {
        Mat red1, red2;
        inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), red1);
        inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), red2);
        color_mask = red1 | red2;
    } else if (color == "blue") {
        inRange(hsv, Scalar(100, 120, 70), Scalar(130, 255, 255), color_mask);
    } else if (color == "yellow") {
        inRange(hsv, Scalar(20, 100, 100), Scalar(30, 255, 255), color_mask);
    } else {
        return Point(-1, -1);
    }
    
    // 形态学操作
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(color_mask, color_mask, MORPH_CLOSE, kernel);
    
    vector<vector<Point>> contours;
    findContours(color_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (contours.empty()) return Point(-1, -1);
    
    // 找到最大的轮廓
    double max_area = 0;
    vector<Point> largest_contour;
    for (const auto& c : contours) {
        double area = contourArea(c);
        if (area > max_area) {
            max_area = area;
            largest_contour = c;
        }
    }
    
    if (max_area < 1000) return Point(-1, -1);
    
    // 计算轮廓的中心点
    Moments m = moments(largest_contour);
    if (m.m00 == 0) return Point(-1, -1);
    
    int center_x = int(m.m10 / m.m00);
    int center_y = int(m.m01 / m.m00);
    
    return Point(center_x, center_y);
}

// 辅助函数：检查颜色是否与检测区域边缘接触
bool check_color_touches_edge(const Mat& mask) {
    if (mask.empty()) return false;
    
    // 检查上边缘
    Mat top_edge = mask.row(0);
    if (countNonZero(top_edge) > 0) return true;
    
    // 检查下边缘
    Mat bottom_edge = mask.row(mask.rows - 1);
    if (countNonZero(bottom_edge) > 0) return true;
    
    // 检查左边缘
    Mat left_edge = mask.col(0);
    if (countNonZero(left_edge) > 0) return true;
    
    // 检查右边缘
    Mat right_edge = mask.col(mask.cols - 1);
    if (countNonZero(right_edge) > 0) return true;
    
    return false;
}

// 新增：航点4智能颜色检测函数 - 避免背景垃圾桶干扰
string detect_gripper_color_smart(const Mat& img) {
    if (img.empty()) {
        ROS_WARN("[智能颜色检测] 图像为空");
        return "";
    }
    
    ROS_INFO("[智能颜色检测] 开始智能检测爪子区域颜色，避免背景垃圾桶干扰");
    
    // 定义检测区域（以爪子中心为中心）
    int gripper_center_x = img.cols / 2;
    int gripper_center_y = img.rows / 2;
    int region_size = 100;  // 100x100像素检测区域
    
    // 修正：检测区域应该向下移动2个正方形边长（约200像素）
    int adjusted_center_y = gripper_center_y + 200;
    
    // 计算检测区域边界
    int roi_x = max(0, gripper_center_x - region_size / 2);
    int roi_y = max(0, adjusted_center_y - region_size / 2);
    int roi_width = min(region_size, img.cols - roi_x);
    int roi_height = min(region_size, img.rows - roi_y);
    
    Rect gripper_roi(roi_x, roi_y, roi_width, roi_height);
    
    if (gripper_roi.width <= 0 || gripper_roi.height <= 0) {
        ROS_WARN("[智能颜色检测] 检测区域无效");
        return "";
    }
    
    Mat roi_img = img(gripper_roi);
    
    // 转换到HSV颜色空间
    Mat hsv;
    cvtColor(roi_img, hsv, COLOR_BGR2HSV);
    
    // 检测红黄蓝三种颜色
    Mat red1, red2, red_mask, yellow_mask, blue_mask;
    
    // 检测红色
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), red1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), red2);
    red_mask = red1 | red2;
    
    // 检测黄色
    inRange(hsv, Scalar(20, 100, 100), Scalar(30, 255, 255), yellow_mask);
    
    // 检测蓝色
    inRange(hsv, Scalar(100, 120, 70), Scalar(130, 255, 255), blue_mask);
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(red_mask, red_mask, MORPH_CLOSE, kernel);
    morphologyEx(yellow_mask, yellow_mask, MORPH_CLOSE, kernel);
    morphologyEx(blue_mask, blue_mask, MORPH_CLOSE, kernel);
    
    // 计算各颜色区域的面积
    int red_area = countNonZero(red_mask);
    int yellow_area = countNonZero(yellow_mask);
    int blue_area = countNonZero(blue_mask);
    
    // 计算检测区域总面积
    int total_area = roi_img.rows * roi_img.cols;
    
    // 计算颜色区域占比
    double red_ratio = (double)red_area / total_area;
    double yellow_ratio = (double)yellow_area / total_area;
    double blue_ratio = (double)blue_area / total_area;
    
    ROS_INFO("[智能颜色检测] 夹爪区域颜色面积 - 红:%d(%.2f%%), 黄:%d(%.2f%%), 蓝:%d(%.2f%%)", 
             red_area, red_ratio * 100, yellow_area, yellow_ratio * 100, blue_area, blue_ratio * 100);
    
    // 智能检测逻辑：检查颜色是否与检测区域边缘接触
    // 盒子在检测区域中间，如果颜色色块没有与边缘接触，说明是盒子颜色
    // 如果整个检测区域只有一种颜色，说明是盒子颜色
    
    // 检查颜色是否与边缘接触
    bool red_touches_edge = check_color_touches_edge(red_mask);
    bool yellow_touches_edge = check_color_touches_edge(yellow_mask);
    bool blue_touches_edge = check_color_touches_edge(blue_mask);
    
    ROS_INFO("[智能颜色检测] 颜色接触边缘情况 - 红:%d, 黄:%d, 蓝:%d", 
             red_touches_edge, yellow_touches_edge, blue_touches_edge);
    
    // 智能判断逻辑
    string detected_color = "";
    
    // 使用与航点3相同的百分比阈值：主阈值1.5%，宽松阈值0.8%
    double area_ratio_threshold = 0.015;  // 1.5%主阈值
    double relaxed_threshold = 0.008;       // 0.8%宽松阈值
    int min_pixel_threshold = 50;          // 最小像素阈值
    
    // 情况1：只有一种颜色且不与边缘接触，说明是盒子颜色
    int color_count = 0;
    if (red_area > min_pixel_threshold && red_ratio > area_ratio_threshold) color_count++;
    if (yellow_area > min_pixel_threshold && yellow_ratio > area_ratio_threshold) color_count++;
    if (blue_area > min_pixel_threshold && blue_ratio > area_ratio_threshold) color_count++;
    
    if (color_count == 1) {
        if (red_area > min_pixel_threshold && red_ratio > area_ratio_threshold && !red_touches_edge) {
            detected_color = "red";
            ROS_INFO("[智能颜色检测] 检测到单一红色且不与边缘接触，确定为盒子颜色");
        } else if (yellow_area > min_pixel_threshold && yellow_ratio > area_ratio_threshold && !yellow_touches_edge) {
            detected_color = "yellow";
            ROS_INFO("[智能颜色检测] 检测到单一黄色且不与边缘接触，确定为盒子颜色");
        } else if (blue_area > min_pixel_threshold && blue_ratio > area_ratio_threshold && !blue_touches_edge) {
            detected_color = "blue";
            ROS_INFO("[智能颜色检测] 检测到单一蓝色且不与边缘接触，确定为盒子颜色");
        }
    }
    
    // 情况2：多种颜色，选择不与边缘接触且面积最大的颜色
    if (detected_color.empty() && color_count > 1) {
        ROS_INFO("[智能颜色检测] 检测到多种颜色，选择不与边缘接触且面积最大的颜色");
        
        int max_area = 0;
        string candidate_color = "";
        
        if (red_area > min_pixel_threshold && red_ratio > area_ratio_threshold && red_area > max_area && !red_touches_edge) {
            max_area = red_area;
            candidate_color = "red";
        }
        if (yellow_area > min_pixel_threshold && yellow_ratio > area_ratio_threshold && yellow_area > max_area && !yellow_touches_edge) {
            max_area = yellow_area;
            candidate_color = "yellow";
        }
        if (blue_area > min_pixel_threshold && blue_ratio > area_ratio_threshold && blue_area > max_area && !blue_touches_edge) {
            max_area = blue_area;
            candidate_color = "blue";
        }
        
        if (!candidate_color.empty()) {
            detected_color = candidate_color;
            ROS_INFO("[智能颜色检测] 选择颜色: %s (面积: %d)", candidate_color.c_str(), max_area);
        }
    }
    
    // 情况3：如果所有颜色都与边缘接触，选择面积最大的颜色
    if (detected_color.empty()) {
        ROS_WARN("[智能颜色检测] 所有颜色都与边缘接触，选择面积最大的颜色");
        
        int max_area = 0;
        if (red_area > min_pixel_threshold && red_ratio > area_ratio_threshold && red_area > max_area) {
            max_area = red_area;
            detected_color = "red";
        }
        if (yellow_area > min_pixel_threshold && yellow_ratio > area_ratio_threshold && yellow_area > max_area) {
            max_area = yellow_area;
            detected_color = "yellow";
        }
        if (blue_area > min_pixel_threshold && blue_ratio > area_ratio_threshold && blue_area > max_area) {
            max_area = blue_area;
            detected_color = "blue";
        }
        
        // 如果使用主阈值没有检测到颜色，尝试使用宽松阈值
        if (detected_color.empty()) {
            ROS_WARN("[智能颜色检测] 主阈值未检测到颜色，尝试使用宽松阈值");
            
            if (red_area > min_pixel_threshold && red_ratio > relaxed_threshold && red_area > max_area) {
                max_area = red_area;
                detected_color = "red";
                ROS_WARN("[智能颜色检测] 使用宽松阈值检测到红色盒子");
            } else if (yellow_area > min_pixel_threshold && yellow_ratio > relaxed_threshold && yellow_area > max_area) {
                max_area = yellow_area;
                detected_color = "yellow";
                ROS_WARN("[智能颜色检测] 使用宽松阈值检测到黄色盒子");
            } else if (blue_area > min_pixel_threshold && blue_ratio > relaxed_threshold && blue_area > max_area) {
                max_area = blue_area;
                detected_color = "blue";
                ROS_WARN("[智能颜色检测] 使用宽松阈值检测到蓝色盒子");
            }
        }
    }
    
    // 可视化显示检测结果
    visualize_smart_color_detection(img, gripper_roi, detected_color, 
                                   red_touches_edge, yellow_touches_edge, blue_touches_edge);
    
    return detected_color;
}

// 新增：航点4放置阶段爪子颜色检测函数 - 7cm×7cm正方形区域，以爪子中心白点为中心
bool detect_gripper_color_region(const Mat& img, const string& target_color) {
    if (img.empty()) {
        ROS_WARN("[爪子颜色检测] 图像为空");
        return false;
    }
    
    ROS_INFO("[爪子颜色检测] 开始检测爪子区域颜色，目标颜色: %s", target_color.c_str());
    
    // 修改：使用与航点3相同的稳定检测区域定位
    int gripper_center_x = img.cols / 2;  // 图像中心x坐标
    int gripper_center_y = img.rows * 2 / 3 + 120 + 120;  // 使用固定位置，图像下方2/3处，再往下移动四分之一个正方形长度（30像素，累计120像素），再下移一个Gripper Area边长（120像素）
    int roi_width = 150;  // 与航点3相同的检测区域宽度
    int roi_height = 240; // 高度扩大为原来的两倍，保持底边位置不变，只向上扩展
    
    // 计算检测区域边界 - 保持底边位置不变，只向上扩展
    int roi_x = max(0, gripper_center_x - roi_width / 2);
    int roi_y = max(0, gripper_center_y - roi_height);  // 从底边向上扩展，保持底边位置不变
    roi_width = min(roi_width, img.cols - roi_x);
    roi_height = min(roi_height, img.rows - roi_y);
    
    Rect gripper_roi(roi_x, roi_y, roi_width, roi_height);
    
    ROS_INFO("[爪子颜色检测] 检测区域: 中心(%d,%d), 大小%dx%d像素", 
             gripper_center_x, gripper_center_y, roi_width, roi_height);
    
    // 使用智能颜色检测函数
    string detected_color = detect_gripper_color_smart(img);
    
    // 可视化显示检测结果
    visualize_gripper_color_detection(img, gripper_roi, detected_color);
    
    // 判断检测结果是否与目标颜色匹配
    if (!detected_color.empty() && detected_color == target_color) {
        ROS_INFO("[爪子颜色检测] 智能检测成功，检测到目标颜色 %s", target_color.c_str());
        return true;
    } else {
        ROS_WARN("[爪子颜色检测] 智能检测失败或颜色不匹配，检测到: %s，目标: %s", 
                 detected_color.c_str(), target_color.c_str());
        
        // 如果智能检测失败，使用传统方法作为备用
        Mat roi_img = img(gripper_roi);
        
        // 转换到HSV颜色空间
        Mat hsv;
        cvtColor(roi_img, hsv, COLOR_BGR2HSV);
        
        // 检测目标颜色
        Mat color_mask;
        if (target_color == "red") {
            Mat red1, red2;
            inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), red1);
            inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), red2);
            color_mask = red1 | red2;
        } else if (target_color == "blue") {
            inRange(hsv, Scalar(100, 120, 70), Scalar(130, 255, 255), color_mask);
        } else if (target_color == "yellow") {
            inRange(hsv, Scalar(20, 100, 100), Scalar(30, 255, 255), color_mask);
        } else {
            ROS_WARN("[爪子颜色检测] 不支持的颜色: %s", target_color.c_str());
            return false;
        }
        
        // 形态学操作去除噪声
        Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
        morphologyEx(color_mask, color_mask, MORPH_CLOSE, kernel);
        
        // 计算颜色区域面积
        int color_area = countNonZero(color_mask);
        
        ROS_INFO("[爪子颜色检测] 传统方法 - %s颜色区域面积: %d 像素", target_color.c_str(), color_area);
        
        // 设置面积阈值（基于7cm×7cm区域大小调整）
        int area_threshold = 500;  // 适当阈值，避免小噪点干扰
        
        if (color_area > area_threshold) {
            ROS_INFO("[爪子颜色检测] 传统方法检测到目标颜色 %s，面积 %d > 阈值 %d", 
                     target_color.c_str(), color_area, area_threshold);
            return true;
        } else {
            ROS_WARN("[爪子颜色检测] 传统方法未检测到足够的目标颜色 %s，面积 %d <= 阈值 %d", 
                     target_color.c_str(), color_area, area_threshold);
            return false;
        }
    }
}

// 新增：方块抓取验证函数 - 改进检测逻辑
// 新增：智能抓取成功判定函数 - 检测第5行第3列区域是否有红黄蓝色像素块
bool check_grab_success_by_grid(const Mat& img) {
    if (img.empty()) {
        ROS_WARN("[抓取成功判定] 图像为空");
        return false;
    }
    
    // 创建可视化图像副本
    Mat display_img = img.clone();
    
    // 将图像按3行3列划分（根据用户建议修改）
    int grid_rows = 3;
    int grid_cols = 3;
    int cell_width = img.cols / grid_cols;
    int cell_height = img.rows / grid_rows;
    
    // 检测第3行第2列区域（索引从1开始）
    int target_row = 2; // 第3行（索引2）
    int target_col = 1; // 第2列（索引1）
    
    // 计算目标区域坐标
    int x = target_col * cell_width;
    int y = target_row * cell_height;
    int width = cell_width;
    int height = cell_height;
    
    // 确保区域在图像范围内
    Rect target_roi(x, y, width, height);
    target_roi = target_roi & Rect(0, 0, img.cols, img.rows);
    
    if (target_roi.width <= 0 || target_roi.height <= 0) {
        ROS_WARN("[抓取成功判定] 目标区域无效");
        return false;
    }
    
    ROS_INFO("[抓取成功判定] 检测第3行第2列区域: (%d,%d,%d,%d)", 
             target_roi.x, target_roi.y, target_roi.width, target_roi.height);
    
    // 可视化：绘制3x3网格和检测区域
    // 绘制网格线
    for (int i = 1; i < grid_rows; i++) {
        line(display_img, Point(0, i * cell_height), Point(img.cols, i * cell_height), Scalar(100, 100, 100), 1);
    }
    for (int j = 1; j < grid_cols; j++) {
        line(display_img, Point(j * cell_width, 0), Point(j * cell_width, img.rows), Scalar(100, 100, 100), 1);
    }
    
    // 绘制检测区域（第3行第2列）
    rectangle(display_img, target_roi, Scalar(0, 255, 255), 2); // 黄色框
    putText(display_img, "Detection Area", Point(target_roi.x + 5, target_roi.y + 20), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
    
    // 转换到HSV颜色空间
    Mat hsv;
    cvtColor(img(target_roi), hsv, COLOR_BGR2HSV);
    
    // 检测红黄蓝色像素块（使用与detect_color_box相同的算法）
    Mat red1, red2, red_mask, yellow_mask, blue_mask;
    
    // 检测红色（与detect_color_box相同的范围）
    inRange(hsv, Scalar(0, 120, 70), Scalar(10, 255, 255), red1);
    inRange(hsv, Scalar(170, 120, 70), Scalar(180, 255, 255), red2);
    red_mask = red1 | red2;
    
    // 检测蓝色（与detect_color_box相同的范围）
    inRange(hsv, Scalar(100, 120, 70), Scalar(130, 255, 255), blue_mask);
    
    // 检测黄色（与detect_color_box相同的范围）
    inRange(hsv, Scalar(20, 100, 100), Scalar(30, 255, 255), yellow_mask);
    
    // 形态学操作去除噪声（与detect_color_box相同）
    Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
    morphologyEx(red_mask, red_mask, MORPH_CLOSE, kernel);
    morphologyEx(blue_mask, blue_mask, MORPH_CLOSE, kernel);
    morphologyEx(yellow_mask, yellow_mask, MORPH_CLOSE, kernel);
    
    // 计算各颜色区域的面积（与detect_color_box相同的方法）
    vector<vector<Point>> red_contours, blue_contours, yellow_contours;
    findContours(red_mask, red_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(blue_mask, blue_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    findContours(yellow_mask, yellow_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    double red_area = 0, blue_area = 0, yellow_area = 0;
    for (const auto& c : red_contours) {
        red_area += contourArea(c);
    }
    for (const auto& c : blue_contours) {
        blue_area += contourArea(c);
    }
    for (const auto& c : yellow_contours) {
        yellow_area += contourArea(c);
    }
    
    ROS_INFO("[抓取成功判定] 第3行第2列区域检测结果 - 红色面积: %.1f, 蓝色面积: %.1f, 黄色面积: %.1f", 
             red_area, blue_area, yellow_area);
    
    // 判定条件：使用与detect_color_box相同的面积阈值逻辑
    double threshold = 100; // 基于区域大小调整的合理阈值
    
    bool grab_success = false;
    string result_text = "Grab Failed";
    Scalar result_color = Scalar(0, 0, 255); // 红色
    
    if (red_area > threshold || blue_area > threshold || yellow_area > threshold) {
        ROS_INFO("[抓取成功判定] 检测到颜色区域，抓取成功");
        grab_success = true;
        result_text = "Grab Success";
        result_color = Scalar(0, 255, 0); // 绿色
    } else {
        ROS_WARN("[抓取成功判定] 未检测到足够颜色区域");
    }
    
    // 可视化：显示检测结果和颜色面积信息
    putText(display_img, result_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, result_color, 2);
    
    // 显示各颜色面积信息（使用英文避免中文显示问号问题）
    string red_info = "Red area: " + to_string((int)red_area);
    string blue_info = "Blue area: " + to_string((int)blue_area);
    string yellow_info = "Yellow area: " + to_string((int)yellow_area);
    
    putText(display_img, red_info, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1);
    putText(display_img, blue_info, Point(10, 85), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 1);
    putText(display_img, yellow_info, Point(10, 110), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
    
    // 更新原图像
    const_cast<Mat&>(img) = display_img;
    
    return grab_success;
}

// 验证盒子是否被抓取并检测盒子颜色（优化版本）
bool verify_box_capture(const Mat& img) {
    if (img.empty()) {
        ROS_WARN("[抓取验证] 图像为空");
        return false;
    }
    
    // 创建可视化图像副本
    Mat display_img = img.clone();
    
    // 重新检测盒子颜色，确保box_color变量正确设置
    ROS_INFO("[抓取验证] 开始检测抓取到的盒子颜色");
    
    // 优化：抓取盒子成功后，机械爪应该在图像中的特定位置
    // 根据用户反馈，检测区域需要再往下移动四分之一个正方形长度
    int gripper_center_x = img.cols / 2;
    int gripper_center_y = img.rows * 2 / 3 + 120;  // 固定在图像下方2/3位置，再往下移动四分之一个正方形长度（30像素，累计120像素）
    
    // 定义检测区域大小（根据盒子大小调整）
    int roi_width = 150;   // 增大检测区域，确保覆盖整个盒子
    int roi_height = 240;  // 高度扩大为原来的两倍，保持底边位置不变，只向上扩展
    
    // 计算检测区域边界 - 保持底边位置不变，只向上扩展
    int roi_x = max(0, gripper_center_x - roi_width / 2);
    int roi_y = max(0, gripper_center_y - roi_height);  // 从底边向上扩展，保持底边位置不变
    roi_width = min(roi_width, img.cols - roi_x);
    roi_height = min(roi_height, img.rows - roi_y);
    
    Rect gripper_roi_rect(roi_x, roi_y, roi_width, roi_height);
    
    Mat roi_img = img(gripper_roi_rect);
    
    ROS_INFO("[抓取验证] 检测夹爪区域: (%d,%d) %dx%d", 
             gripper_roi_rect.x, gripper_roi_rect.y, gripper_roi_rect.width, gripper_roi_rect.height);
    
    // 可视化：绘制夹爪检测区域
    rectangle(display_img, gripper_roi_rect, Scalar(0, 255, 0), 2); // 绿色框
    putText(display_img, "Gripper Detection Area", Point(gripper_roi_rect.x + 5, gripper_roi_rect.y - 10), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    
    // 转换到HSV颜色空间
    Mat img_hsv;
    cvtColor(roi_img, img_hsv, COLOR_BGR2HSV);
    
    // 优化颜色范围定义（更精确的颜色阈值）
    // 红色范围（考虑红色在HSV空间中的两个区域）
    Mat red_mask1, red_mask2, red_mask;
    inRange(img_hsv, Scalar(0, 100, 80), Scalar(8, 255, 255), red_mask1);    // 更精确的红色范围
    inRange(img_hsv, Scalar(172, 100, 80), Scalar(180, 255, 255), red_mask2);
    red_mask = red_mask1 | red_mask2;
    
    // 黄色范围（调整饱和度和亮度阈值）
    Mat yellow_mask;
    inRange(img_hsv, Scalar(22, 120, 100), Scalar(32, 255, 255), yellow_mask);
    
    // 蓝色范围（调整色调和饱和度）
    Mat blue_mask;
    inRange(img_hsv, Scalar(105, 100, 80), Scalar(125, 255, 255), blue_mask);
    
    // 形态学操作去噪（使用更合适的核大小）
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(red_mask, red_mask, MORPH_OPEN, kernel);  // 先开运算去噪
    morphologyEx(red_mask, red_mask, MORPH_CLOSE, kernel); // 再闭运算填充
    
    morphologyEx(yellow_mask, yellow_mask, MORPH_OPEN, kernel);
    morphologyEx(yellow_mask, yellow_mask, MORPH_CLOSE, kernel);
    
    morphologyEx(blue_mask, blue_mask, MORPH_OPEN, kernel);
    morphologyEx(blue_mask, blue_mask, MORPH_CLOSE, kernel);
    
    // 计算各颜色区域的面积
    int red_area = countNonZero(red_mask);
    int yellow_area = countNonZero(yellow_mask);
    int blue_area = countNonZero(blue_mask);
    
    // 计算检测区域总面积
    int total_area = roi_img.rows * roi_img.cols;
    
    // 计算颜色区域占比
    double red_ratio = (double)red_area / total_area;
    double yellow_ratio = (double)yellow_area / total_area;
    double blue_ratio = (double)blue_area / total_area;
    
    ROS_INFO("[抓取验证] 夹爪区域颜色面积 - 红:%d(%.2f%%), 黄:%d(%.2f%%), 蓝:%d(%.2f%%)", 
             red_area, red_ratio * 100, yellow_area, yellow_ratio * 100, blue_area, blue_ratio * 100);
    
    // 优化：抓取盒子成功后，盒子颜色应该更明显，使用更宽松的阈值
    // 根据实际测试结果调整阈值：红色区域719像素，占比3.99%应该被识别
    // 调整阈值：主阈值1.5%，宽松阈值0.8%（因抓取角度问题导致露出面积小）
    double area_ratio_threshold = 0.015;  // 降低到1.5%的区域占比阈值（抓取后盒子更明显）
    int min_pixel_threshold = 100;        // 提高最小像素数量阈值，避免噪声干扰
    
    // 多条件判断：既要满足面积占比，也要满足最小像素数量
    bool red_detected = (red_ratio > area_ratio_threshold) && (red_area > min_pixel_threshold);
    bool yellow_detected = (yellow_ratio > area_ratio_threshold) && (yellow_area > min_pixel_threshold);
    bool blue_detected = (blue_ratio > area_ratio_threshold) && (blue_area > min_pixel_threshold);
    
    // 检查是否检测到多个颜色（可能干扰）
    int detected_colors = 0;
    if (red_detected) detected_colors++;
    if (yellow_detected) detected_colors++;
    if (blue_detected) detected_colors++;
    
    string detected_color = "";
    
    if (detected_colors > 1) {
        ROS_WARN("[抓取验证] 检测到多个颜色，可能存在干扰");
        // 优化：抓取盒子后，盒子颜色应该占主导地位，选择面积最大的颜色
        if (red_area > yellow_area && red_area > blue_area) {
            detected_color = "red";
            ROS_INFO("[抓取验证] 选择红色（面积最大：%d像素）", red_area);
        } else if (yellow_area > red_area && yellow_area > blue_area) {
            detected_color = "yellow";
            ROS_INFO("[抓取验证] 选择黄色（面积最大：%d像素）", yellow_area);
        } else {
            detected_color = "blue";
            ROS_INFO("[抓取验证] 选择蓝色（面积最大：%d像素）", blue_area);
        }
    } else if (detected_colors == 1) {
        // 单一颜色检测
        if (red_detected) {
            detected_color = "red";
            ROS_INFO("[抓取验证] 检测到红色盒子（面积：%d像素）", red_area);
        } else if (yellow_detected) {
            detected_color = "yellow";
            ROS_INFO("[抓取验证] 检测到黄色盒子（面积：%d像素）", yellow_area);
        } else if (blue_detected) {
            detected_color = "blue";
            ROS_INFO("[抓取验证] 检测到蓝色盒子（面积：%d像素）", blue_area);
        }
    } else {
        // 如果未检测到有效颜色，尝试使用更宽松的阈值
        double relaxed_threshold = 0.008;  // 降低到0.8%的宽松阈值（确保抓取角度不佳时也能识别）
        int relaxed_pixel_threshold = 50; // 适当降低像素阈值
        
        if (red_ratio > relaxed_threshold && red_area > relaxed_pixel_threshold) {
            detected_color = "red";
            ROS_WARN("[抓取验证] 使用宽松阈值检测到红色盒子（面积：%d像素）", red_area);
        } else if (yellow_ratio > relaxed_threshold && yellow_area > relaxed_pixel_threshold) {
            detected_color = "yellow";
            ROS_WARN("[抓取验证] 使用宽松阈值检测到黄色盒子（面积：%d像素）", yellow_area);
        } else if (blue_ratio > relaxed_threshold && blue_area > relaxed_pixel_threshold) {
            detected_color = "blue";
            ROS_WARN("[抓取验证] 使用宽松阈值检测到蓝色盒子（面积：%d像素）", blue_area);
        }
    }
    
    // 更新盒子颜色
    string result_text = "Grab Verification";
    Scalar result_color = Scalar(0, 255, 0); // 绿色
    
    if (!detected_color.empty()) {
        box_color = detected_color;
        ROS_INFO("[抓取验证] 抓取成功，检测到%s盒子", box_color.c_str());
        result_text = "Detected " + detected_color + " box";
    } else {
        // 如果没有检测到明显的颜色，保持原有颜色或使用默认值
        if (box_color.empty()) {
            ROS_WARN("[抓取验证] 无法确定盒子颜色，使用默认值");
            result_text = "Cannot determine box color";
            result_color = Scalar(0, 255, 255); // 黄色
            // 可以根据抓取位置或其他信息推断颜色
            // 这里暂时保持原有逻辑
        } else {
            ROS_INFO("[抓取验证] 抓取成功，使用原有颜色: %s", box_color.c_str());
            result_text = "Using original color: " + box_color;
        }
    }
    
    // 可视化：显示检测结果和颜色占比信息
    putText(display_img, result_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2);
    
    // 显示各颜色占比信息（使用英文避免中文显示问号问题）
    string red_info = "Red ratio: " + to_string((int)(red_ratio * 100)) + "%";
    string yellow_info = "Yellow ratio: " + to_string((int)(yellow_ratio * 100)) + "%";
    string blue_info = "Blue ratio: " + to_string((int)(blue_ratio * 100)) + "%";
    
    putText(display_img, red_info, Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 1);
    putText(display_img, yellow_info, Point(10, 85), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 1);
    putText(display_img, blue_info, Point(10, 110), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 1);
    
    // 更新原图像
    const_cast<Mat&>(img) = display_img;
    
    // 确保可视化结果被显示（窗口标题改为英文）
    imshow("Task Vision", img);
    waitKey(1); // 1ms延迟，实现流畅的连续帧显示
    
    return true;
}

// 全局变量存储识别到的数字
int detected_number = 0;

// 数字区域提取函数 - 优化版本，提高红色方框检测准确性
Rect extract_digit_region(const Mat& img) {
    if (img.empty()) {
        return Rect(0, 0, 0, 0);
    }
    
    try {
        Mat img_hsv;
        cvtColor(img, img_hsv, COLOR_BGR2HSV);
        
        // 检测红色区域（红色方框）- 调整HSV范围提高检测准确性
        Mat red_mask1, red_mask2, red_mask;
        inRange(img_hsv, Scalar(0, 50, 50), Scalar(10, 255, 255), red_mask1);
        inRange(img_hsv, Scalar(170, 50, 50), Scalar(180, 255, 255), red_mask2);
        red_mask = red_mask1 | red_mask2;
        
        // 形态学操作去除噪点 - 调整参数提高检测稳定性
        Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
        morphologyEx(red_mask, red_mask, MORPH_CLOSE, kernel);
        morphologyEx(red_mask, red_mask, MORPH_OPEN, kernel);
        
        // 查找红色方框轮廓
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(red_mask, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        
        if (!contours.empty()) {
            // 找到最大轮廓（红色方框）
            double max_area = 0;
            int max_idx = -1;
            for (int i = 0; i < contours.size(); i++) {
                double area = contourArea(contours[i]);
                if (area > max_area && area > 300) { // 降低面积阈值，提高检测灵敏度
                    max_area = area;
                    max_idx = i;
                }
            }
            
            if (max_idx >= 0) {
                // 获取红色方框的边界框
                Rect red_box = boundingRect(contours[max_idx]);
                
                // 确保红色方框足够大且比例合理
                if (red_box.width >= 50 && red_box.height >= 50 && 
                    red_box.width <= img.cols * 0.8 && red_box.height <= img.rows * 0.8) {
                    
                    // 从红色方框边缘向内裁剪10%（上、下、左、右各裁剪10%）
                    int margin_x = red_box.width * 0.10;  // 水平方向裁剪10%
                    int margin_y = red_box.height * 0.10; // 垂直方向裁剪10%
                    
                    int x = red_box.x + margin_x;
                    int y = red_box.y + margin_y;
                    int width = red_box.width - 2 * margin_x;
                    int height = red_box.height - 2 * margin_y;
                    
                    Rect digit_roi(x, y, width, height);
                    
                    // 确保区域在图像范围内
                    digit_roi = digit_roi & Rect(0, 0, img.cols, img.rows);
                    
                    if (digit_roi.width >= 40 && digit_roi.height >= 40) {
                        ROS_INFO("[Digit Region Extraction] Detected red box, extracting digit region with 10%% inward crop: %dx%d (original box: %dx%d)", 
                                digit_roi.width, digit_roi.height, red_box.width, red_box.height);
                        return digit_roi;
                    }
                }
            }
        }
        
        // 如果没有检测到红色方框，返回图像中心区域（调整为100x100以匹配模板尺寸）
        int center_x = max(0, (img.cols - 100) / 2);
        int center_y = max(0, (img.rows - 100) / 2);
        
        ROS_WARN("[Digit Region Extraction] No red box detected, using center region");
        return Rect(center_x, center_y, 100, 100);
        
    } catch (const cv::Exception& e) {
        ROS_ERROR("[Digit Region Extraction] Processing failed: %s", e.what());
        return Rect(0, 0, img.cols, img.rows);
    }
}

// 数字模板匹配函数 - 简化版本，提高识别稳定性
bool match_digit_template(const Mat& img, const Mat& template_img, double& score) {
    if (img.empty() || template_img.empty()) {
        return false;
    }
    
    try {
        // 直接使用输入图像，不进行额外的区域提取
        Mat img_processed = img.clone();
        Mat template_processed = template_img.clone();
        
        // 统一调整到标准100x100尺寸，确保与模板加载时尺寸一致
        resize(img_processed, img_processed, Size(100, 100));
        resize(template_processed, template_processed, Size(100, 100));
        
        // 转换为灰度图
        Mat img_gray, template_gray;
        cvtColor(img_processed, img_gray, COLOR_BGR2GRAY);
        cvtColor(template_processed, template_gray, COLOR_BGR2GRAY);
        
        // 简化预处理：仅使用高斯模糊去噪
        GaussianBlur(img_gray, img_gray, Size(3, 3), 0);
        GaussianBlur(template_gray, template_gray, Size(3, 3), 0);
        
        // 使用单一匹配方法：TM_CCOEFF_NORMED（最稳定）
        Mat result;
        matchTemplate(img_gray, template_gray, result, TM_CCOEFF_NORMED);
        
        double min_val, max_val;
        Point min_loc, max_loc;
        minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);
        
        score = max_val;
        
        ROS_INFO("[Digit Matching] Template score: %.3f", score);
        
        return true;
    } catch (const cv::Exception& e) {
        ROS_ERROR("[Digit Recognition] Template matching failed: %s", e.what());
        return false;
    }
}

// 加载数字模板
vector<Mat> load_digit_templates() {
    vector<Mat> templates;
    string pkg_path = ros::package::getPath("mobile_pkg");
    string shuzi_dir = pkg_path + "/shuzi";
    
    // 检查shuzi目录是否存在
    DIR* dir = opendir(shuzi_dir.c_str());
    if (!dir) {
        ROS_WARN("[数字识别] 无法打开 shuzi 目录: %s", shuzi_dir.c_str());
        return templates;
    }
    
    // 支持的图片格式
    vector<string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp"};
    
    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        if (entry->d_type == DT_REG) {
            string filename = entry->d_name;
            
            // 检查是否是图片文件
            bool is_image = false;
            for (const auto& ext : image_extensions) {
                if (filename.length() > ext.length() && 
                    filename.substr(filename.length() - ext.length()) == ext) {
                    is_image = true;
                    break;
                }
            }
            
            if (is_image) {
                string full_path = shuzi_dir + "/" + filename;
                Mat template_img = imread(full_path, IMREAD_COLOR);
                if (!template_img.empty()) {
                    // 调整模板大小到与摄像头中间区域相同的尺寸（100x100）
                    resize(template_img, template_img, Size(100, 100));
                    templates.push_back(template_img);
                    ROS_INFO("[数字识别] 加载数字模板: %s (尺寸: %dx%d)", filename.c_str(), template_img.cols, template_img.rows);
                } else {
                    ROS_ERROR("[数字识别] 无法加载模板文件: %s", full_path.c_str());
                }
            }
        }
    }
    closedir(dir);
    
    // 使用shuzi文件夹中的模板文件，不进行数量检查
    
    return templates;
}

// 轮廓识别数字算法 - 基于宽高比识别数字1和2
bool detect_number_by_contour(const Mat& img, int& detected_num, double& contour_score) {
    if (img.empty()) {
        ROS_ERROR("[轮廓识别] 图像为空");
        return false;
    }
    
    // 提取数字区域（从红色方框）
    Rect digit_region = extract_digit_region(img);
    Mat number_region = img(digit_region);
    
    // 转换为灰度图
    Mat gray;
    cvtColor(number_region, gray, COLOR_BGR2GRAY);
    
    // 二值化
    Mat binary;
    threshold(gray, binary, 128, 255, THRESH_BINARY);
    
    // 查找数字轮廓
    vector<vector<Point>> num_contours;
    findContours(binary, num_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    if (num_contours.empty()) {
        ROS_WARN("[轮廓识别] 未检测到数字轮廓");
        return false;
    }
    
    // 找到最大的数字轮廓
    Rect num_rect;
    for (const auto& contour : num_contours) {
        Rect rect = boundingRect(contour);
        if (rect.area() > num_rect.area()) {
            num_rect = rect;
        }
    }
    
    // 根据宽高比判断数字
    double width_ratio = (double)num_rect.width / num_rect.height;
    ROS_INFO("[轮廓识别] 数字轮廓宽度: %d, 高度: %d, 宽高比: %.2f", 
             num_rect.width, num_rect.height, width_ratio);
    
    // 计算轮廓识别得分（基于轮廓面积和宽高比的置信度）
    double area_confidence = min(1.0, (double)num_rect.area() / 1000.0); // 面积越大越可信
    double ratio_confidence = 1.0 - abs(width_ratio - 0.5) / 0.5; // 宽高比越接近0.5越可信
    contour_score = (area_confidence + ratio_confidence) / 2.0;
    
    if (width_ratio < 0.5) {
        detected_num = 1;
        ROS_INFO("[轮廓识别] 识别结果: 数字1, 置信度: %.2f", contour_score);
    } else {
        detected_num = 2;
        ROS_INFO("[轮廓识别] 识别结果: 数字2, 置信度: %.2f", contour_score);
    }
    
    return true;
}

void detect_num(const Mat& img) {
    // 清空之前的检测结果
    num_ok = false;
    detected_number = 0;
    
    // 创建可视化图像副本
    Mat display_img = img.clone();
    
    // 加载数字模板
    static vector<Mat> digit_templates = load_digit_templates();
    if (digit_templates.empty()) {
        ROS_ERROR("[数字识别] 无法加载数字模板");
        return;
    }
    
    // 提取数字区域（从红色方框或图像中心）
    Rect digit_region = extract_digit_region(img);
    Mat digit_roi = img(digit_region);
    
    // 显示裁剪前后的方框对比
    Mat before_crop = img.clone();
    rectangle(before_crop, digit_region, Scalar(0, 255, 0), 3); // 绿色框表示原始检测区域
    putText(before_crop, "Before Crop", Point(digit_region.x + 10, digit_region.y + 30), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 0), 2);
    
    Mat after_crop = digit_roi.clone();
    if (after_crop.cols != 100 || after_crop.rows != 100) {
        resize(after_crop, after_crop, Size(100, 100));
    }
    rectangle(after_crop, Rect(0, 0, after_crop.cols, after_crop.rows), Scalar(255, 0, 0), 3); // 蓝色框表示裁剪后区域
    putText(after_crop, "After Crop", Point(10, 30), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
    
    // 在主显示图像上绘制数字检测区域
    rectangle(display_img, digit_region, Scalar(0, 255, 0), 2); // 绿色框表示数字检测区域
    putText(display_img, "Digit Detection Area", Point(digit_region.x + 10, digit_region.y + 30), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
    
    // 将提取的数字区域缩放到标准尺寸（100x100），确保与模板尺寸一致
    if (digit_roi.cols != 100 || digit_roi.rows != 100) {
        resize(digit_roi, digit_roi, Size(100, 100));
        ROS_INFO("[数字识别] 缩放数字区域到标准尺寸: %dx%d -> 100x100", digit_region.width, digit_region.height);
    }
    
    // 裁剪后区域边框显示（绿色边框）
    if (digit_region.width > 0 && digit_region.height > 0) {
        rectangle(display_img, digit_region, Scalar(0, 255, 0), 3); // 绿色边框，线宽3
        putText(display_img, "After Crop", Point(digit_region.x, digit_region.y - 10), 
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
    }
    
    // 使用两种算法进行数字识别，设置50%权重计分机制
    
    // 算法1：模板匹配算法
    double template_best_score = 0.0;
    int template_best_digit = 0;
    vector<double> template_scores;
    
    for (int i = 0; i < digit_templates.size(); i++) {
        double score;
        if (match_digit_template(digit_roi, digit_templates[i], score)) {
            template_scores.push_back(score);
            ROS_INFO("[模板匹配] 模板 %d 匹配得分: %.3f", i + 1, score);
            if (score > template_best_score) {
                template_best_score = score;
                template_best_digit = i + 1;
            }
        }
    }
    
    // 如果只有一个模板，直接使用该得分
    if (template_scores.size() == 1) {
        template_best_score = template_scores[0];
        template_best_digit = 1;
    }
    
    // 算法2：轮廓识别算法
    int contour_detected_num = 0;
    double contour_score = 0.0;
    bool contour_success = detect_number_by_contour(img, contour_detected_num, contour_score);
    
    ROS_INFO("[算法对比] 模板匹配得分: %.3f (数字%d), 轮廓识别得分: %.3f (数字%d)", 
             template_best_score, template_best_digit, contour_score, contour_detected_num);
    
    // 50%权重计分机制
    double final_score = 0.0;
    int final_digit = 0;
    
    if (contour_success && template_best_score > 0) {
        // 两种算法都有效，使用50%权重
        final_score = (template_best_score * 0.5) + (contour_score * 0.5);
        
        // 如果两种算法识别结果一致，使用该结果
        if (template_best_digit == contour_detected_num) {
            final_digit = template_best_digit;
            ROS_INFO("[权重计分] 算法结果一致，最终数字: %d，综合得分: %.3f", final_digit, final_score);
        } else {
            // 算法结果不一致，选择得分更高的算法
            if (template_best_score >= contour_score) {
                final_digit = template_best_digit;
                ROS_INFO("[权重计分] 算法结果不一致，选择模板匹配结果: %d，综合得分: %.3f", final_digit, final_score);
            } else {
                final_digit = contour_detected_num;
                ROS_INFO("[权重计分] 算法结果不一致，选择轮廓识别结果: %d，综合得分: %.3f", final_digit, final_score);
            }
        }
    } else if (contour_success) {
        // 只有轮廓识别有效
        final_score = contour_score;
        final_digit = contour_detected_num;
        ROS_INFO("[权重计分] 仅轮廓识别有效，最终数字: %d，得分: %.3f", final_digit, final_score);
    } else if (template_best_score > 0) {
        // 只有模板匹配有效
        final_score = template_best_score;
        final_digit = template_best_digit;
        ROS_INFO("[权重计分] 仅模板匹配有效，最终数字: %d，得分: %.3f", final_digit, final_score);
    }
    
    // 判断是否成功识别
    string result_text;
    Scalar result_color;
    bool is_confident = false;
    
    // 降低阈值要求，提高识别成功率
    if (final_score > 0.15) { // 降低阈值到0.15
        detected_number = final_digit;
        num_ok = true;
        is_confident = true;
        result_text = "数字识别成功: " + to_string(detected_number);
        result_color = Scalar(0, 255, 0); // 绿色表示成功
        ROS_INFO("[数字识别] 成功识别数字: %d，综合得分: %.3f", detected_number, final_score);
    } else {
        result_text = "数字识别失败";
        result_color = Scalar(0, 0, 255); // 红色表示失败
        ROS_WARN("[数字识别] 未检测到有效数字，综合得分: %.3f", final_score);
    }
    
    // 可视化：绘制检测结果
    putText(display_img, result_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, result_color, 2);
    
    // 显示综合得分信息
    putText(display_img, "Final Score: " + to_string(final_score).substr(0, 4), Point(10, 60), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    
    // 显示两种算法的得分对比
    putText(display_img, "Template Score: " + to_string(template_best_score).substr(0, 4), Point(10, 80), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 200, 100), 1);
    putText(display_img, "Contour Score: " + to_string(contour_score).substr(0, 4), Point(10, 100), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(100, 200, 255), 1);
    
    // 显示权重信息
    putText(display_img, "Weight: Template 50% + Contour 50%", Point(10, 120), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
    
    // 显示算法结果
    string algorithm_text = "Template: " + to_string(template_best_digit) + " | Contour: " + to_string(contour_detected_num);
    putText(display_img, algorithm_text, Point(10, 140), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(200, 200, 200), 1);
    
    // 更新原图像以显示可视化结果
    const_cast<Mat&>(img) = display_img;
}

void detect_xn(const Mat& img) {
    // 创建可视化图像副本
    Mat display_img = img.clone();
    
    string pkg_path = ros::package::getPath("mobile_pkg");
    string xn_path = pkg_path + "/muban/xn.jpg";
    Mat xn_template = imread(xn_path, IMREAD_COLOR);
    if (xn_template.empty()) {
        ROS_WARN("[xn识别] xn模板未加载");
        
        // 可视化：显示模板未加载信息
        putText(display_img, "XN Template Not Loaded", Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
        
        // 更新原图像以显示可视化结果
        const_cast<Mat&>(img) = display_img;
        return;
    }
    
    double score = template_match(img, xn_template);
    
    // 可视化：绘制检测结果
    string result_text;
    Scalar result_color;
    
    if (score > 0.6) {
        xn_ok = true;
        result_text = "XN Detected: YES";
        result_color = Scalar(0, 255, 0); // 绿色表示检测成功
        ROS_INFO("[识别] 检测到xn，匹配得分：%.2f", score);
    } else {
        result_text = "XN Detected: NO";
        result_color = Scalar(0, 0, 255); // 红色表示未检测到
        ROS_INFO("[识别] 未检测到xn，匹配得分：%.2f", score);
    }
    
    // 在图像顶部显示检测结果
    putText(display_img, result_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, result_color, 2);
    
    // 显示匹配得分信息
    putText(display_img, "Match Score: " + to_string(score).substr(0, 4), Point(10, 60), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    
    // 显示检测区域（图像中心区域）
    Rect detection_roi(img.cols/4, img.rows/4, img.cols/2, img.rows/2);
    rectangle(display_img, detection_roi, Scalar(255, 255, 0), 2); // 青色框表示检测区域
    putText(display_img, "XN Detection Area", Point(detection_roi.x + 10, detection_roi.y + 30), 
            FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 0), 1);
    
    // 更新原图像以显示可视化结果
    const_cast<Mat&>(img) = display_img;
}

// 改进的图像回调函数 - 添加图像缓存和连续帧显示
void img_callback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        current_image = cv_bridge::toCvCopy(msg, "bgr8")->image;
        has_new_image = true;
        last_image_time = ros::Time::now();
        
        if (current_step == DETECT_QD && !qd_ok) {
            detect_qd(current_image);
        }
        else if (current_step == DETECT_BOX && !box_ok) {
            detect_box(current_image);
            
            // 航点3颜色像素块检测显示控制
            if (show_color_pixels_detection && color_pixels_frame_counter < COLOR_PIXELS_MAX_FRAMES) {
                // 定义航点3的检测区域（图像中心区域）
                int img_center_x = current_image.cols / 2;
                int img_center_y = current_image.rows / 2;
                int roi_width = 400;  // 检测区域宽度
                int roi_height = 300; // 检测区域高度
                
                int roi_x = max(0, img_center_x - roi_width / 2);
                int roi_y = max(0, img_center_y - roi_height / 2);
                roi_width = min(roi_width, current_image.cols - roi_x);
                roi_height = min(roi_height, current_image.rows - roi_y);
                
                Rect detection_roi(roi_x, roi_y, roi_width, roi_height);
                
                // 绘制检测区域
                rectangle(current_image, detection_roi, Scalar(255, 0, 0), 3); // 蓝色框表示航点3检测区域
                
                // 添加文字标注
                putText(current_image, "航点3颜色像素块检测区域", Point(roi_x, roi_y - 15), 
                        FONT_HERSHEY_SIMPLEX, 0.7, Scalar(255, 0, 0), 2);
                putText(current_image, "ROI: (" + to_string(roi_x) + "," + to_string(roi_y) + ") - (" + 
                        to_string(roi_width) + "x" + to_string(roi_height) + ")", 
                        Point(10, 90), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
                
                // 显示剩余帧数
                string frame_info = "航点3检测帧数: " + to_string(color_pixels_frame_counter) + "/" + to_string(COLOR_PIXELS_MAX_FRAMES);
                putText(current_image, frame_info, Point(10, 120), 
                        FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
                
                // 增加帧计数器
                color_pixels_frame_counter++;
                
                // 检查是否达到最大帧数
                if (color_pixels_frame_counter >= COLOR_PIXELS_MAX_FRAMES) {
                    show_color_pixels_detection = false;
                    ROS_INFO("[航点3] 颜色像素块检测区域显示完成，共显示%d帧", COLOR_PIXELS_MAX_FRAMES);
                }
            }
        }
        else if (current_step == DETECT_NUM && !num_ok) {
            detect_num(current_image);
        }
        
        // 数字识别裁剪边框显示控制（独立于数字识别结果）
        if (current_step == DETECT_NUM && show_crop_borders && frame_counter < MAX_FRAMES) {
            // 获取红色方框位置（未裁剪区域）
            Rect red_box = extract_digit_region(current_image);
            
            // 绘制红色边框表示未裁剪区域
            rectangle(current_image, red_box, Scalar(0, 0, 255), 3); // 红色边框，线宽3
            putText(current_image, "Before Crop", Point(red_box.x, red_box.y - 10), 
                    FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
            
            // 显示剩余帧数信息
            string frame_info = "Crop Borders: " + to_string(MAX_FRAMES - frame_counter) + " frames left";
            putText(current_image, frame_info, Point(10, current_image.rows - 30), 
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
            
            // 增加帧计数器
            frame_counter++;
            
            // 检查是否达到最大帧数
            if (frame_counter >= MAX_FRAMES) {
                show_crop_borders = false;
                ROS_INFO("[数字识别] 裁剪边框显示完成，共显示%d帧", MAX_FRAMES);
            }
        }
        else if (current_step == DETECT_XN && !xn_ok) {
            detect_xn(current_image);
        }
        else if (current_step == DETECT_COLOR_BOX) {
            // 颜色方框检测在状态机中处理
        }
        
        // 连续帧显示：使用waitKey(1)实现流畅显示，同时响应键盘事件
        imshow("Task Vision", current_image);
        
        // 裁剪对比窗口在detect_num函数中显示，这里不需要重复显示
        
        waitKey(1); // 1ms延迟，实现流畅的连续帧显示
    } catch (const cv::Exception& e) {
        ROS_ERROR("[视觉] OpenCV处理失败: %s", e.what());
    } catch (...) {
        ROS_ERROR("[视觉] 处理失败");
    }
}

// 新增：等待新图像的函数
bool wait_for_new_image() {
    // 删除超时机制，一直等待直到获取到新图像
    while (ros::ok()) {
        if (has_new_image) {
            has_new_image = false;
            return true;
        }
        ros::Duration(0.1).sleep();
        ros::spinOnce();
    }
    return false;
}

// 导航回调函数
void navi_callback(const std_msgs::String::ConstPtr& msg) {
    if (msg->data != "done") return;
    
    // 获取当前时间并计算从任务开始到现在的总时间
    ros::Time current_time = ros::Time::now();
    double elapsed_time = (current_time - start_time).toSec();
    // 修复：使用浮点数计算避免精度丢失
    int minutes = static_cast<int>(elapsed_time / 60);
    int seconds = static_cast<int>(elapsed_time - minutes * 60);  // 修复：使用浮点数计算秒数，避免%操作符的精度丢失
    
    switch (current_step) {
        case GOTO_WP1:
            ROS_INFO("[导航] 到达航点1 → 前往航点2");
            ROS_INFO("[计时器] 到达航点1时间: %d分%d秒", minutes, seconds);
            nav_msg.data = "2";
            waypoint_pub.publish(nav_msg);
            current_step = GOTO_WP2;
            break;
        case GOTO_WP2:
            ROS_INFO("[导航] 到达航点2 → 开始识别QD");
            ROS_INFO("[计时器] 到达航点2时间: %d分%d秒", minutes, seconds);
            // 重置人物识别检测区域显示状态
            current_step = DETECT_QD;
            break;
        case GOTO_WP3:
            ROS_INFO("[导航] 到达航点3 → 开始识别方块");
            ROS_INFO("[计时器] 到达航点3时间: %d分%d秒", minutes, seconds);
            current_step = DETECT_BOX;
            break;
        case GOTO_WP4:
            ROS_INFO("[导航] 到达航点4 → 开始检测颜色方框");
            ROS_INFO("[计时器] 到达航点4时间: %d分%d秒", minutes, seconds);
            current_step = DETECT_COLOR_BOX;
            break;
        case GOTO_WP5:
            ROS_INFO("[导航] 到达航点5 → 开始识别数字");
            ROS_INFO("[计时器] 到达航点5时间: %d分%d秒", minutes, seconds);
            // 启动边框显示状态
            show_crop_borders = true;
            frame_counter = 0;
            current_step = DETECT_NUM;
            ROS_INFO("[数字识别] 启动裁剪边框显示，将持续%d帧", MAX_FRAMES);
            break;
        case GOTO_WP1_FINAL:
            ROS_INFO("[导航] 返回航点1 → 任务完成");
            ROS_INFO("[计时器] 任务完成总时间: %d分%d秒", minutes, seconds);
            current_step = DONE;
            break;
        default:
            break;
    }
}

// 主函数
int main(int argc, char** argv) {
    // 设置字符编码，确保中文日志正确显示
    setlocale(LC_ALL, "");
    setlocale(LC_CTYPE, "zh_CN.UTF-8");
    
    init(argc, argv, "cruise_node");
    NodeHandle nh;

    // 设置导航速度参数 - 按照实际使用的WpbLocalPlanner设置
    // 线速度参数 - 设置为0.8m/s（巡航点间移动速度）
    nh.setParam("/move_base/WpbLocalPlanner/max_vel_trans", 3.3);  // WpbLocalPlanner最大平移速度设置为1.5m/s
    nh.setParam("/move_base/WpbLocalPlanner/max_vel_rot", 1.5);    // WpbLocalPlanner最大旋转速度设置为1.5rad/s
    
    // 加速度参数 - 设置为推荐范围最大值
    nh.setParam("/move_base/WpbLocalPlanner/acc_scale_trans", 2.5);  // WpbLocalPlanner平移加速度缩放因子设置为2.5
    nh.setParam("/move_base/WpbLocalPlanner/acc_scale_rot", 3.0);    // WpbLocalPlanner旋转加速度缩放因子设置为3.0
    
    // 目标容差参数 - 设置为推荐范围最大值
    nh.setParam("/move_base/WpbLocalPlanner/goal_dist_tolerance", 0.15); // WpbLocalPlanner目标距离容差设置为0.15m
    nh.setParam("/move_base/WpbLocalPlanner/goal_yaw_tolerance", 0.2);   // WpbLocalPlanner目标航向容差设置为0.2rad
    
    // 兼容性设置：保留其他规划器参数设置（虽然实际不使用）
    nh.setParam("/move_base/DWAPlannerROS/max_vel_x", 0.8);      // DWA规划器最大x方向速度设置为0.8m/s
    nh.setParam("/move_base/TebLocalPlannerROS/max_vel_x", 0.8);  // TEB规划器最大速度设置为0.8m/s
    nh.setParam("/move_base/DWAPlannerROS/max_vel_theta", 1.5);  // DWA规划器最大旋转速度设置为1.5rad/s
    nh.setParam("/move_base/TebLocalPlannerROS/max_vel_theta", 1.5); // TEB规划器最大旋转速度设置为1.5rad/s
    
    ROS_INFO("[系统] 导航速度参数已设置：巡航点间移动速度3.3m/s，任务执行速度0.5m/s");

    // 初始化TF监听器
    tf_listener = new tf::TransformListener();
    ROS_INFO("[系统] TF监听器初始化完成");

    waypoint_pub = nh.advertise<std_msgs::String>("/waterplus/navi_waypoint", 10);
    mani_pub = nh.advertise<sensor_msgs::JointState>("/wpb_mani/joint_ctrl", 10);
    gripper_pub = nh.advertise<std_msgs::Float64>("/wpb_mani/gripper_position_controller/command", 10);
    cmd_vel_pub = nh.advertise<geometry_msgs::Twist>("/cmd_vel", 10);
    img_sub = nh.subscribe("/depth/image_color", 1, img_callback);
    navi_sub = nh.subscribe("/waterplus/navi_result", 10, navi_callback);
    
    // 新增：wpb_mani抓取相关发布器和订阅器
    plane_height_pub = nh.advertise<std_msgs::Float64>("/wpb_mani/plane_height", 10);
    grab_box_pub = nh.advertise<geometry_msgs::Pose>("/wpb_mani/grab_box", 10);
    ros::Subscriber box_result_sub = nh.subscribe("/wpb_mani/boxes_3d", 10, BoxCoordCB);
    ros::Subscriber res_sub = nh.subscribe("/wpb_mani/grab_result", 10, GrabResultCB);
    
    // 新增：频率降低机制 - 订阅来自wpb_mani模块的cmd_vel指令
    ros::Subscriber wpb_mani_cmd_sub = nh.subscribe("/wpb_mani/cmd_vel", 10, wpbManiCmdVelCallback);
    
    // 新增：盒子和小车坐标同步输出订阅器
    robot_pose_sub = nh.subscribe("/wpb_mani/pose_diff", 10, RobotPoseCB);
    box_and_robot_sub = nh.subscribe("/wpb_mani/boxes_3d", 10, BoxAndRobotCoordSyncCB);
    



    ROS_INFO("[系统] 正在加载 data/ 目录中的 击倒目标 模板图片...");
    load_qd_template();

    ROS_INFO("[系统] 正在加载 data/ 目录中的角色模板...");
    load_character_templates();

    ROS_INFO("[初始化] 机械臂抬起，夹爪打开");
    send_mani(MANI_UP);
    set_gripper(GRIPPER_OPEN);
    namedWindow("Task Vision", WINDOW_AUTOSIZE);

    // 修改：按照航点1-2-3-4-5-1的顺序执行任务
    ROS_INFO("[系统] 任务模式启动，按照航点1-2-3-4-5-1的顺序执行任务");
    
    // 添加计时器：记录任务开始时间
    start_time = ros::Time::now();
    double start_time_sec = start_time.toSec();
    // 修复：使用浮点数计算避免精度丢失
    int start_minutes = static_cast<int>(start_time_sec / 60);
    int start_seconds = static_cast<int>(start_time_sec - start_minutes * 60);  // 修复：使用浮点数计算秒数
    ROS_INFO("[计时器] 任务开始时间: %d分%d秒", start_minutes, start_seconds);
    
    nav_msg.data = "1";
    waypoint_pub.publish(nav_msg);
    current_step = GOTO_WP1;

    Rate loop_rate(10);
    while (ok() && current_step != DONE) {
        switch (current_step) {
            case DETECT_QD:
                if (qd_ok) {
                    ROS_INFO("[任务] 任务目标识别完成 → 移动到 %s 侧", qd_side.c_str());
                    current_step = MOVE_TO_QD_SIDE;
                }
                break;

            case MOVE_TO_QD_SIDE:
                move_lateral(qd_side, 1.0, 0.5); // 优化：横向移动速度从0.5提升到0.7m/s，持续1秒
                ROS_INFO("[任务] 已横向对齐 → 准备前进靠近人物");
                current_step = MOVE_TO_QD_FRONT;
                break;

            case MOVE_TO_QD_FRONT:
                move_forward(1.0, 0.75); // 优化：前进速度从0.7改为0.75m/s，持续1秒
                ROS_INFO("[任务] 已前进到人物面前 → 执行击倒");
                current_step = KNOCK_QD;
                break;

            case KNOCK_QD: {
                ROS_INFO("[状态机] 执行击倒动作（使用异步机械臂控制）");
                
                // 优化：使用异步机械臂控制，减少等待时间
                static AsyncManipulator manipulator;
                
                // 第一步：异步发送击倒动作
                manipulator.send_mani_async(MANI_KNOCK);
                
                // 第二步：等待击倒动作完全完成（确保机械臂完整伸出）
                ROS_INFO("[机械臂] 等待击倒动作完全完成...");
                while (!manipulator.wait_for_completion(0.1)) {
                    ros::Duration(0.05).sleep();
                }
                ROS_INFO("[机械臂] 击倒动作完成");
                
                // 第三步：发送抬起动作
                manipulator.send_mani_async(MANI_UP);
                
                // 第四步：在机械臂抬起动作执行的同时，开始左转准备前往航点3
                ROS_INFO("[导航] 机械臂执行抬起动作的同时，开始左转前往航点3");
                
                // 优化：移除固定超时等待，改为基于角速度的动态左转
                // 左转90度（1.57弧度），角速度1.5弧度/秒，理论时间约1.05秒
                geometry_msgs::Twist twist;
                twist.angular.z = 1.5; // 角速度1.5弧度/秒
                
                // 记录开始时间
                ros::Time start_time = ros::Time::now();
                
                // 持续左转直到达到90度
                while (ros::ok()) {
                    cmd_vel_pub.publish(twist);
                    
                    // 计算已转动时间
                    ros::Duration elapsed = ros::Time::now() - start_time;
                    double elapsed_seconds = elapsed.toSec();
                    
                    // 检查是否达到90度（1.57弧度）
                    if (elapsed_seconds >= 1.05) {
                        ROS_INFO("[导航] 左转完成，转动时间：%.2f秒", elapsed_seconds);
                        break;
                    }
                    
                    // 短暂等待，避免过度占用CPU
                    ros::Duration(0.01).sleep();
                }
                
                // 停止左转
                twist.angular.z = 0;
                cmd_vel_pub.publish(twist);
                
                // 第五步：等待抬起动作完成
                ROS_INFO("[机械臂] 等待抬起动作完成...");
                while (!manipulator.wait_for_completion(0.1)) {
                    ros::Duration(0.05).sleep();
                }
                ROS_INFO("[机械臂] 抬起动作完成");
                
                ROS_INFO("[导航] 航点2击倒完成，前往航点3");
                nav_msg.data = "3";
                waypoint_pub.publish(nav_msg);
                current_step = GOTO_WP3;
                qd_ok = false;
                qd_side = "";
                // 重试计数器已删除，检测功能交给wpb_mani处理
                break;
            }

            case DETECT_BOX: {
                ROS_INFO("[任务] 航点3盒子抓取阶段（第%d次抓取）", box_count + 1);
                
                // 启用航点3颜色像素块检测显示控制
                show_color_pixels_detection = true;
                color_pixels_frame_counter = 0;
                ROS_INFO("[航点3] 启用颜色像素块检测区域显示，将显示%d帧", COLOR_PIXELS_MAX_FRAMES);
                
                // 修改：在航点3处不进行任何检测行为，直接前进1秒，速度0.27
                ROS_INFO("[移动] 到达航点3，前进1秒，速度0.27m/s");
                move_forward(1.0, 0.27); // 优化：前进速度从0.3改为0.27m/s
                
                current_step = GRAB_BOX;
                break;
            }

            case MOVE_TO_TABLE_FRONT: {
                ROS_INFO("[任务] 初始定位阶段 - 靠近桌子准备抓取");
                
                // 重置抓取相关状态
                grab_in_progress = false;
                grab_success = false;
                
                // 执行前进靠近目标区域
                ROS_INFO("[移动] 前进靠近目标区域，速度0.5m/s，持续1.0秒");
                move_forward(1.0, 0.25); // 优化：前进速度从0.25提升到0.5m/s
                
                // 重置y值监控系统
                reset_y_monitoring();
                
                current_step = GRAB_BOX;
                break;
            }

            case GRAB_BOX: {
                ROS_INFO("[抓取] 开始抓取方块（第%d个方块，第%d次抓取尝试）", box_count + 1, grab_attempt_count + 1);
                
                // 修改：前进1秒已完成，直接启动wpb_mani模块
                ROS_INFO("[抓取] 前进1秒已完成，启动wpb_mani模块，开始自动调整车身位置和抓取");
                
                // 重置抓取状态
                grab_success = false;
                grab_in_progress = false;
                grab_coord_sent = false;
                
                // 重置y值监控系统
                reset_y_monitoring();
                
                // 启动wpb_mani模块
                wpb_mani_active = true;
                
                // 发布平面高度触发检测
                std_msgs::Float64 plane_height_msg;
                plane_height_msg.data = 0.22; // 桌子高度
                plane_height_pub.publish(plane_height_msg);
                ROS_INFO("[抓取] 发布平面高度: %.2f米", plane_height_msg.data);
                
                // 优化：缩短等待wpb_mani模块检测盒子的时间
                ros::Duration(0.5).sleep();
                
                // 等待抓取结果，实现完整的y值监控系统
                ros::Time grab_start_time = ros::Time::now();
                bool large_change_triggered = false;
                bool stable_triggered = false;
                bool fine_triggered = false;
                int large_change_counter = 0;
                int stable_counter = 0;
                int fine_counter = 0;
                double last_y = 0.0;
                bool first_y_received = false;
                
                // 频率降低机制：将30Hz指令降低到5Hz
                int frequency_divider = 6; // 30Hz / 6 = 5Hz
                int loop_counter = 0;
                
                while (ros::ok() && !grab_success) {
                    ros::Duration(0.033).sleep(); // 控制频率30Hz
                    spinOnce();
                    
                    // 检查抓取结果回调（不受频率降低影响，必须每次检查）
                    if (grab_result_received) {
                        if (grab_success) {
                            ROS_INFO("[抓取] 收到抓取结果回调，抓取成功");
                            y_monitoring_enabled = false; // 禁用y值监控
                            break;
                        } else {
                            ROS_INFO("[抓取] 收到抓取结果回调，抓取失败，继续监控");
                            grab_result_received = false; // 重置回调标志，等待下一次回调
                        }
                    }
                    
                    // 删除抓取超时检查（用户要求删除抓取超时逻辑）
                    
                    // 频率降低逻辑：每2次循环只处理1次wpb_mani模块指令
                    loop_counter++;
                    if (loop_counter % frequency_divider != 0) {
                        // 跳过本次循环，不处理wpb_mani模块指令
                        continue;
                    }
                    
                    ROS_INFO("[频率降低] 处理wpb_mani模块指令 (5Hz)");
                    
                    // 获取当前y值
                    if (current_y != 0.0) {
                        if (!first_y_received) {
                            last_y = current_y;
                            first_y_received = true;
                            continue;
                        }
                        
                        // 计算y值变化
                        double delta_y = abs(current_y - last_y);
                        last_y = current_y;
                        
                        // Y值监控逻辑（追踪阶段）
                        if (y_monitoring_enabled && !large_change_triggered) {
                            // 逻辑A：大幅偏移处理
                            if (delta_y > 0.1) { // y_change_large = 0.1
                                large_change_counter++;
                                ROS_INFO("[y值监控] 追踪阶段：y值变化%.3f米，大幅偏移计数%d", delta_y, large_change_counter);
                                
                                if (large_change_counter >= 2) {
                                    ROS_INFO("[y值监控] 追踪阶段：累积%d次y值变化>0.1米，触发大幅偏移处理", large_change_counter);
                                    
                                    // 发布0速度指令（30Hz频率，持续0.5秒）
                                    for (int i = 0; i < 15; i++) {
                                        geometry_msgs::Twist stop_cmd;
                                        stop_cmd.linear.x = 0.0;
                                        stop_cmd.linear.y = 0.0;
                                        stop_cmd.linear.z = 0.0;
                                        stop_cmd.angular.x = 0.0;
                                        stop_cmd.angular.y = 0.0;
                                        stop_cmd.angular.z = 0.0;
                                        cmd_vel_pub.publish(stop_cmd);
                                        ros::Duration(0.033).sleep();
                                    }
                                    
                                    // 暂停/重启wpb_mani模块
                                    wpb_mani_active = false;
                                    ros::Duration(0.5).sleep();
                                    wpb_mani_active = true;
                                    
                                    large_change_triggered = true;
                                    ROS_INFO("[y值监控] 追踪阶段：大幅偏移处理完成，重启wpb_mani模块");
                                }
                            } else {
                                large_change_counter = 0; // 重置大幅偏移计数
                            }
                        }
                        
                        // 逻辑B：稳定判定
                        if (y_monitoring_enabled && large_change_triggered && !stable_triggered) {
                            if (delta_y < 0.01) { // y_change_small = 0.01
                                stable_counter++;
                                ROS_INFO("[y值监控] 追踪阶段：y值变化%.3f米，稳定计数%d", delta_y, stable_counter);
                                
                                if (stable_counter >= 8) { // stable_count_threshold = 8
                                    ROS_INFO("[y值监控] 追踪阶段：连续%d次y值变化<0.01米，判定稳定对准", stable_counter);
                                    stable_triggered = true;
                                }
                            } else {
                                stable_counter = 0; // 重置稳定计数
                            }
                        }
                        
                        // 精细对准与抓取
                        if (y_monitoring_enabled && stable_triggered && !fine_triggered) {
                            if (delta_y < 0.002) { // y_change_fine = 0.002
                                fine_counter++;
                                ROS_INFO("[y值监控] 精细对准：y值变化%.3f米，精细计数%d", delta_y, fine_counter);
                                
                                if (fine_counter >= 12) { // fine_aligned_count_threshold = 12
                                    ROS_INFO("[y值监控] 精细对准：连续%d次y值变化<0.002米，触发抓取", fine_counter);
                                    
                                    // 发布0速度指令
                                    geometry_msgs::Twist stop_cmd;
                                    stop_cmd.linear.x = 0.0;
                                    stop_cmd.linear.y = 0.0;
                                    stop_cmd.linear.z = 0.0;
                                    stop_cmd.angular.x = 0.0;
                                    stop_cmd.angular.y = 0.0;
                                    stop_cmd.angular.z = 0.0;
                                    cmd_vel_pub.publish(stop_cmd);
                                    
                                    // 第一步：张开夹爪准备抓取
                                    set_gripper(GRIPPER_OPEN); // 张开夹爪
                                    Duration(0.2).sleep(); // 优化：缩短等待夹爪张开时间
                                    
                                    // 第二步：伸出机械臂
                                    send_mani(MANI_DOWN); // 伸出机械臂
                                    Duration(0.2).sleep(); // 优化：缩短等待机械臂伸出时间
                                    
                                    // 第三步：车身控制将盒子移动到夹爪中央后闭合夹爪
                                    move_forward(1.5, 0.1); // final_approach_duration = 1.5秒
                                    set_gripper(GRIPPER_CLOSE); // 闭合夹爪
                                    Duration(0.2).sleep(); // 优化：缩短等待夹爪闭合时间
                                    
                                    // 标记抓取成功
                                    grab_success = true;
                                    fine_triggered = true;
                                    
                                    // 抓取成功后禁用y值监控
                                    y_monitoring_enabled = false;
                                    ROS_INFO("[y值监控] 抓取完成，禁用y值监控");
                                }
                            } else {
                                fine_counter = 0; // 重置精细计数
                            }
                        }
                    }
                }
                
                // 抓取成功处理
                if (grab_success) {
                    ROS_INFO("[抓取] 方块抓取成功！");
                    box_count++;
                    
                    // 抓取成功后禁用y值监控
                    y_monitoring_enabled = false;
                    ROS_INFO("[y值监控] 抓取成功，禁用y值监控");
                    
                    // 重置抓取尝试计数
                    grab_attempt_count = 0;
                    rotation_angle = 0;
                    
                    // 停止wpb_mani模块
                    wpb_mani_active = false;
                    
                    // 优化：缩短收回机械臂的等待时间
                    send_mani(MANI_UP);
                    ros::Duration(0.8).sleep();
                    
                    // 检测是否成功抓取
                    if (wait_for_new_image()) {
                        if (verify_box_capture(current_image)) {
                            ROS_INFO("[抓取] 确认机械臂持有%s盒子，前往航点4放置盒子", box_color.c_str());
                            nav_msg.data = "4";
                            waypoint_pub.publish(nav_msg);
                            current_step = GOTO_WP4;
                        } else {
                            ROS_WARN("[抓取] 抓取失败，重新尝试抓取");
                            grab_success = false;
                            box_count--; // 恢复盒子计数
                        }
                    } else {
                        ROS_WARN("[抓取] 图像更新超时，无法验证抓取结果");
                        // 即使图像更新超时，也尝试继续任务
                        ROS_INFO("[抓取] 继续前往航点4放置盒子");
                        nav_msg.data = "4";
                        waypoint_pub.publish(nav_msg);
                        current_step = GOTO_WP4;
                    }
                    
                    // 完整系统重置
                    grab_success = false;
                    grab_in_progress = false;
                    grab_coord_sent = false;
                    grab_result_received = false;
                    
                    // 重置y值监控系统
                    reset_y_monitoring();
                    
                    // 重置wpb_mani模块状态
                    wpb_mani_active = false;
                    
                    ROS_INFO("[系统重置] 抓取成功，完成系统状态重置");
                } else {
                    // 抓取失败（删除超时逻辑）
                    ROS_WARN("[抓取] 抓取失败，重新尝试");
                    grab_attempt_count++;
                    
                    // 完整系统重置
                    grab_success = false;
                    grab_in_progress = false;
                    grab_coord_sent = false;
                    grab_result_received = false;
                    
                    // 停止wpb_mani模块
                    wpb_mani_active = false;
                    
                    // 重置y值监控系统
                    reset_y_monitoring();
                    
                    // 收回机械臂
                    send_mani(MANI_UP);
                    
                    // 完整系统重置日志
                    ROS_INFO("[系统重置] 抓取失败，完成系统状态重置");
                    
                    // 如果抓取尝试次数超过限制，返回盒子检测阶段
                    if (grab_attempt_count >= 3) {
                        ROS_WARN("[抓取] 抓取尝试次数超过限制，返回盒子检测阶段");
                        current_step = DETECT_BOX;
                        grab_attempt_count = 0; // 重置尝试次数
                    } else {
                        // 否则重新尝试抓取
                        current_step = GRAB_BOX;
                    }
                }

                break;
            }

            case DETECT_COLOR_BOX: {
                ROS_INFO("[状态] DETECT_COLOR_BOX - 检测颜色方框");
                
                // 移除重复的抓取检测：在航点3抓取完成后已经验证过抓取结果
                ROS_INFO("[颜色匹配] 当前爪子中盒子颜色: %s，寻找匹配的颜色方框", box_color.c_str());
                
                bool color_matched = false;
                selected_color = "";  // 重置为默认值
                
                // 检测所有颜色方框
                vector<string> detected_colors = detect_all_color_boxes(current_image);
                
                // 同时显示机械爪上盒子的检测区域
                if (!box_color.empty()) {
                    // 修复：使用与航点3相同的稳定检测区域定位
                    int gripper_center_x = current_image.cols / 2;
                    int gripper_center_y = current_image.rows * 2 / 3 + 90;  // 固定在图像下方2/3位置，再往下移动四分之一个正方形长度（30像素，累计90像素），再下移一个Gripper Area边长（120像素）
                    int roi_width = 150;
                    int roi_height = 120;
                    int roi_x = max(0, gripper_center_x - roi_width / 2);
                    int roi_y = max(0, gripper_center_y - roi_height / 2);
                    roi_width = min(roi_width, current_image.cols - roi_x);
                    roi_height = min(roi_height, current_image.rows - roi_y);
                    Rect gripper_roi(roi_x, roi_y, roi_width, roi_height);
                    
                    // 启用颜色方框检测显示
                    show_color_box_detection = true;
                    color_box_frame_counter = 0;
                    
                    // 调用可视化函数显示机械爪检测区域（函数内部已包含imshow和waitKey）
                    visualize_gripper_color_detection(current_image, gripper_roi, box_color);
                }
                
                if (!detected_colors.empty()) {
                    ROS_INFO("[颜色检测] 检测到 %zu 个颜色方框: ", detected_colors.size());
                    for (const auto& color : detected_colors) {
                        ROS_INFO("  - %s", color.c_str());
                    }
                    
                    // 选择最匹配的方框
                    selected_color = select_best_match_box(box_color, detected_colors, current_image);
                    
                    if (!selected_color.empty()) {
                        ROS_INFO("[颜色匹配] 当前盒子颜色: %s, 选择方框颜色: %s", 
                                 box_color.c_str(), selected_color.c_str());
                        
                        // 获取选定方框的位置
                        Point box_center = get_color_box_position(current_image, selected_color);
                        if (box_center.x > 0 && box_center.y > 0) {
                            ROS_INFO("[方框位置] %s方框位于图像坐标: (%d, %d)", 
                                     selected_color.c_str(), box_center.x, box_center.y);
                            color_matched = true;
                        } else {
                            ROS_WARN("[颜色方框] 无法获取方框位置");
                        }
                    } else {
                        // 关键修复：当颜色不匹配时，不继续放置操作
                        ROS_WARN("[颜色匹配] 颜色不匹配，放弃放置操作，返回航点3重新抓取");
                        
                        // 清空盒子颜色信息，准备重新抓取
                        box_color.clear();
                        
                        // 返回航点3重新抓取
                        nav_msg.data = "3";
                        waypoint_pub.publish(nav_msg);
                        current_step = GOTO_WP3;
                        break;
                    }
                } else {
                    ROS_WARN("[颜色方框] 未检测到任何颜色方框");
                }
                
                if (color_matched) {
                    current_step = MOVE_TO_COLOR_BOX;
                    ROS_INFO("[状态转换] DETECT_COLOR_BOX -> MOVE_TO_COLOR_BOX");
                } else {
                    // 检测失败，继续尝试检测
                    ROS_WARN("[颜色方框] 检测失败，继续尝试检测颜色方框");
                    // 保持当前状态，继续检测
                }
                break;
            }

            case MOVE_TO_COLOR_BOX: {
                ROS_INFO("[横向移动] 开始横向移动到%s颜色方框前", selected_color.c_str());
                
                Point box_pos = get_color_box_position(current_image, selected_color);
                if (box_pos.x != -1 && box_pos.y != -1) {
                    ROS_INFO("[横向移动] %s颜色方框位置: (%d, %d)", selected_color.c_str(), box_pos.x, box_pos.y);
                    
                    // 计算横向移动方向
                    int image_center_x = current_image.cols / 2;
                    int horizontal_offset = box_pos.x - image_center_x;
                    
                    ROS_INFO("[横向移动] 方框水平偏移: %d 像素", horizontal_offset);
                    
                    // 根据方框位置决定移动方向，固定移动1秒，速度0.58米/秒
                    // 如果方框水平偏移小于70像素，视为方框在前方，不需要横向移动
                    if (abs(horizontal_offset) < 70) {
                        // 方框在正前方，不需要移动
                        ROS_INFO("[横向移动] 方框水平偏移小于70像素，视为方框在前方，不需要横向移动");
                    } else if (horizontal_offset < 0) {
                        // 方框在左侧，向左移动1秒
                        ROS_INFO("[横向移动] 向左移动1秒，速度0.58米/秒");
                        move_lateral("left", 1.0, 0.58, false);
                    } else {
                        // 方框在右侧，向右移动1秒
                        ROS_INFO("[横向移动] 向右移动1秒，速度0.58米/秒");
                        move_lateral("right", 1.0, 0.58, false);
                    }
                    
                    current_step = MOVE_FORWARD_TO_BOX;
                } else {
                    ROS_WARN("[横向移动] 无法获取方框位置，直接放置盒子");
                    current_step = PLACE_BOX;
                }
                break;
            }

            case MOVE_FORWARD_TO_BOX: {
                ROS_INFO("[前进移动] 前进到%s颜色方框前", selected_color.c_str());
                
                // 前进1秒，速度0.4米/秒
                move_forward(1.0, 0.4); // 1.0秒 * 0.4米/秒 = 0.4米距离
                
                ROS_INFO("[前进移动] 已到达%s颜色方框前，准备放置盒子", selected_color.c_str());
                current_step = PLACE_BOX;
                break;
            }

            case PLACE_BOX: {
                ROS_INFO("[放置流程] 开始放置%s方块（第%d个方块）", box_color.c_str(), box_count);
                
                // 删除爪子颜色检测验证（已在DETECT_COLOR_BOX状态完成颜色匹配）
                ROS_INFO("[放置流程] 颜色匹配已在航点4完成，直接进行放置操作");
                
                // 1. 根据目标高度计算机械臂伸出距离和角度
                ROS_INFO("[机械臂] 计算放置盒子的机械臂角度");
                
                // 机械臂参数
                float mani_base_height = 0.25;    // link3高度
                float joint3_lenght = 0.128;      // 第一节臂长度
                float joint4_lenght = 0.124 + 0.024; // 第二节臂长度
                float z_offset = 0.22;            // 优化：目标高度改为0.22米
                
                // 计算第一节臂的俯仰角
                float angle = 0;
                float actual_z_offset = z_offset - mani_base_height;
                
                // 解方程 joint3_lenght*cos(angle) - joint4_lenght*sin(angle) = z_offset
                float tmp = sqrtf(joint3_lenght * joint3_lenght + joint4_lenght * joint4_lenght);
                float b = asin(joint3_lenght/tmp);
                angle = b - asin(actual_z_offset/tmp);
                
                // 计算机械臂伸出距离
                float reachout_x_offset = joint3_lenght*sin(angle) + joint4_lenght*cos(angle) - joint3_lenght - 0.02;
                
                ROS_INFO("[机械臂] 计算完成：角度=%.3frad, 伸出距离=%.3fm, 目标高度=%.3fm", 
                         angle, reachout_x_offset, z_offset);
                
                // 优化方案：参考cruise_node优化放置流程，减少等待时间
                // 第一步：伸出机械臂到放置位置（夹爪保持闭合状态）
                ROS_INFO("[机械臂] 第一步：伸出机械臂到放置位置");
                
                // 关键修复：使用正确的机械臂角度向量
                // 根据机械臂结构，关节2控制俯仰，关节4控制末端执行器角度
                vector<double> place_angles = {0, angle, 0, 0};  // 关节2角度=angle，关节4角度=0（保持水平）
                
                ROS_INFO("[机械臂] 发送放置姿态: joint1=0, joint2=%.3f, joint3=0, joint4=0", angle);
                
                // 先只控制机械臂伸出，夹爪保持闭合状态（0.0表示闭合）
                send_mani(place_angles); // 只控制机械臂伸出
                Duration(0.3).sleep(); // 优化：机械臂伸出等待时间调整为0.3秒
                
                // 第二步：张开夹爪释放盒子（优化版本）
                ROS_INFO("[机械臂] 第二步：张开夹爪释放盒子（优化版本）");
                
                // 关键优化：确保机械臂完全到位后再控制夹爪
                ROS_INFO("[夹爪控制] 等待机械臂完全到位...");
                Duration(0.2).sleep(); // 额外等待确保机械臂稳定
                
                // 优化夹爪控制：增加重试机制和更长的等待时间
                ROS_INFO("[夹爪控制] 开始夹爪张开操作");
                set_gripper(1.0); // 只控制夹爪张开，1.0表示张开夹爪（实际转换为0.9角度）
                Duration(0.5).sleep(); // 优化：夹爪张开等待时间增加到0.5秒，确保完全张开
                
                // 第三步：等待盒子完全放下（关键优化：确保盒子成功放入垃圾桶）
                ROS_INFO("[放置] 等待盒子完全放下...");
                Duration(0.5).sleep(); // 优化：盒子放下等待时间调整为0.5秒
                
                // 第四步：抬起机械臂完成放置（夹爪保持张开状态）
                ROS_INFO("[机械臂] 抬起机械臂完成放置");
                send_mani(MANI_UP); // 只控制机械臂抬起，夹爪保持张开状态
                Duration(0.3).sleep(); // 优化：机械臂抬起等待时间调整为0.3秒
                
                // 6. 清空盒子颜色信息
                string placed_color = box_color;
                box_color.clear();
                
                ROS_INFO("[放置流程] %s方块放置完成", placed_color.c_str());

                if (box_count >= MAX_BOX) {
                    ROS_INFO("[任务] 所有%d个方块已放置完成，前往航点5执行下一个任务", MAX_BOX);
                    
                    // 优化：航点4放置完成后左转前往航点5以提高速度
                    ROS_INFO("[导航] 航点4放置完成，左转前往航点5");
                    
                    // 先左转90度，使机器人朝向航点5方向
                    geometry_msgs::Twist twist;
                    twist.angular.z = 1.5; // 角速度1.5弧度/秒
                    cmd_vel_pub.publish(twist);
                    Duration(1.05).sleep(); // 90度 = 1.57弧度，时间1.05秒
                    twist.angular.z = 0;
                    cmd_vel_pub.publish(twist);
                    
                    ROS_INFO("[导航] 左转完成，前往航点5");
                    nav_msg.data = "5";
                    waypoint_pub.publish(nav_msg);
                    current_step = GOTO_WP5;
                } else {
                    // 返回航点3前重新启用y值监控
                    y_monitoring_enabled = true;
                    reset_y_monitoring(); // 重置y值监控系统
                    ROS_INFO("[y值监控] 准备返回航点3，重新启用y值监控");
                    
                    ROS_INFO("[任务] 还有%d个方块需要抓取，返回航点3继续抓取", MAX_BOX - box_count);
                    nav_msg.data = "3";
                    waypoint_pub.publish(nav_msg);
                    current_step = GOTO_WP3;
                }
                break;
            }

            case DETECT_NUM:
                if (num_ok) {
                    ROS_INFO("[数字识别] 识别到数字: %d，将原地转圈 %d 圈", detected_number, detected_number);
                    current_step = ROTATE;
                } else {
                    // 持续进行数字识别
                    detect_num(current_image);
                }
                break;

            case ROTATE: {
                ROS_INFO("[转圈任务] 开始原地转圈，圈数: %d", detected_number);
                
                // 设置恒定的角速度（2弧度/秒）
                double angular_velocity = 2.0; // 2弧度/秒
                
                // 计算转完指定圈数所需的总时间
                // 每圈需要的时间 = 2π / 角速度 + 0.04秒延迟补偿
                double time_per_circle = 2 * M_PI / angular_velocity + 0.04; // 约3.2秒/圈（补偿小车延迟）
                double total_time = detected_number * time_per_circle; // 总时间
                
                ROS_INFO("[转圈任务] 角速度: %.2f 弧度/秒, 每圈时间: %.2f 秒, 总时间: %.2f 秒", 
                        angular_velocity, time_per_circle, total_time);
                
                // 开始旋转
                geometry_msgs::Twist twist;
                twist.angular.z = angular_velocity;
                cmd_vel_pub.publish(twist);
                
                // 记录开始时间
                double start_time = Time::now().toSec();
                
                // 持续旋转指定时间
                while (Time::now().toSec() - start_time < total_time) {
                    // 实时显示进度
                    double elapsed_time = Time::now().toSec() - start_time;
                    double current_circles = elapsed_time / time_per_circle;
                    
                    // 每0.5秒显示一次进度
                    static double last_progress_time = 0;
                    if (Time::now().toSec() - last_progress_time > 0.5) {
                        ROS_INFO("[转圈任务] 进度: %.1f圈/%.0f圈 (%.1f%%)", 
                                current_circles, (double)detected_number, 
                                (current_circles / detected_number) * 100);
                        last_progress_time = Time::now().toSec();
                    }
                    
                    Duration(0.1).sleep(); // 控制循环频率
                }
                
                // 精确停止
                twist.angular.z = 0;
                cmd_vel_pub.publish(twist);
                
                ROS_INFO("[转圈任务] 转圈完成，实际转圈数: %.1f圈", 
                        (Time::now().toSec() - start_time) / time_per_circle);
                
                ROS_INFO("[转圈任务] 转圈完成，返回航点1");
                nav_msg.data = "1";
                waypoint_pub.publish(nav_msg);
                current_step = GOTO_WP1_FINAL;
                num_ok = false;
                detected_number = 0;
                break;
            }

            case DETECT_XN:
                if (xn_ok) {
                    current_step = GOTO_WP3;
                    xn_ok = false;
                }
                break;

            default:
                break;
        }
        spinOnce();
        loop_rate.sleep();
    }

    ROS_INFO("[系统] 任务完成！得分：%d分", box_count * 10);
    destroyAllWindows();
    return 0;
}

// 新增：检测所有颜色盒子的函数 - 返回颜色字符串列表
vector<string> detect_all_color_boxes(const Mat& img) {
    vector<string> detected_colors;
    if (img.empty()) return detected_colors;
    
    // 创建可视化图像副本
    Mat display_img = img.clone();
    
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // 检测红色盒子
    Mat red1, red2, red;
    inRange(hsv, Scalar(0, 80, 50), Scalar(15, 255, 255), red1);
    inRange(hsv, Scalar(160, 80, 50), Scalar(180, 255, 255), red2);
    red = red1 | red2;
    
    // 检测蓝色盒子
    Mat blue;
    inRange(hsv, Scalar(90, 80, 50), Scalar(140, 255, 255), blue);
    
    // 检测黄色盒子
    Mat yellow;
    inRange(hsv, Scalar(15, 80, 80), Scalar(35, 255, 255), yellow);
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(red, red, MORPH_CLOSE, kernel);
    morphologyEx(blue, blue, MORPH_CLOSE, kernel);
    morphologyEx(yellow, yellow, MORPH_CLOSE, kernel);
    
    // 检测红色盒子
    vector<vector<Point>> red_contours;
    findContours(red, red_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (const auto& c : red_contours) {
        double area = contourArea(c);
        if (area > 500 && has_valid_shape(vector<vector<Point>>{c})) {
            detected_colors.push_back("red");
            
            // 可视化：绘制红色方框轮廓
            Rect bounding_rect = boundingRect(c);
            rectangle(display_img, bounding_rect, Scalar(0, 0, 255), 2); // 红色框
            putText(display_img, "Red Box", Point(bounding_rect.x, bounding_rect.y - 5), 
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 255), 1);
            
            break; // 每个颜色只添加一次
        }
    }
    
    // 检测蓝色盒子
    vector<vector<Point>> blue_contours;
    findContours(blue, blue_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (const auto& c : blue_contours) {
        double area = contourArea(c);
        if (area > 500 && has_valid_shape(vector<vector<Point>>{c})) {
            detected_colors.push_back("blue");
            
            // 可视化：绘制蓝色方框轮廓
            Rect bounding_rect = boundingRect(c);
            rectangle(display_img, bounding_rect, Scalar(255, 0, 0), 2); // 蓝色框
            putText(display_img, "Blue Box", Point(bounding_rect.x, bounding_rect.y - 5), 
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 0), 1);
            
            break; // 每个颜色只添加一次
        }
    }
    
    // 检测黄色盒子
    vector<vector<Point>> yellow_contours;
    findContours(yellow, yellow_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (const auto& c : yellow_contours) {
        double area = contourArea(c);
        if (area > 500 && has_valid_shape(vector<vector<Point>>{c})) {
            detected_colors.push_back("yellow");
            
            // 可视化：绘制黄色方框轮廓
            Rect bounding_rect = boundingRect(c);
            rectangle(display_img, bounding_rect, Scalar(0, 255, 255), 2); // 黄色框
            putText(display_img, "Yellow Box", Point(bounding_rect.x, bounding_rect.y - 5), 
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255), 1);
            
            break; // 每个颜色只添加一次
        }
    }
    
    // 将可视化结果更新到原图像
    const_cast<Mat&>(img) = display_img;
    
    return detected_colors;
}

// 新增：检测所有颜色盒子的函数 - 根据像素水平偏移选择盒子
vector<BoxInfo> detect_all_color_boxes_with_offset(const Mat& img) {
    vector<BoxInfo> detected_boxes;
    if (img.empty()) return detected_boxes;
    
    // 创建可视化图像副本
    Mat display_img = img.clone();
    
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // 检测红色盒子
    Mat red1, red2, red;
    inRange(hsv, Scalar(0, 80, 50), Scalar(15, 255, 255), red1);
    inRange(hsv, Scalar(160, 80, 50), Scalar(180, 255, 255), red2);
    red = red1 | red2;
    
    // 检测蓝色盒子
    Mat blue;
    inRange(hsv, Scalar(90, 80, 50), Scalar(140, 255, 255), blue);
    
    // 检测黄色盒子
    Mat yellow;
    inRange(hsv, Scalar(15, 80, 80), Scalar(35, 255, 255), yellow);
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(red, red, MORPH_CLOSE, kernel);
    morphologyEx(blue, blue, MORPH_CLOSE, kernel);
    morphologyEx(yellow, yellow, MORPH_CLOSE, kernel);
    
    // 检测红色盒子并计算像素水平偏移
    vector<vector<Point>> red_contours;
    findContours(red, red_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (const auto& c : red_contours) {
        double area = contourArea(c);
        if (area > 500 && has_valid_shape(vector<vector<Point>>{c})) {
            Moments m = moments(c);
            if (m.m00 != 0) {
                int center_x = int(m.m10 / m.m00);
                int img_center_x = img.cols / 2;
                int pixel_offset = center_x - img_center_x;
                
                BoxInfo box_info;
                box_info.color = "red";
                box_info.pixel_offset = pixel_offset;
                box_info.center_x = center_x;
                detected_boxes.push_back(box_info);
                
                ROS_INFO("[盒子检测] 红色盒子，像素水平偏移: %d", pixel_offset);
            }
        }
    }
    
    // 检测蓝色盒子并计算像素水平偏移
    vector<vector<Point>> blue_contours;
    findContours(blue, blue_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (const auto& c : blue_contours) {
        double area = contourArea(c);
        if (area > 500 && has_valid_shape(vector<vector<Point>>{c})) {
            Moments m = moments(c);
            if (m.m00 != 0) {
                int center_x = int(m.m10 / m.m00);
                int img_center_x = img.cols / 2;
                int pixel_offset = center_x - img_center_x;
                
                BoxInfo box_info;
                box_info.color = "blue";
                box_info.pixel_offset = pixel_offset;
                box_info.center_x = center_x;
                detected_boxes.push_back(box_info);
                
                ROS_INFO("[盒子检测] 蓝色盒子，像素水平偏移: %d", pixel_offset);
            }
        }
    }
    
    // 检测黄色盒子并计算像素水平偏移
    vector<vector<Point>> yellow_contours;
    findContours(yellow, yellow_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    for (const auto& c : yellow_contours) {
        double area = contourArea(c);
        if (area > 500 && has_valid_shape(vector<vector<Point>>{c})) {
            Moments m = moments(c);
            if (m.m00 != 0) {
                int center_x = int(m.m10 / m.m00);
                int img_center_x = img.cols / 2;
                int pixel_offset = center_x - img_center_x;
                
                BoxInfo box_info;
                box_info.color = "yellow";
                box_info.pixel_offset = pixel_offset;
                box_info.center_x = center_x;
                detected_boxes.push_back(box_info);
                
                ROS_INFO("[盒子检测] 黄色盒子，像素水平偏移: %d", pixel_offset);
            }
        }
    }
    
    // 可视化：绘制检测到的盒子信息
    string result_text = "检测到盒子数量: " + to_string(detected_boxes.size());
    Scalar result_color = (detected_boxes.size() > 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
    
    // 在图像顶部显示检测结果
    putText(display_img, result_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2);
    
    // 显示每个检测到的盒子信息
    for (size_t i = 0; i < detected_boxes.size(); i++) {
        const auto& box = detected_boxes[i];
        string box_info = "盒子" + to_string(i+1) + ": " + box.color + ", 偏移: " + to_string(box.pixel_offset);
        putText(display_img, box_info, Point(10, 60 + i*30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
    }
    
    // 更新原图像显示可视化结果
    const_cast<Mat&>(img) = display_img;
    
    return detected_boxes;
}

// 新增：根据像素水平偏移选择盒子
BoxInfo select_box_by_pixel_offset(const vector<BoxInfo>& detected_boxes) {
    if (detected_boxes.empty()) {
        BoxInfo empty_box;
        empty_box.color = "";
        empty_box.pixel_offset = 0;
        empty_box.center_x = 0;
        return empty_box;
    }
    
    // 分类盒子：前方盒子（偏移70像素以内）和左右边盒子（偏移100像素以上）
    vector<BoxInfo> front_boxes;  // 前方盒子
    vector<BoxInfo> side_boxes;   // 左右边盒子
    
    for (const auto& box : detected_boxes) {
        if (abs(box.pixel_offset) <= 70) {
            front_boxes.push_back(box);
        } else if (abs(box.pixel_offset) >= 100) {
            side_boxes.push_back(box);
        }
    }
    
    ROS_INFO("[盒子选择] 前方盒子数量: %zu, 左右边盒子数量: %zu", 
             front_boxes.size(), side_boxes.size());
    
    // 优先选择左右边盒子（偏移100像素以上）
    if (!side_boxes.empty()) {
        // 随机选择一个左右边盒子
        int random_index = rand() % side_boxes.size();
        BoxInfo selected_box = side_boxes[random_index];
        ROS_INFO("[盒子选择] 随机选择左右边盒子: %s, 像素偏移: %d", 
                 selected_box.color.c_str(), selected_box.pixel_offset);
        return selected_box;
    }
    
    // 如果没有左右边盒子，选择前方盒子
    if (!front_boxes.empty()) {
        // 随机选择一个前方盒子
        int random_index = rand() % front_boxes.size();
        BoxInfo selected_box = front_boxes[random_index];
        ROS_INFO("[盒子选择] 随机选择前方盒子: %s, 像素偏移: %d", 
                 selected_box.color.c_str(), selected_box.pixel_offset);
        return selected_box;
    }
    
    // 如果没有符合条件的盒子，返回第一个检测到的盒子
    BoxInfo selected_box = detected_boxes[0];
    ROS_WARN("[盒子选择] 没有符合偏移条件的盒子，选择第一个检测到的盒子: %s, 像素偏移: %d", 
             selected_box.color.c_str(), selected_box.pixel_offset);
    return selected_box;
}

// 新增：形状验证函数，确保检测到的是方框形状
bool has_valid_shape(const vector<vector<Point>>& contours) {
    if (contours.empty()) return false;
    
    // 找到最大轮廓
    double max_area = 0;
    vector<Point> largest_contour;
    for (const auto& c : contours) {
        double area = contourArea(c);
        if (area > max_area) {
            max_area = area;
            largest_contour = c;
        }
    }
    
    if (largest_contour.empty()) return false;
    
    // 计算轮廓的矩形边界
    Rect bounding_rect = boundingRect(largest_contour);
    
    // 计算宽高比，方框应该接近正方形
    double aspect_ratio = (double)bounding_rect.width / bounding_rect.height;
    
    // 放宽宽高比限制，适应不同角度和距离的方框
    return (aspect_ratio >= 0.3 && aspect_ratio <= 3.0);
}

// 新增：计算颜色方框的面积
int calculate_color_box_area(const Mat& img, const string& color_name) {
    if (img.empty()) {
        ROS_WARN("[面积计算] 图像为空");
        return 0;
    }
    
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    Mat color_mask;
    
    // 根据颜色名称设置HSV阈值
    if (color_name == "red") {
        Mat red1, red2;
        inRange(hsv, Scalar(0, 80, 50), Scalar(15, 255, 255), red1);
        inRange(hsv, Scalar(160, 80, 50), Scalar(180, 255, 255), red2);
        color_mask = red1 | red2;
    } else if (color_name == "yellow") {
        inRange(hsv, Scalar(15, 80, 80), Scalar(35, 255, 255), color_mask);
    } else if (color_name == "blue") {
        inRange(hsv, Scalar(90, 80, 50), Scalar(140, 255, 255), color_mask);
    } else {
        ROS_WARN("[面积计算] 未知颜色: %s", color_name.c_str());
        return 0;
    }
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(color_mask, color_mask, MORPH_CLOSE, kernel);
    
    // 查找轮廓并计算面积
    vector<vector<Point>> contours;
    findContours(color_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    int max_area = 0;
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > max_area && has_valid_shape(vector<vector<Point>>{contour})) {
            max_area = static_cast<int>(area);
        }
    }
    
    ROS_INFO("[面积计算] %s颜色方框面积: %d像素", color_name.c_str(), max_area);
    return max_area;
}

// 新增：选择最匹配的方框颜色（区分垃圾桶和机械爪上的盒子）
string select_best_match_box(const string& current_box_color, const vector<string>& detected_colors, const Mat& img) {
    // 如果当前盒子颜色为空，尝试智能匹配策略
    if (current_box_color.empty()) {
        ROS_WARN("[颜色匹配] 当前盒子颜色为空，尝试智能匹配");
        
        if (detected_colors.empty()) {
            ROS_WARN("[颜色匹配] 没有检测到任何颜色方框");
            return "";
        }
        
        // 智能匹配策略1：根据方框数量进行推断
        if (detected_colors.size() == 3) {
            // 如果检测到3个方框，说明所有颜色都存在
            // 根据方框位置推断当前盒子颜色
            // 假设方框从左到右排列：红、黄、蓝
            
            // 检查方框颜色组合
            bool red_found = false, yellow_found = false, blue_found = false;
            for (const auto& color : detected_colors) {
                if (color == "red") red_found = true;
                if (color == "yellow") yellow_found = true;
                if (color == "blue") blue_found = true;
            }
            
            // 如果颜色组合完整，可以推断当前盒子颜色
            if (red_found && yellow_found && blue_found) {
                // 假设当前盒子是中间颜色（黄色）
                string inferred_color = "yellow";
                ROS_WARN("[颜色匹配] 推断当前盒子颜色为: %s（基于方框位置）", inferred_color.c_str());
                
                // 验证推断的颜色是否在方框中存在
                for (const auto& color : detected_colors) {
                    if (color == inferred_color) {
                        ROS_INFO("[颜色匹配] 找到匹配的颜色方框: %s", inferred_color.c_str());
                        return inferred_color;
                    }
                }
            }
        }
        
        // 智能匹配策略2：如果只有2个方框，选择不重复的颜色
        if (detected_colors.size() == 2) {
            // 检查两个方框的颜色
            string color1 = detected_colors[0];
            string color2 = detected_colors[1];
            
            // 如果两个方框颜色不同，选择第三个颜色
            if (color1 != color2) {
                string third_color = "";
                if (color1 == "red" && color2 == "yellow") third_color = "blue";
                else if (color1 == "red" && color2 == "blue") third_color = "yellow";
                else if (color1 == "yellow" && color2 == "blue") third_color = "red";
                
                if (!third_color.empty()) {
                    ROS_WARN("[颜色匹配] 推断当前盒子颜色为: %s（基于缺失颜色）", third_color.c_str());
                    return third_color;
                }
            }
        }
        
        // 如果无法推断，返回第一个方框颜色作为备选
        ROS_WARN("[颜色匹配] 无法推断盒子颜色，使用第一个方框颜色: %s", detected_colors[0].c_str());
        return detected_colors[0];
    }
    
    ROS_INFO("[颜色匹配] 当前爪子中盒子颜色: %s, 寻找匹配的颜色方框", current_box_color.c_str());
    
    // 检查是否有相同颜色的多个方框（需要面积区分）
    int same_color_count = 0;
    for (const auto& color : detected_colors) {
        if (color == current_box_color) {
            same_color_count++;
        }
    }
    
    // 如果有相同颜色的多个方框，优先使用面积区分逻辑
    if (same_color_count >= 2) {
        ROS_INFO("[颜色匹配] 检测到%d个相同颜色的方框，启用面积区分逻辑", same_color_count);
        
        // 计算所有检测到的颜色方框的面积
        map<string, int> color_areas;
        for (const auto& color : detected_colors) {
            int area = calculate_color_box_area(img, color);
            color_areas[color] = area;
            ROS_INFO("[面积区分] %s颜色方框面积: %d像素", color.c_str(), area);
        }
        
        // 找到面积最大和最小的方框
        string largest_color = "";
        string smallest_color = "";
        int max_area = 0;
        int min_area = INT_MAX;
        
        for (const auto& pair : color_areas) {
            if (pair.second > max_area) {
                max_area = pair.second;
                largest_color = pair.first;
            }
            if (pair.second < min_area) {
                min_area = pair.second;
                smallest_color = pair.first;
            }
        }
        
        // 面积区分逻辑：面积大的为垃圾桶，面积小的为机械爪上的盒子
        if (!largest_color.empty() && !smallest_color.empty()) {
            ROS_INFO("[面积区分] 面积最大方框: %s (%d像素) - 垃圾桶", largest_color.c_str(), max_area);
            ROS_INFO("[面积区分] 面积最下方框: %s (%d像素) - 机械爪上的盒子", smallest_color.c_str(), min_area);
            
            // 关键修复：当前盒子颜色是夹爪上的盒子颜色，应该选择相同颜色的垃圾桶
            // 如果当前盒子颜色与面积最大的方框匹配，说明找到了正确的目标垃圾桶
            if (current_box_color == largest_color) {
                ROS_INFO("[面积区分] 当前盒子颜色与面积最大方框匹配，选择正确的目标垃圾桶: %s", largest_color.c_str());
                return largest_color;
            }
            // 如果当前盒子颜色与面积最小的方框匹配，说明夹爪上的盒子被检测为小面积方框
            // 应该选择相同颜色的大面积方框（垃圾桶）
            else if (current_box_color == smallest_color) {
                // 寻找相同颜色的大面积方框（垃圾桶）
                for (const auto& color : detected_colors) {
                    if (color == current_box_color && color_areas[color] > min_area * 1.5) {
                        ROS_INFO("[面积区分] 找到相同颜色的大面积方框（垃圾桶）: %s", color.c_str());
                        return color;
                    }
                }
                // 如果没有找到相同颜色的大面积方框，选择面积最大的方框
                ROS_WARN("[面积区分] 未找到相同颜色的大面积方框，选择面积最大的垃圾桶: %s", largest_color.c_str());
                return largest_color;
            }
            // 如果都不匹配，寻找相同颜色的方框
            else {
                for (const auto& color : detected_colors) {
                    if (color == current_box_color) {
                        ROS_INFO("[面积区分] 找到相同颜色的方框: %s", color.c_str());
                        return color;
                    }
                }
                // 如果没有相同颜色的方框，选择面积最大的方框
                ROS_WARN("[面积区分] 未找到相同颜色的方框，选择面积最大的垃圾桶: %s", largest_color.c_str());
                return largest_color;
            }
        }
    }
    
    // 如果没有相同颜色的多个方框，正常匹配相同颜色的方框
    for (const auto& color : detected_colors) {
        if (color == current_box_color) {
            ROS_INFO("[颜色匹配] 找到匹配的颜色方框: %s", color.c_str());
            return color;
        }
    }
    
    // 如果没有相同颜色的方框，尝试其他匹配策略
    ROS_WARN("[颜色匹配] 没有找到相同颜色的方框，尝试其他匹配策略");
    
    // 策略1：检查是否有重复颜色（可能检测错误）
    map<string, int> color_count;
    for (const auto& color : detected_colors) {
        color_count[color]++;
    }
    
    // 如果有重复颜色，选择出现次数最多的颜色
    string most_common_color = "";
    int max_count = 0;
    for (const auto& pair : color_count) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common_color = pair.first;
        }
    }
    
    if (max_count > 1) {
        ROS_WARN("[颜色匹配] 检测到重复颜色，选择出现次数最多的颜色: %s", most_common_color.c_str());
        return most_common_color;
    }
    
    // 新增策略：基于位置验证的匹配
    // 夹爪上的盒子通常位于图像下方，而目标垃圾桶位于图像前方
    if (detected_colors.size() >= 2) {
        ROS_INFO("[位置验证] 开始基于位置验证的匹配策略");
        
        // 获取所有检测到的方框位置
        map<string, Point> color_positions;
        for (const auto& color : detected_colors) {
            Point pos = get_color_box_position(img, color);
            if (pos.x > 0 && pos.y > 0) {
                color_positions[color] = pos;
                ROS_INFO("[位置验证] %s颜色方框位置: (%d, %d)", color.c_str(), pos.x, pos.y);
            }
        }
        
        // 如果检测到多个方框，选择位于图像前方（中心区域）的方框
        if (color_positions.size() >= 2) {
            int image_center_y = img.rows / 2;
            
            // 找到最靠近图像中心的方框（前方目标）
            string front_color = "";
            int min_y_distance = INT_MAX;
            
            for (const auto& pair : color_positions) {
                int y_distance = abs(pair.second.y - image_center_y);
                if (y_distance < min_y_distance) {
                    min_y_distance = y_distance;
                    front_color = pair.first;
                }
            }
            
            if (!front_color.empty()) {
                ROS_INFO("[位置验证] 选择位于前方的方框: %s (Y距离: %d像素)", 
                         front_color.c_str(), min_y_distance);
                return front_color;
            }
        }
    }
    
    // 策略2：根据面积区分垃圾桶和机械爪上的盒子
    if (detected_colors.size() >= 2) {
        ROS_INFO("[面积区分] 开始根据面积区分垃圾桶和机械爪上的盒子");
        
        // 计算所有检测到的颜色方框的面积
        map<string, int> color_areas;
        for (const auto& color : detected_colors) {
            int area = calculate_color_box_area(img, color);
            color_areas[color] = area;
            ROS_INFO("[面积区分] %s颜色方框面积: %d像素", color.c_str(), area);
        }
        
        // 找到面积最大和最小的方框
        string largest_color = "";
        string smallest_color = "";
        int max_area = 0;
        int min_area = INT_MAX;
        
        for (const auto& pair : color_areas) {
            if (pair.second > max_area) {
                max_area = pair.second;
                largest_color = pair.first;
            }
            if (pair.second < min_area) {
                min_area = pair.second;
                smallest_color = pair.first;
            }
        }
        
        // 面积区分逻辑：面积大的为垃圾桶，面积小的为机械爪上的盒子
        if (!largest_color.empty() && !smallest_color.empty() && max_area > min_area * 1.5) {
            ROS_INFO("[面积区分] 面积最大方框: %s (%d像素) - 垃圾桶", largest_color.c_str(), max_area);
            ROS_INFO("[面积区分] 面积最下方框: %s (%d像素) - 机械爪上的盒子", smallest_color.c_str(), min_area);
            
            // 关键修复：当前盒子颜色是夹爪上的盒子颜色，应该选择相同颜色的垃圾桶
            // 如果当前盒子颜色与面积最小的方框匹配，说明夹爪上的盒子被检测为小面积方框
            // 应该选择相同颜色的大面积方框（垃圾桶）
            if (current_box_color == smallest_color) {
                // 寻找相同颜色的大面积方框（垃圾桶）
                for (const auto& color : detected_colors) {
                    if (color == current_box_color && color_areas[color] > min_area * 1.5) {
                        ROS_INFO("[面积区分] 找到相同颜色的大面积方框（垃圾桶）: %s", color.c_str());
                        return color;
                    }
                }
                // 如果没有找到相同颜色的大面积方框，选择面积最大的方框
                ROS_WARN("[面积区分] 未找到相同颜色的大面积方框，选择面积最大的垃圾桶: %s", largest_color.c_str());
                return largest_color;
            }
            // 如果当前盒子颜色与面积最大的方框匹配，说明找到了正确的目标垃圾桶
            else if (current_box_color == largest_color) {
                ROS_INFO("[面积区分] 当前盒子颜色与面积最大方框匹配，选择正确的目标垃圾桶: %s", largest_color.c_str());
                return largest_color;
            }
            // 如果都不匹配，寻找相同颜色的方框
            else {
                for (const auto& color : detected_colors) {
                    if (color == current_box_color) {
                        ROS_INFO("[面积区分] 找到相同颜色的方框: %s", color.c_str());
                        return color;
                    }
                }
                // 如果没有相同颜色的方框，选择面积最大的方框
                ROS_WARN("[面积区分] 未找到相同颜色的方框，选择面积最大的垃圾桶: %s", largest_color.c_str());
                return largest_color;
            }
        } else {
            ROS_WARN("[面积区分] 面积差异不明显，无法区分垃圾桶和机械爪盒子");
        }
    }
    
    // 策略3：根据方框位置选择（假设方框按红黄蓝顺序排列）
    if (detected_colors.size() == 3) {
        // 检查颜色组合
        bool red_found = false, yellow_found = false, blue_found = false;
        for (const auto& color : detected_colors) {
            if (color == "red") red_found = true;
            if (color == "yellow") yellow_found = true;
            if (color == "blue") blue_found = true;
        }
        
        // 如果颜色顺序正确，根据当前盒子颜色选择匹配方框
        if (red_found && yellow_found && blue_found) {
            // 如果当前盒子是红色，匹配黄色方框
            if (current_box_color == "red") {
                ROS_WARN("[颜色匹配] 根据方框位置匹配黄色方框");
                return "yellow";
            }
            // 如果当前盒子是黄色，匹配蓝色方框
            else if (current_box_color == "yellow") {
                ROS_WARN("[颜色匹配] 根据方框位置匹配蓝色方框");
                return "blue";
            }
            // 如果当前盒子是蓝色，匹配红色方框
            else if (current_box_color == "blue") {
                ROS_WARN("[颜色匹配] 根据方框位置匹配红色方框");
                return "red";
            }
        }
    }
    
    // 策略4：选择第一个方框作为备选
    ROS_WARN("[颜色匹配] 使用第一个方框颜色作为备选: %s", detected_colors[0].c_str());
    return detected_colors[0];
}

// 新增：更新y值监控系统
void update_y_monitoring(double current_y) {
    static double last_y = 0.0;
    static ros::Time last_time = ros::Time::now();
    
    // 如果y值监控未启用，直接返回
    if (!y_monitoring_enabled) {
        return;
    }
    
    // 计算y值变化量
    double y_change = fabs(current_y - last_y);
    last_y = current_y;
    
    // 记录调试信息
    ROS_INFO("[Y值监控] 当前Y值: %.3f, 变化量: %.3f", current_y, y_change);
    
    // 追踪阶段：监控y值变化
    if (!in_fine_tune_phase) {
        // 监控y值变化大于0.1（累积两次）
        if (y_change > 0.1) {
            y_large_change_counter++;
            ROS_INFO("[Y值监控] 大变化计数: %d", y_large_change_counter);
        } else {
            y_large_change_counter = 0;
        }
        
        // 监控y值变化小于0.01（连续8次）
        if (y_change < 0.01) {
            y_small_change_counter++;
            ROS_INFO("[Y值监控] 小变化计数: %d", y_small_change_counter);
        } else {
            y_small_change_counter = 0;
        }
        
        // 触发大变化处理（累积两次>0.1）
        if (y_large_change_counter >= 2 && !y_large_change_triggered) {
            ROS_WARN("[Y值监控] 触发大变化处理：关闭wpb_mani模块");
            y_large_change_triggered = true;
            
            // 关闭wpb_mani模块
            wpb_mani_enabled = false;
            
            // 发布0速度指令（30Hz，持续0.5秒）
            publish_zero_velocity_for_duration(0.5);
            
            // 0.5秒后重启wpb_mani模块
            ros::Duration(0.5).sleep();
            wpb_mani_enabled = true;
            ROS_INFO("[Y值监控] 重启wpb_mani模块");
        }
        
        // 触发小变化处理（连续8次<0.01）
        if (y_small_change_counter >= 8 && !y_small_change_triggered) {
            ROS_WARN("[Y值监控] 触发小变化处理：进入微调阶段");
            y_small_change_triggered = true;
            in_fine_tune_phase = true;
            y_fine_tune_counter = 0;
        }
    }
    // 微调阶段：监控y值变化小于0.002（连续12次）
    else {
        if (y_change < 0.002) {
            y_fine_tune_counter++;
            ROS_INFO("[Y值监控] 微调计数: %d", y_fine_tune_counter);
        } else {
            y_fine_tune_counter = 0;
        }
        
        // 触发微调完成处理（连续12次<0.002）
        if (y_fine_tune_counter >= 12 && !y_fine_tune_triggered) {
            ROS_WARN("[Y值监控] 触发微调完成：发布0速度命令并抓取盒子");
            y_fine_tune_triggered = true;
            
            // 发布0速度命令
            publish_zero_velocity();
            
            // 马上伸出机械臂
            extend_manipulator();
            
            // 向前进1.5秒抓取盒子
            move_forward(1.5, 0.25);
        }
    }
}

// 新增：重置y值监控系统
void reset_y_monitoring_system() {
    y_large_change_counter = 0;
    y_small_change_counter = 0;
    y_fine_tune_counter = 0;
    y_large_change_triggered = false;
    y_small_change_triggered = false;
    y_fine_tune_triggered = false;
    y_monitoring_enabled = false;
    in_fine_tune_phase = false;
    
    ROS_INFO("[Y值监控] 系统已重置");
}

// 新增：发布0速度指令（持续指定时间）
void publish_zero_velocity_for_duration(double duration) {
    ros::Rate rate(30); // 30Hz
    ros::Time start_time = ros::Time::now();
    
    while ((ros::Time::now() - start_time).toSec() < duration) {
        // 发布0速度指令
        geometry_msgs::Twist cmd_vel;
        cmd_vel.linear.x = 0.0;
        cmd_vel.linear.y = 0.0;
        cmd_vel.linear.z = 0.0;
        cmd_vel.angular.x = 0.0;
        cmd_vel.angular.y = 0.0;
        cmd_vel.angular.z = 0.0;
        
        cmd_vel_pub.publish(cmd_vel);
        rate.sleep();
    }
    
    ROS_INFO("[Y值监控] 0速度指令发布完成，持续时间: %.1f秒", duration);
}

// 新增：发布单次0速度指令
void publish_zero_velocity() {
    geometry_msgs::Twist cmd_vel;
    cmd_vel.linear.x = 0.0;
    cmd_vel.linear.y = 0.0;
    cmd_vel.linear.z = 0.0;
    cmd_vel.angular.x = 0.0;
    cmd_vel.angular.y = 0.0;
    cmd_vel.angular.z = 0.0;
    
    cmd_vel_pub.publish(cmd_vel);
    ROS_INFO("[Y值监控] 发布单次0速度指令");
}

// 新增：伸出机械臂
void extend_manipulator() {
    // 设置机械臂到抓取位置
    std_msgs::String mani_msg;
    mani_msg.data = "grab";
    mani_pub.publish(mani_msg);
    
    ROS_INFO("[Y值监控] 伸出机械臂准备抓取");
}

// 新增：航点3颜色像素块检测函数
vector<BoxInfo> detect_color_pixels_on_desk(const Mat& img) {
    vector<BoxInfo> detected_boxes;
    if (img.empty()) return detected_boxes;
    
    // 创建可视化图像副本
    Mat display_img = img.clone();
    
    // 转换到HSV颜色空间，便于颜色检测
    Mat hsv_img;
    cvtColor(img, hsv_img, COLOR_BGR2HSV);
    
    // 定义红色、黄色、蓝色的HSV范围
    // 红色范围（两个范围，因为红色在HSV中跨越0度）
    Scalar red_lower1(0, 120, 70);
    Scalar red_upper1(10, 255, 255);
    Scalar red_lower2(170, 120, 70);
    Scalar red_upper2(180, 255, 255);
    
    // 黄色范围
    Scalar yellow_lower(20, 100, 100);
    Scalar yellow_upper(30, 255, 255);
    
    // 蓝色范围
    Scalar blue_lower(100, 150, 50);
    Scalar blue_upper(140, 255, 255);
    
    // 检测红色像素块
    Mat red_mask1, red_mask2, red_mask;
    inRange(hsv_img, red_lower1, red_upper1, red_mask1);
    inRange(hsv_img, red_lower2, red_upper2, red_mask2);
    red_mask = red_mask1 | red_mask2;
    
    // 检测黄色像素块
    Mat yellow_mask;
    inRange(hsv_img, yellow_lower, yellow_upper, yellow_mask);
    
    // 检测蓝色像素块
    Mat blue_mask;
    inRange(hsv_img, blue_lower, blue_upper, blue_mask);
    
    // 形态学操作，去除噪声
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    morphologyEx(red_mask, red_mask, MORPH_OPEN, kernel);
    morphologyEx(yellow_mask, yellow_mask, MORPH_OPEN, kernel);
    morphologyEx(blue_mask, blue_mask, MORPH_OPEN, kernel);
    
    // 查找红色像素块的轮廓
    vector<vector<Point>> red_contours;
    findContours(red_mask, red_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // 查找黄色像素块的轮廓
    vector<vector<Point>> yellow_contours;
    findContours(yellow_mask, yellow_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // 查找蓝色像素块的轮廓
    vector<vector<Point>> blue_contours;
    findContours(blue_mask, blue_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // 处理红色像素块
    for (const auto& contour : red_contours) {
        double area = contourArea(contour);
        if (area > 100) { // 面积阈值，过滤小噪声
            Rect bounding_rect = boundingRect(contour);
            BoxInfo box_info;
            box_info.color = "red";
            box_info.x = bounding_rect.x + bounding_rect.width / 2; // 中心x坐标
            box_info.y = bounding_rect.y + bounding_rect.height / 2; // 中心y坐标
            box_info.width = bounding_rect.width;
            box_info.height = bounding_rect.height;
            box_info.area = area;
            detected_boxes.push_back(box_info);
            
            ROS_INFO("[颜色像素块检测] 检测到红色像素块，位置: (%d, %d), 面积: %.1f", 
                     box_info.x, box_info.y, area);
        }
    }
    
    // 处理黄色像素块
    for (const auto& contour : yellow_contours) {
        double area = contourArea(contour);
        if (area > 100) { // 面积阈值，过滤小噪声
            Rect bounding_rect = boundingRect(contour);
            BoxInfo box_info;
            box_info.color = "yellow";
            box_info.x = bounding_rect.x + bounding_rect.width / 2;
            box_info.y = bounding_rect.y + bounding_rect.height / 2;
            box_info.width = bounding_rect.width;
            box_info.height = bounding_rect.height;
            box_info.area = area;
            detected_boxes.push_back(box_info);
            
            ROS_INFO("[颜色像素块检测] 检测到黄色像素块，位置: (%d, %d), 面积: %.1f", 
                     box_info.x, box_info.y, area);
        }
    }
    
    // 处理蓝色像素块
    for (const auto& contour : blue_contours) {
        double area = contourArea(contour);
        if (area > 100) { // 面积阈值，过滤小噪声
            Rect bounding_rect = boundingRect(contour);
            BoxInfo box_info;
            box_info.color = "blue";
            box_info.x = bounding_rect.x + bounding_rect.width / 2;
            box_info.y = bounding_rect.y + bounding_rect.height / 2;
            box_info.width = bounding_rect.width;
            box_info.height = bounding_rect.height;
            box_info.area = area;
            detected_boxes.push_back(box_info);
            
            ROS_INFO("[颜色像素块检测] 检测到蓝色像素块，位置: (%d, %d), 面积: %.1f", 
                     box_info.x, box_info.y, area);
        }
    }
    
    // 可视化：绘制检测到的颜色像素块
    for (const auto& box : detected_boxes) {
        // 根据颜色设置绘制颜色
        Scalar box_color;
        if (box.color == "red") box_color = Scalar(0, 0, 255);
        else if (box.color == "blue") box_color = Scalar(255, 0, 0);
        else if (box.color == "yellow") box_color = Scalar(0, 255, 255);
        else box_color = Scalar(255, 255, 255);
        
        // 绘制矩形框
        Rect box_rect(box.x - box.width/2, box.y - box.height/2, box.width, box.height);
        rectangle(display_img, box_rect, box_color, 2);
        
        // 绘制中心点
        circle(display_img, Point(box.x, box.y), 3, box_color, -1);
        
        // 添加文字标注
        string box_label = box.color + " (" + to_string(int(box.area)) + ")";
        putText(display_img, box_label, Point(box.x - 30, box.y - 10), 
                FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);
    }
    
    // 显示检测结果统计
    string result_text = "检测到颜色像素块: " + to_string(detected_boxes.size());
    Scalar result_color = (detected_boxes.size() > 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
    putText(display_img, result_text, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2);
    
    // 更新原图像显示可视化结果
    const_cast<Mat&>(img) = display_img;
    
    ROS_INFO("[颜色像素块检测] 总共检测到 %zu 个颜色像素块", detected_boxes.size());
    return detected_boxes;
}

// 新增：计算盒子模型像素偏移量
int calculate_box_pixel_offset(const Mat& img, const string& color) {
    if (img.empty()) {
        ROS_WARN("[像素偏移] 图像为空，无法计算像素偏移");
        return 0;
    }
    
    Mat hsv;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    
    // 根据颜色设置HSV范围
    Mat color_mask;
    if (color == "red") {
        Mat red1, red2;
        inRange(hsv, Scalar(0, 80, 50), Scalar(15, 255, 255), red1);
        inRange(hsv, Scalar(160, 80, 50), Scalar(180, 255, 255), red2);
        color_mask = red1 | red2;
    } else if (color == "blue") {
        inRange(hsv, Scalar(90, 80, 50), Scalar(140, 255, 255), color_mask);
    } else if (color == "yellow") {
        inRange(hsv, Scalar(15, 80, 80), Scalar(35, 255, 255), color_mask);
    } else {
        ROS_WARN("[像素偏移] 未知颜色: %s", color.c_str());
        return 0;
    }
    
    // 形态学操作去除噪声
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(color_mask, color_mask, MORPH_CLOSE, kernel);
    
    // 查找轮廓
    vector<vector<Point>> contours;
    findContours(color_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    
    // 找到最大轮廓
    double max_area = 0;
    Point2f center(0, 0);
    
    for (const auto& contour : contours) {
        double area = contourArea(contour);
        if (area > max_area && area > 500) {
            max_area = area;
            Moments m = moments(contour);
            if (m.m00 != 0) {
                center.x = float(m.m10 / m.m00);
                center.y = float(m.m01 / m.m00);
            }
        }
    }
    
    if (max_area == 0) {
        ROS_WARN("[像素偏移] 未检测到有效的%s盒子轮廓", color.c_str());
        return 0;
    }
    
    // 计算像素偏移量（相对于图像中心）
    int img_center_x = img.cols / 2;
    int pixel_offset = int(center.x) - img_center_x;
    
    ROS_INFO("[像素偏移] %s盒子像素偏移量: %d (中心位置: %.1f, 图像中心: %d)", 
             color.c_str(), pixel_offset, center.x, img_center_x);
    
    return pixel_offset;
}

// 新增：根据颜色区域选择移动方向
string select_direction_by_color_regions(const vector<BoxInfo>& color_regions) {
    if (color_regions.empty()) {
        ROS_WARN("[方向选择] 颜色区域为空，无法选择移动方向");
        return "none";
    }
    
    // 获取图像宽度（假设为640像素）
    int image_width = 640;
    int center_x = image_width / 2;
    
    // 计算所有检测到的颜色区域的平均水平位置
    double avg_x = 0.0;
    double total_area = 0.0;
    
    for (const auto& region : color_regions) {
        avg_x += region.x * region.area; // 加权平均，面积大的区域权重更高
        total_area += region.area;
    }
    
    if (total_area == 0) {
        ROS_WARN("[方向选择] 颜色区域总面积为零，无法计算平均位置");
        return "none";
    }
    
    avg_x /= total_area;
    
    // 计算水平偏移量
    double horizontal_offset = avg_x - center_x;
    
    ROS_INFO("[方向选择] 加权平均位置: %.1f, 中心位置: %d, 水平偏移: %.1f", 
             avg_x, center_x, horizontal_offset);
    
    // 根据水平偏移选择移动方向
    if (fabs(horizontal_offset) < 70) { // pixel_threshold_near = 70
        // 偏移量很小，说明盒子在正前方
        ROS_INFO("[方向选择] 盒子在正前方，不需要横向移动");
        return "forward";
    } else if (horizontal_offset > 0) {
        // 偏移量为正，说明盒子在右侧
        ROS_INFO("[方向选择] 盒子在右侧，需要向左横向移动");
        return "left";
    } else {
        // 偏移量为负，说明盒子在左侧
        ROS_INFO("[方向选择] 盒子在左侧，需要向右横向移动");
        return "right";
    }
}

// 新增：执行初始定位移动
void execute_initial_positioning(const string& direction) {
    if (direction == "none" || direction == "forward") {
        ROS_INFO("[初始定位] 不需要横向移动，直接进入抓取准备");
        return;
    }
    
    // 横向移动参数
    double lateral_velocity = 0.5; // m/s
    double lateral_duration = 1.0; // 秒
    
    // 前进移动参数
    double forward_velocity = 0.25; // m/s
    double forward_duration = 1.0; // 秒
    
    // 执行横向移动
    if (direction == "left") {
        ROS_INFO("[初始定位] 向左横向移动，速度%.1fm/s，持续%.1f秒", 
                 lateral_velocity, lateral_duration);
        move_lateral("left", lateral_duration, lateral_velocity);
    } else if (direction == "right") {
        ROS_INFO("[初始定位] 向右横向移动，速度%.1fm/s，持续%.1f秒", 
                 lateral_velocity, lateral_duration);
        move_lateral("right", lateral_duration, lateral_velocity);
    }
    
    // 执行前进移动
    ROS_INFO("[初始定位] 前进靠近目标区域，速度%.1fm/s，持续%.1f秒", 
             forward_velocity, forward_duration);
    move_forward(forward_duration, forward_velocity);
}

// 新增：根据水平偏移选择方向的横向移动逻辑
string select_direction_by_horizontal_offset(const vector<BoxInfo>& detected_boxes) {
    if (detected_boxes.empty()) {
        ROS_WARN("[方向选择] 未检测到任何颜色像素块，无法选择移动方向");
        return "none";
    }
    
    // 获取图像宽度（假设为640像素）
    int image_width = 640;
    int center_x = image_width / 2;
    
    // 计算所有检测到的像素块的平均水平位置
    double avg_x = 0.0;
    for (const auto& box : detected_boxes) {
        avg_x += box.x;
    }
    avg_x /= detected_boxes.size();
    
    // 计算水平偏移量
    double horizontal_offset = avg_x - center_x;
    
    ROS_INFO("[方向选择] 平均水平位置: %.1f, 中心位置: %d, 水平偏移: %.1f", 
             avg_x, center_x, horizontal_offset);
    
    // 根据水平偏移选择移动方向
    if (fabs(horizontal_offset) < 50) {
        // 偏移量很小，说明盒子在正前方
        ROS_INFO("[方向选择] 盒子在正前方，不需要横向移动");
        return "forward";
    } else if (horizontal_offset > 0) {
        // 偏移量为正，说明盒子在右侧
        ROS_INFO("[方向选择] 盒子在右侧，需要向左横向移动");
        return "left";
    } else {
        // 偏移量为负，说明盒子在左侧
        ROS_INFO("[方向选择] 盒子在左侧，需要向右横向移动");
        return "right";
    }
}

// 新增：频率降低机制 - 接收来自wpb_mani模块的cmd_vel指令
void wpbManiCmdVelCallback(const geometry_msgs::Twist::ConstPtr& msg) {
    // 只有在GRAB_BOX状态且wpb_mani模块活跃时才处理指令
    if (current_step == GRAB_BOX && wpb_mani_active) {
        // 存储最新的指令
        last_wpb_mani_cmd = *msg;
        has_new_wpb_mani_cmd = true;
        
        ROS_DEBUG("[频率降低] 接收到wpb_mani模块指令: vx=%.3f, vy=%.3f, vz=%.3f", 
                  msg->linear.x, msg->linear.y, msg->linear.z);
    }
}

// 新增：频率降低机制 - 以10Hz频率发布指令
void publishReducedFrequencyCmd() {
    // 只有在GRAB_BOX状态且wpb_mani模块活跃时才处理指令
    if (current_step == GRAB_BOX && wpb_mani_active && has_new_wpb_mani_cmd) {
        ros::Time current_time = ros::Time::now();
        
        // 检查是否达到目标间隔（0.1秒，对应10Hz）
        if ((current_time - last_cmd_time).toSec() >= 0.1) {
            // 发布指令
            cmd_vel_pub.publish(last_wpb_mani_cmd);
            last_cmd_time = current_time;
            has_new_wpb_mani_cmd = false; // 重置标志
            
            ROS_DEBUG("[频率降低] 以10Hz频率发布指令: vx=%.3f, vy=%.3f, vz=%.3f", 
                      last_wpb_mani_cmd.linear.x, last_wpb_mani_cmd.linear.y, last_wpb_mani_cmd.linear.z);
        }
    }
}

// 可视化函数：显示机械爪检测区域
void visualize_detection_regions(const Mat& img, const Rect& gripper_roi, const string& detection_method) {
    // 直接在传入的图像上绘制（不创建新窗口）
    Mat& display_img = const_cast<Mat&>(img);
    
    // 绘制机械爪检测区域
    rectangle(display_img, gripper_roi, Scalar(0, 255, 0), 3); // 绿色框表示机械爪检测区域
    
    // 添加文字标注
    putText(display_img, "Gripper Detection Area", Point(gripper_roi.x, gripper_roi.y - 15), 
            FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
    putText(display_img, "Method: " + detection_method, Point(10, 30), 
            FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
    putText(display_img, "ROI: (" + to_string(gripper_roi.x) + "," + to_string(gripper_roi.y) + 
            ") - (" + to_string(gripper_roi.width) + "x" + to_string(gripper_roi.height) + ")", 
            Point(10, 60), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 2);
}

// 可视化函数：显示智能颜色检测结果
void visualize_smart_color_detection(const Mat& img, const Rect& gripper_roi, const string& detected_color,
                                    bool red_touches_edge, bool yellow_touches_edge, bool blue_touches_edge) {
    // 创建图像副本进行绘制，避免修改原始图像
    Mat display_img = img.clone();
    
    // 绘制机械爪检测区域
    rectangle(display_img, gripper_roi, Scalar(0, 255, 0), 3); // 绿色框表示机械爪检测区域
    
    // 计算爪子中心白点位置
    Point gripper_center(gripper_roi.x + gripper_roi.width / 2, gripper_roi.y + gripper_roi.height / 2);
    
    // 绘制颜色检测区域
    rectangle(display_img, gripper_roi, Scalar(0, 0, 255), 2); // 红色框表示颜色检测区域
    
    // 绘制爪子中心点
    circle(display_img, gripper_center, 5, Scalar(255, 255, 255), -1); // 白色中心点
    
    // 添加文字标注
    string status_text = "Smart Color Detection";
    Scalar status_color = Scalar(0, 255, 255); // 黄色
    
    if (!detected_color.empty()) {
        status_text = "Detected: " + detected_color;
        status_color = Scalar(0, 255, 0); // 绿色
    } else {
        status_text = "No Color Detected";
        status_color = Scalar(0, 0, 255); // 红色
    }
    
    putText(display_img, status_text, Point(10, 30), 
            FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2);
    
    // 显示边缘接触情况
    string edge_info = "Edge Contact - Red:" + string(red_touches_edge ? "Yes" : "No") + 
                      " Yellow:" + string(yellow_touches_edge ? "Yes" : "No") + 
                      " Blue:" + string(blue_touches_edge ? "Yes" : "No");
    
    putText(display_img, edge_info, Point(10, 60), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    
    // 显示检测区域信息
    string roi_info = "Detection ROI: " + to_string(gripper_roi.width) + "x" + to_string(gripper_roi.height);
    putText(display_img, roi_info, Point(10, 85), 
            FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
    
    // 更新原图像并显示
    const_cast<Mat&>(img) = display_img;
    imshow("Task Vision", img);
    waitKey(1); // 1ms延迟，实现流畅的连续帧显示
}

// 新增：颜色分类函数实现
string classify_color(double red, double green, double blue) {
    // 改进的颜色分类逻辑 - 放宽阈值并增加灰色识别
    
    // 1. 黑色检测（最暗的颜色）
    if (red < 60 && green < 60 && blue < 60) {
        return "black"; // 黑色服装
    }
    
    // 2. 白色/灰色检测（明亮的颜色）
    if (red > 180 && green > 180 && blue > 180) {
        return "white"; // 白色服装
    } else if (abs(red - green) < 30 && abs(red - blue) < 30 && abs(green - blue) < 30) {
        // RGB值相近，判断为灰色系
        if (red > 120) return "light_gray"; // 浅灰色
        else if (red > 60) return "gray"; // 灰色
        else return "dark_gray"; // 深灰色
    }
    
    // 3. 彩色检测（放宽阈值）
    
    // 红色检测：红色分量明显高于其他分量
    if (red > max(green, blue) + 30 && red > 80) {
        if (red > 120 && green < 80 && blue < 80) return "red"; // 纯红色
        else if (red > 100) return "reddish"; // 偏红色
    }
    
    // 蓝色检测：蓝色分量明显高于其他分量
    if (blue > max(red, green) + 30 && blue > 80) {
        if (blue > 120 && red < 80 && green < 80) return "blue"; // 纯蓝色
        else if (blue > 100) return "bluish"; // 偏蓝色
    }
    
    // 绿色检测：绿色分量明显高于其他分量
    if (green > max(red, blue) + 30 && green > 80) {
        if (green > 120 && red < 80 && blue < 80) return "green"; // 纯绿色
        else if (green > 100) return "greenish"; // 偏绿色
    }
    
    // 黄色检测：红色和绿色都较高，蓝色较低
    if (red > 100 && green > 100 && blue < 80) {
        if (red > 130 && green > 130) return "yellow"; // 黄色
        else return "yellowish"; // 偏黄色
    }
    
    // 粉色检测：红色较高，绿色中等，蓝色较低（针对桃红色优化）
    if (red > 70 && green > 50 && blue < 90) {
        // 桃红色特征：红色明显高于绿色，但差值不太大，蓝色较低
        if (red > 90 && abs(red - green) < 60 && red > green + 10) {
            return "pink"; // 粉色/桃红色
        }
    }
    
    // 橙色检测：红色很高，绿色中等，蓝色很低
    if (red > 120 && green > 70 && blue < 60) {
        if (red > 140 && green > 80) return "orange"; // 橙色
    }
    
    // 紫色检测：红色和蓝色都较高，绿色较低
    if (red > 80 && blue > 80 && green < 70) {
        if (red > 100 && blue > 100) return "purple"; // 紫色
    }
    
    // 棕色检测：红色和绿色中等，蓝色很低
    if (red > 70 && green > 50 && blue < 50) {
        if (red > 90 && green > 60) return "brown"; // 棕色
    }
    
    // 如果无法识别，返回未知
    return "unknown";
}

// 可视化函数：显示机械爪中盒子颜色检测区域
void visualize_gripper_color_detection(const Mat& img, const Rect& gripper_roi, const string& target_color) {
    // 检查是否应该显示颜色方框检测区域
    if (show_color_box_detection && color_box_frame_counter < COLOR_BOX_MAX_FRAMES) {
        // 创建图像副本进行绘制，避免修改原始图像
        Mat display_img = img.clone();
        
        // 绘制机械爪检测区域
        rectangle(display_img, gripper_roi, Scalar(0, 255, 0), 3); // 绿色框表示机械爪检测区域
        
        // 计算爪子中心白点位置（假设在机械爪区域中心）
        Point gripper_center(gripper_roi.x + gripper_roi.width / 2, gripper_roi.y + gripper_roi.height / 2);
        
        // 绘制7cm×7cm颜色检测区域（假设1像素=0.1cm，所以70×70像素）
        int color_roi_size = 70;
        Rect color_roi(gripper_center.x - color_roi_size/2, gripper_center.y - color_roi_size/2, 
                       color_roi_size, color_roi_size);
        
        // 确保区域在图像范围内
        color_roi = color_roi & Rect(0, 0, img.cols, img.rows);
        
        // 绘制颜色检测区域
        rectangle(display_img, color_roi, Scalar(0, 0, 255), 3); // 红色框表示颜色检测区域
        
        // 绘制爪子中心点
        circle(display_img, gripper_center, 5, Scalar(255, 255, 255), -1); // 白色中心点
        
        // 添加文字标注（使用英文避免中文显示问号问题）
        putText(display_img, "Gripper Area", Point(gripper_roi.x, gripper_roi.y - 15), 
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 0), 2);
        putText(display_img, "Color Detection Area", Point(color_roi.x, color_roi.y - 15), 
                FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 0, 255), 2);
        putText(display_img, "Target Color: " + target_color, Point(10, 30), 
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(255, 255, 255), 2);
        putText(display_img, "Gripper Center", Point(gripper_center.x + 10, gripper_center.y - 10), 
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
        
        // 添加帧数显示信息
        string frame_info = "Frame: " + to_string(color_box_frame_counter) + "/" + to_string(COLOR_BOX_MAX_FRAMES);
        putText(display_img, frame_info, Point(10, 60), 
                FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 255, 255), 1);
        
        // 显示可视化结果（窗口标题也改为英文）
        imshow("Task Vision", display_img);
        waitKey(1); // 1ms延迟，实现流畅的连续帧显示
        
        // 更新原始图像以保持一致性
        const_cast<Mat&>(img) = display_img;
        
        // 增加帧计数器
        color_box_frame_counter++;
        
        // 检查是否达到最大帧数
        if (color_box_frame_counter >= COLOR_BOX_MAX_FRAMES) {
            show_color_box_detection = false;
            ROS_INFO("[颜色方框] 颜色方框检测区域显示完成，共显示%d帧", COLOR_BOX_MAX_FRAMES);
        }
    }
}
