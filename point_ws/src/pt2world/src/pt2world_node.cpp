#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Bool.h>
#include <sensor_msgs/Image.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <iostream>
#include <string>
#include <boost/bind.hpp>
#include <deque>
#include <vector>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

// using namespace cv;
using namespace std; // for debug

class Camera {
public:
    Camera() {}
    Camera(const std::string& config_file) { readParameters(config_file); }
    cv::Point2d& undistorted();
    void getPose(const geometry_msgs::PoseStampedConstPtr& poseptr);
    void getPoint(const geometry_msgs::PointStampedConstPtr& imgptr);
    Eigen::Vector3d pixel2world(const cv::Point2d& pixelCoor, bool flag = 0);
    void readParameters(const std::string& config_file);
    void init(ros::NodeHandle& nh);
    bool run = false;
private:
    double high_3d;
    cv::Point2d pt;
    bool fisheyeFlag;
    int image_width, image_height;
    Eigen::Vector3d camera_displacement;
    Eigen::Quaternion<double> camera_pose;
    std::string sub_point_topic, sub_pose_topic, pub_topic;
    ros::Subscriber poin_sub, pose_sub;
    cv::Mat camera_intrinsic_matrix, distort_coeff_matrix; 
    cv::Mat newCamMat; // for image Calibration
    ros::Publisher world_pt_pub, drop_pub;
};

void Camera::getPoint(const geometry_msgs::PointStampedConstPtr& pointStampptr) {
    ROS_INFO("PointStamped recevied!");
    this->run = true;
    this->pt.x = pointStampptr->point.x;
    this->pt.y = pointStampptr->point.y;
}

void Camera::getPose(const geometry_msgs::PoseStampedConstPtr& poseptr) {
    ROS_INFO("PoseStamped recevied!");
    this->high_3d = poseptr->pose.position.z;
    this->camera_pose.w() = poseptr->pose.orientation.w;
    this->camera_pose.x() = poseptr->pose.orientation.x;
    this->camera_pose.y() = poseptr->pose.orientation.y;
    this->camera_pose.z() = poseptr->pose.orientation.z;

    this->camera_displacement.x() = poseptr->pose.position.x;
    this->camera_displacement.y() = poseptr->pose.position.y;
    this->camera_displacement.z() = poseptr->pose.position.z;
}

cv::Point2d& Camera::undistorted() {

    std::vector<cv::Point2d> src_pts {pt};
    std::vector<cv::Point2d> dst_pts;

    cv::fisheye::undistortPoints(
        src_pts, dst_pts, camera_intrinsic_matrix,
        distort_coeff_matrix,
        cv::noArray(),
        newCamMat);
    
    ROS_INFO("The undistorted pt at (%.5f, %.5f)", dst_pts[0].x, dst_pts[0].y);
    return dst_pts[0]; // 不要返回局部变量的引用
}

Eigen::Vector3d Camera::pixel2world(const cv::Point2d& pixelCoor, bool flag) {
    Eigen::Vector3d pt = Eigen::Vector3d{pixelCoor.x, pixelCoor.y, 1};
    Eigen::Matrix3d intrinsic = (Eigen::Matrix3d() << this->camera_intrinsic_matrix.at<double>(0, 0), 0, this->camera_intrinsic_matrix.at<double>(0, 3), 
                                                      0, this->camera_intrinsic_matrix.at<double>(1, 1), this->camera_intrinsic_matrix.at<double>(1, 3), 
                                                      0, 0, 1 ).finished();

    Eigen::Vector3d cam_pt = intrinsic.inverse() * high_3d * pt;

    Eigen::Matrix3d cam2body = flag ? (Eigen::Matrix3d() << 0, 0, 1, -1, 0, 0, 0, -1, 0).finished() : (Eigen::Matrix3d() << 0, -1, 0, -1, 0, 0, 0, 0, -1 ).finished(); // 前视： 下视

    Eigen::Vector3d body_pt = cam2body * cam_pt;

    Eigen::Vector3d world_pt = this->camera_pose * body_pt + this->camera_displacement;

    geometry_msgs::Point pub_pt;
    pub_pt.x = world_pt.x(); 
    pub_pt.y = world_pt.y(); 
    pub_pt.z = world_pt.z(); 
    this->world_pt_pub.publish(pub_pt);

    ROS_INFO("In world coordinate pt at (%.5f, %.5f, %.5f)", world_pt.x(), world_pt.y(), world_pt.z());

    return world_pt;
}

void Camera::readParameters(const std::string& config_file) {

    FILE *fh = fopen(config_file.c_str(),"r");
    if(fh == NULL) {
        ROS_WARN("config_file dosen't exist; wrong config_file path");
        ROS_BREAK();
        return;          
    }
    fclose(fh);

    cv::FileStorage fsread(config_file, cv::FileStorage::READ);

    fsread["camera_matrix"] >> this->camera_intrinsic_matrix;

    fsread["distortion_coefficients"] >> this->distort_coeff_matrix;
    image_width = fsread["image_width"];
    image_height = fsread["image_height"];

    fsread["fisheye_model"] >> this->fisheyeFlag;

    fsread["sub_point_topic"] >> this->sub_point_topic;
    fsread["sub_pose_topic"] >> this->sub_pose_topic;
    fsread["pub_world_point_topic"] >> this->pub_topic;

    if (camera_intrinsic_matrix.empty() || distort_coeff_matrix.empty()) {
            ROS_INFO("camera init failed ... ");    
    } else {
        ROS_INFO("camera init successed");
    }

    ROS_INFO("image image_width is %d, image_height is %d", image_width, image_height);
    ROS_INFO("sub_point_topic is %s. ", this->sub_point_topic.c_str());
    ROS_INFO("sub_pose_topic is %s. ", this->sub_pose_topic.c_str());
    ROS_INFO("pub_topic is %s. ", this->pub_topic.c_str());

    fsread.release();
}

void Camera::init(ros::NodeHandle& nh) {

    cv::Mat map1, map2;

    if (this->fisheyeFlag) {
        // 估计新的相机内参矩阵,无畸变后的
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
                camera_intrinsic_matrix, distort_coeff_matrix, 
                cv::Size(image_width, image_height), cv::Matx33d::eye(), 
                newCamMat, 1
            );
        cv::fisheye::initUndistortRectifyMap(
                camera_intrinsic_matrix, distort_coeff_matrix, 
                cv::Matx33d::eye(), newCamMat, 
                cv::Size(image_width, image_height), CV_16SC2, 
                map1, map2
            );
    } else {
        cv::Mat new_camera_matric = cv::getOptimalNewCameraMatrix(
                camera_intrinsic_matrix, distort_coeff_matrix, 
                cv::Size(image_width, image_height), 1
            );
        cv::initUndistortRectifyMap(
                camera_intrinsic_matrix, distort_coeff_matrix, 
                cv::Mat(), new_camera_matric, 
                cv::Size(image_width, image_height), CV_16SC2, 
                map1, map2
            );
    }

    std::cout << "new camera matrix is\n" << newCamMat << std::endl;

    assert(map1.size().area() > 0 && map2.size().area() > 0);
    ROS_INFO("init successed.");

    poin_sub = nh.subscribe<geometry_msgs::PointStamped>(this->sub_point_topic, 10, boost::bind(&Camera::getPoint, this, _1));
    pose_sub = nh.subscribe<geometry_msgs::PoseStamped>(this->sub_pose_topic, 10, boost::bind(&Camera::getPose, this, _1));
    world_pt_pub = nh.advertise<geometry_msgs::Point>(this->pub_topic, 10);
    ROS_INFO("waiting for message....... ");
}

geometry_msgs::Point calcu_stand_deviation(const deque<Eigen::Vector3d>& points) {
    geometry_msgs::Point mean_point;

    for(auto& point : points) {
        mean_point.x += point.x();
        mean_point.y += point.y();
        mean_point.z += point.z();
    }

    mean_point.x /= points.size();
    mean_point.y /= points.size();
    mean_point.z /= points.size();

    geometry_msgs::Point temp;
    for(const Eigen::Vector3d& point: points) {
        temp.x += pow((point.x() - mean_point.x), 2);
        temp.y += pow((point.y() - mean_point.y), 2);
        temp.z += pow((point.z() - mean_point.z), 2);
    }

    temp.x /= points.size() - 1;
    temp.y /= points.size() - 1;
    temp.z /= points.size() - 1;

    temp.x = sqrt(temp.x); temp.y = sqrt(temp.y); temp.z = sqrt(temp.z);
    return temp;
}

void judgeLanding(const deque<Eigen::Vector3d>& points, ros::Publisher& drop_pub) {
    geometry_msgs::Point temp = calcu_stand_deviation(points);
    
    if(abs(temp.x) < 0.01 &&  abs(temp.y) < 0.01 && abs(temp.z) < 0.01) {
        std_msgs::Bool drop;
        drop.data = 1; // drop
        drop_pub.publish(drop);
    }
}

int main(int argc, char** argv) {

    ros::init(argc, argv, "undistort_node");

    ros::NodeHandle nh;

    std::deque<Eigen::Vector3d> points;

    ros::Publisher drop_pub = nh.advertise<std_msgs::Bool>("/drop_flag", 10);

    if(argc != 2) {
        printf("please intput: rosrun undistort undistort_node [config file] \n"
               "for example: rosrun undistort undistort_node "
               "~/kalibr_ws/config/mono_config.yaml \n");
        return 1;
    }

    Camera cam = Camera();
    cam.readParameters(argv[1]);
    cam.init(nh);

    ros::Rate hz(100);

    while(ros::ok()) {

        if (cam.run) {
            cv::Point2d pt = cam.undistorted();
            Eigen::Vector3d world_pt = cam.pixel2world(pt);
            points.push_back(world_pt);
            if (points.size() > 50) points.pop_front();
            judgeLanding(points, drop_pub);
        }

        ros::spinOnce();
        hz.sleep();
    }

    return 0;
}
