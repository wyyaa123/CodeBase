#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <iostream>
#include <string>
#include <boost/bind.hpp>
#include <vector>

// using namespace cv;
using namespace std; // for debug

class Image {
public:
    Image() {}
    Image(const std::string& config_file) { readParameters(config_file); }

    void undistorted(const sensor_msgs::ImageConstPtr& imgptr);

    void readParameters(const std::string& config_file);
    void initsubs(ros::NodeHandle& nh, std::vector<ros::Subscriber>&);
private:
    cv::Size img_size; 
    int width = 0, height = 0;
    bool pub_undistort, fisheyeFlag;
    std::string sub_topic, pub_topic;
    cv::Mat camera_intrinsic_matrix, distort_coeff_matrix;

    void initCam();
};

int main(int argc, char** argv) {

    ros::init(argc, argv, "undistort_node");

    ros::NodeHandle nh;

    std::vector<ros::Subscriber> subs = std::vector<ros::Subscriber>(1, ros::Subscriber());

    if(argc != 2) {
        printf("please intput: rosrun undistort undistort_node [config file] \n"
               "for example: rosrun undistort undistort_node "
               "~/kalibr_ws/config/mono_config.yaml \n");
        return 1;
    }

    Image img = Image();
    img.readParameters(argv[1]);
    img.initsubs(nh, subs);
    // ros::Subscriber sub = nh.subscribe<sensor_msgs::Image>("/prometheus/camera/rgb/image_raw", 10, img_sub);
    ros::spin();

    return 0;
}

void Image::readParameters(const std::string& config_file) {

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

    if (camera_intrinsic_matrix.empty() || distort_coeff_matrix.empty()) {
            std::cout << "init failed ... " << std::endl;    
    } else {
        std::cout << "init successed" << std::endl;
    }

    // std::cout << "camera_matrix = " << camera_intrinsic_matrix << std::endl;
    // std::cout << "distort_coeff_matrix = " << distort_coeff_matrix << std::endl;

    fsread["fisheye_model"] >> this->fisheyeFlag;
    // std::cout << "fisheyeFlag == " << fisheyeFlag << std::endl;

    int width = fsread["image_width"], height = fsread["image_height"];

    this->img_size = cv::Size(width, height);
    // std::cout << "img_width is " << width << "\nimg_height is " << height << std::endl;

    fsread["pub_undistort"] >> this->pub_undistort;
    fsread["sub_img_topic"] >> this->sub_topic;
    fsread["pub_img_topic"] >> this->pub_topic;

    ROS_INFO("sub_img_topic is %s. ", this->sub_topic.c_str());
    ROS_INFO("pub_img_topic is %s. ", this->pub_topic.c_str());
    ROS_INFO("waiting for image....... ");

    fsread.release();
}

void Image::undistorted(const sensor_msgs::ImageConstPtr& imgptr) {

    ROS_INFO("image recevied!");

    cv::Mat map1, map2;
    cv::Mat newCamMat;

    if (this->fisheyeFlag) {
        // 估计新的相机内参矩阵,无畸变后的
        cv::fisheye::estimateNewCameraMatrixForUndistortRectify(
        camera_intrinsic_matrix, distort_coeff_matrix, img_size,
        cv::Matx33d::eye(), newCamMat, 1);
        cv::fisheye::initUndistortRectifyMap(camera_intrinsic_matrix, distort_coeff_matrix, 
        cv::Matx33d::eye(), newCamMat, img_size,
        CV_16SC2, map1, map2);
    } else {
        cv::Mat new_camera_matric = 
        cv::getOptimalNewCameraMatrix(camera_intrinsic_matrix, distort_coeff_matrix, 
        img_size, 1);
        cv::initUndistortRectifyMap(
        camera_intrinsic_matrix, distort_coeff_matrix, cv::Mat(),
        new_camera_matric, img_size, CV_16SC2, map1, map2);
        // new_camera_matric.copyTo(camera_intrinsic_matrix);
    }

    // 图像去畸变
    cv::Mat cam_im = cv_bridge::toCvCopy(imgptr, "bgr8")->image;
    
    cv::Mat correct_image;
    cv::remap(cam_im, correct_image, map1, map2, cv::INTER_LINEAR);

    cv::Mat undistort_im;
    cv::fisheye::undistortImage(
        cam_im, undistort_im,
        camera_intrinsic_matrix,
        distort_coeff_matrix,
        newCamMat,
        cam_im.size());

    ROS_INFO("raw_image %s", cv::imwrite("data/raw_image.png", cam_im) ? "raw_img saved.": "raw_img save faild");

    // std::cout << (cv::imwrite("data/remap.jpg", correct_image) ? 
    //                          "remap_img saved.": "remap_img save faild")
    //                          << std::endl;

    // std::cout << (cv::imwrite("data/undistort_img.jpg",undistort_im) ?
    //                          "undistort_img saved.": "undistort_img save faild")
    //                          << std::endl;

    // cv::Mat substrct_im;
    // cv::subtract(undistort_im, correct_image, substrct_im);
    // std::cout << (cv::imwrite("data/substrct_img.jpg",substrct_im) ?
    //                           "substrct_img saved.": "substrct_img save faild")
    //                           << std::endl;


    //某个点去畸变
    std::vector<cv::Point2f> src_pts {cv::Point2f(500,500)};
    std::vector<cv::Point2f> dst_pts;

    cv::fisheye::undistortPoints(
        src_pts, dst_pts, camera_intrinsic_matrix,
        distort_coeff_matrix,
        cv::noArray(),
        newCamMat);
        
    std::cout << "dst_pts= " << dst_pts[0] << std::endl;
}

void Image::initsubs(ros::NodeHandle& nh, std::vector<ros::Subscriber>& subs) {
    ROS_INFO("init successed.");
    subs[0] = nh.subscribe<sensor_msgs::Image>(this->sub_topic, 10, boost::bind(&Image::undistorted, this, _1));
}

