#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <dirent.h>
#include <vector>
#include <string>
#include <algorithm>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include "estimator/estimator.h"
#include "utility/visualization.h"

using namespace std;   
using namespace Eigen;

Estimator estimator;

Eigen::Matrix3d c1Rc0, c0Rc1;
Eigen::Vector3d c1Tc0, c0Tc1;

vector<string> ReadImg(string dir_path);

int main(int argc, char** argv) {

    ros::init(argc, argv, "vins_estimator");
	ros::NodeHandle n("~");
	ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

	ros::Publisher pubLeftImage = n.advertise<sensor_msgs::Image>("/leftImage",1000);
	ros::Publisher pubRightImage = n.advertise<sensor_msgs::Image>("/rightImage",1000);

	if(argc != 4) {
		printf("please intput: rosrun vins reality_odom_test [config file] [datas folder] \n"
			   "for example: rosrun vins reality_odom_test " 
			   "~/m_vins/src/vins_estimator/config/OAK_D_640x400/oak_stereo_imu.yaml" 
			   "~/m_vins/bags/reality1_test/raw_left/" 
			   "~/m_vins/bags/reality1_test/raw_right/ \n");
		return 1;
	}

	string config_file = argv[1];
	printf("config_file: %s\n", argv[1]);
    vector<string> left_imgs = ReadImg(argv[2]);
    vector<string> right_imgs = ReadImg(argv[3]);
	printf("read sequence:\n %s\n%s\n", argv[2], argv[3]);
    printf("total have %ld left-images, %ld right-images\n", left_imgs.size(), right_imgs.size());

	readParameters(config_file);
	estimator.setParameter();
	registerPub(n);

    sort(left_imgs.begin(), left_imgs.end(), [](const string& lhs, const string& rhs) {
        return stol(lhs.substr(0, lhs.size() - 4)) < stol(rhs.substr(0, rhs.size() -4));
    });

    sort(right_imgs.begin(), right_imgs.end(), [](const string& lhs, const string& rhs) {
        return stol(lhs.substr(0, lhs.size() - 4)) < stol(rhs.substr(0, rhs.size() -4));
    });

    // printf("%s\n", left_imgs[0].c_str());
    // printf("%s\n", left_imgs[1].c_str());
    // printf("%s\n", left_imgs[2].c_str());
    // printf("%s\n", left_imgs[3].c_str());
    // printf("%s\n", left_imgs[4].c_str());

    for (size_t i = 0; i != left_imgs.size(); ++i) {
        if (ros::ok()) {
			printf("\nprocess image %d\n", int(i));
            
            string leftImagePath = string(argv[2]) + left_imgs[i] + ".png";
            string rightImagePath = string(argv[3]) + right_imgs[i] + ".png";
            ros::Time left_timestamp = ros::Time(stol(left_imgs[i].substr(0, left_imgs.size() - 4)) * 1e-9);
            ros::Time right_timestamp = ros::Time(stol(right_imgs[i].substr(0, right_imgs.size() - 4)) * 1e-9);


            cv::Mat imLeft = cv::imread(leftImagePath, CV_LOAD_IMAGE_GRAYSCALE );
			sensor_msgs::ImagePtr imLeftMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imLeft).toImageMsg();
			imLeftMsg->header.stamp = left_timestamp;
			pubLeftImage.publish(imLeftMsg);

			cv::Mat imRight = cv::imread(rightImagePath, CV_LOAD_IMAGE_GRAYSCALE );
			sensor_msgs::ImagePtr imRightMsg = cv_bridge::CvImage(std_msgs::Header(), "mono8", imRight).toImageMsg();
			imRightMsg->header.stamp = right_timestamp;
			pubRightImage.publish(imRightMsg);

			estimator.inputImage(left_timestamp.toSec(), imLeft, imRight);
			
			Eigen::Matrix<double, 4, 4> pose;
			estimator.getPoseInWorldFrame(pose);
            printf ("%f %f %f %f %f %f %f %f %f %f %f %f \n",pose(0,0), pose(0,1), pose(0,2),pose(0,3),
                                                                        pose(1,0), pose(1,1), pose(1,2),pose(1,3),
                                                                        pose(2,0), pose(2,1), pose(2,2),pose(2,3));
        }
    }

    return 0;
}

vector<string> ReadImg(string dir_path) {

    DIR* dir = opendir(dir_path.c_str());
    dirent* entry;
    vector<string> imgs = vector<string>();

    if (dir == NULL) {
        fprintf(stderr, "Failed to open directory.\n");
        return imgs;
    }

    // 遍历目录中的所有文件
    while ((entry = readdir(dir)) != NULL) {
        if (entry->d_type == DT_REG) {
            // printf("%s\n", entry->d_name);
            imgs.push_back(string(entry->d_name));
        }
    }

    if (!closedir(dir)) printf("successed closed!\n");
    else { printf("Failed closed!\n"); exit(-1); }

    return imgs;
}
