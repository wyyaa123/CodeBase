import os
import numpy as np

import rosbag 
import rospy
from sensor_msgs.msg import Image, Imu
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
import cv2 as cv

def ReadRGB(file_dir):
    "Here assume the name of image is the timestamp"
    file_names_left = sorted(os.listdir(file_dir))
    def get_timestamp(file_name):
        return np.float64(file_name[:-4])
    timestamps = map(get_timestamp, file_names_left)
    print("Total add %i images!"%(np.size(file_names_left)))
    return file_names_left, list(timestamps)

def ReadIMU(file_path):
    '''return IMU data and timestamp of IMU'''
    file = open(file_path, 'r')
    all = file.readlines()
    timestamp = []
    imu_data = []
    index = 0
    for f in all:
        index = index + 1
        line = f.rstrip('\n').split(' ') # here maybe ',' or ' '
        timestamp.append(line[0])
        imu_data.append(line[1:7])
    print("Total add %i imus!"%(index))
    return imu_data, timestamp

def CreateBag():

    left_image_dir = "/home/nrc/m_vins/bags/reality_test1/raw_left_clear/"
    right_image_dir = "/home/nrc/m_vins/bags/reality_test1/raw_right_clear/"
    imu_path = '/home/nrc/m_vins/bags/imu.txt'

    bag = rosbag.Bag("data.bag", 'w')
    file_names_left, imgstamp_left = ReadRGB(left_image_dir)
    file_names_right, imgstamp_right = ReadRGB(right_image_dir)
    # print(file_names_left)
    # print(timestamp)
    # imu_data, imustamp = ReadIMU(imu_path)
    # print(imu_data)
    # print(timestamp)
    print('working!')

    print(len(file_names_left))
    # print(len(imu_data))

    for i in range(len(file_names_left)):
        img = Image()
        img = CvBridge().cv2_to_imgmsg(cv.imread(left_image_dir + file_names_left[i], cv.IMREAD_GRAYSCALE))
        img.header.frame_id = "camera"
        img.header.stamp = rospy.Time(imgstamp_left[i] * 1e-9)
        img.encoding = "mono8"
        bag.write("/stereo_inertial_publisher/left/image_rect", img, img.header.stamp)

    for i in range(len(file_names_right)):
        img = Image()
        img = CvBridge().cv2_to_imgmsg(cv.imread(right_image_dir + file_names_right[i], cv.IMREAD_GRAYSCALE))
        img.header.frame_id = "camera"
        img.header.stamp = rospy.Time(imgstamp_right[i] * 1e-9)
        img.encoding = "mono8"
        bag.write("/stereo_inertial_publisher/right/image_rect", img, img.header.stamp)

    # for i in range(0, len(imu_data)):
    #     # print(i)
    #     imu = Imu()
    #     angular_v = Vector3()
    #     linear_a = Vector3()
    #     angular_v.x = float(imu_data[i][0])
    #     angular_v.y = float(imu_data[i][1])
    #     angular_v.z = float(imu_data[i][2])
    #     linear_a.x = float(imu_data[i][3])
    #     linear_a.y = float(imu_data[i][4])
    #     linear_a.z = float(imu_data[i][5])
    #     imuStamp = rospy.rostime.Time.from_sec(float(imustamp[i]) * 1e-9)  # according to the timestamp unit
    #     imu.header.stamp=imuStamp
    #     imu.angular_velocity = angular_v
    #     imu.linear_acceleration = linear_a

    #     bag.write("/camera/imu",imu,imuStamp)

    bag.close()


if __name__ == "__main__":
    # bag = rosbag.Bag("", 'w')
    # bag.write("/camera/imu", Image(), )
    CreateBag()
