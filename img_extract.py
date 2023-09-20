# -*- encoding: utf-8 -*-
'''
@File    :   img_extract.py
@Time    :   2023/09/01 11:32:18
@Author  :   orCate 
@Version :   1.0
@Contact :   8631143542@qq.com
'''

# here put the import lib
import roslib;  
import rosbag
import rospy
import cv2 as cv
import numpy as np
import tqdm
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from cv_bridge import CvBridgeError

path='./test/' #存放图片的位置
class ImageCreator():
   def __init__(self):
        self.bridge = CvBridge()
    #    self.cnt = 8455
        with rosbag.Bag('near.bag', 'r') as bag:   #要读取的bag文件；
            bar = tqdm.tqdm(bag.read_messages(), total=bag.get_message_count(), desc="ok!")
            for topic, msg, t in bar:
                if topic == "/iris_0/realsense/depth_camera/color/image_raw":  #rbg图像的topic；
                    try:
                        cv_image = self.bridge.imgmsg_to_cv2(msg,"bgr8")
                    except CvBridgeError as e:
                        print(e)
                    timestr = "%.6f" %  msg.header.stamp.to_sec()
                    # %.6f表示小数点后带有6位，可根据精确度需要修改；
                    image_name = timestr + ".png" #图像命名：时间戳.
                    # image_name = "{0}".format(self.cnt) + ".png"
                    # if True:
                    #     image_blue = cv_image[:, :, 0]
                    #     image_green = cv_image[:, :, 1]
                    #    image_red = cv_image[:, :, 2] 
                    #    cv_image = np.dstack((image_blue, image_red, image_green)) #bgr->gbr # opencv: BGR plt: RGB
                    
                    if cv.imwrite(path + image_name, cv_image): bar.set_description_str("ok")  #保存；: print('OK!')
                    else: bar.set_description_str("failed!")
                if topic == "深度图topic" :
                    try:
                        depth_image = self.bridge.imgmsg_to_cv2(msg, "passthrough")
                    except CvBridgeError as e:
                        print(e)

                    timestr = "%.6f" %  msg.header.stamp.to_sec()
                    # %.6f表示小数点后带有6位，可根据精确度需要修改；
                    name = timestr + ".npy" #图像命名：时间戳.
                    # name = "{0}".format(self.cnt) + ".npy"
                    np.save(path + name, depth_image)

                time.sleep(1)
                

if __name__ == '__main__':

    #rospy.init_node(PKG)

    try:
        image_creator = ImageCreator()
    except rospy.ROSInterruptException:
        pass