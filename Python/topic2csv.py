import rospy
import pandas as pd
from nav_msgs.msg import Odometry as _Odometry

def Odometry_callback(Pose: _Odometry):
    
    global data

    pose = {
        "timestamp": [Pose.header.stamp.to_sec()], 
        "P_x": [Pose.pose.pose.position.x],
        "P_y": [Pose.pose.pose.position.y],
        "P_z": [Pose.pose.pose.position.z],
        "q_w": [Pose.pose.pose.orientation.w],
        "q_x": [Pose.pose.pose.orientation.x],
        "q_y": [Pose.pose.pose.orientation.y],
        "q_z": [Pose.pose.pose.orientation.z],
    }

    data = pd.concat([data, pd.DataFrame(pose)], axis=0)

if __name__ == "__main__":

    
    data = pd.DataFrame()

    rospy.init_node("listener_p")

    # img_sub = rospy.Subscriber("chatter", String, doMsg, queue_size=10)
    # imu_sub = rospy.Subscriber("chatter", String, doMsg, queue_size=10)
    odometry_sub = rospy.Subscriber("/vins_estimator/odometry", _Odometry, Odometry_callback, queue_size=10)

    rospy.spin() 

    data.to_csv("./bluredPose.csv", index=False) # after Ctrl+C 
