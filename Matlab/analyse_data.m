clear;clc;
%% 读取数据
bag1 = rosbag('./deblurred/DeblurGan-V2/pose.bag');
bag2 = rosbag('./deblurred/FFTformer/pose.bag');
bag3 = rosbag('./deblurred/GT/pose.bag');
bag4 = rosbag('./deblurred/NAFNet/pose.bag');
bag5 = rosbag('./deblurred/Reformer/pose.bag');
bag6 = rosbag('./deblurred/Restormer/pose.bag');

deblurgan_odom_msgs = select(bag1, "Topic", "/pose");
fftfomer_odom_msgs = select(bag2, "Topic", "/pose");
gt_msgs = select(bag3, "Topic", "/pose"); % GroundTruth Pose
nafnet_msgs = select(bag4, "Topic", "/pose");
reformer_msgs = select(bag5, "Topic", "/pose");
restormer_msgs = select(bag6, "Topic", "/pose");

deblurgan_data = readMessages(deblurgan_odom_msgs,'DataFormat','struct');
fftfomer_data = readMessages(fftfomer_odom_msgs,'DataFormat','struct');
gt_data = readMessages(gt_msgs,'DataFormat','struct');
nafnet_data = readMessages(nafnet_msgs,'DataFormat','struct');
restormer_data = readMessages(reformer_msgs,'DataFormat','struct');




