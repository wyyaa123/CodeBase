clear;clc;
%% read bag
bag1 = rosbag('./deblurred/Restormer/pose.bag');
bag2 = rosbag('./deblurred/FFTformer/pose.bag');
bag3 = rosbag('./deblurred/GT/pose.bag');
bag4 = rosbag('./deblurred/NAFNet/pose.bag');
bag5 = rosbag('./deblurred/Reformer/pose.bag');
% bag6 = rosbag('./deblurred/DeblurGan-V2/pose.bag');

restormer_odom_msgs = select(bag1, "Topic", "/pose");
fftfomer_odom_msgs = select(bag2, "Topic", "/pose");
gt_msgs = select(bag3, "Topic", "/pose"); % GroundTruth Pose
nafnet_msgs = select(bag4, "Topic", "/pose");
reformer_msgs = select(bag5, "Topic", "/pose");
% deblurgan_msgs = select(bag6, "Topic", "/pose");

restormer_odom_data = readMessages(restormer_odom_msgs, 'DataFormat', 'struct');
fftfomer_odom_data = readMessages(fftfomer_odom_msgs, 'DataFormat', 'struct');
gt_odom_data = readMessages(gt_msgs, 'DataFormat', 'struct');
nafnet_odom_data = readMessages(nafnet_msgs, 'DataFormat', 'struct');
reformer_odom_data = readMessages(reformer_msgs, 'DataFormat','struct');
% deblurgan_odom_data = readMessages(deblurgan_msgs, 'DataFormat', 'struct');

%% prepare data
restormer_odom = [];
restormer_time = [];

fftfomer_odom = [];
fftfomer_time = [];

gt_odom = [];
gt_time = [];

nafnet_odom = [];
nafnet_time = [];

reformer_odom = [];
reformer_time = [];


for i = 1:length(restormer_odom_data)
    restormer_time(i) = double(restormer_odom_data{i}.Header.Stamp.Sec) + double(restormer_odom_data{1}.Header.Stamp.Nsec) * 1e-9;
    restormer_odom(i, 1) = restormer_odom_data{i}.Pose.Pose.Position.X;
    restormer_odom(i, 2) = restormer_odom_data{i}.Pose.Pose.Position.Y;
    restormer_odom(i, 3) = restormer_odom_data{i}.Pose.Pose.Position.Z;
end

for i = 1:length(fftfomer_odom_data)
    fftfomer_time(i) = double(fftfomer_odom_data{i}.Header.Stamp.Sec) + double(fftfomer_odom_data{1}.Header.Stamp.Nsec) * 1e-9;
    fftfomer_odom(i, 1) = fftfomer_odom_data{i}.Pose.Pose.Position.X;
    fftfomer_odom(i, 2) = fftfomer_odom_data{i}.Pose.Pose.Position.Y;
    fftfomer_odom(i, 3) = fftfomer_odom_data{i}.Pose.Pose.Position.Z;
end

for i = 1:length(gt_odom_data)
    gt_time(i) = double(gt_odom_data{i}.Header.Stamp.Sec) + double(gt_odom_data{1}.Header.Stamp.Nsec) * 1e-9;
    gt_odom(i, 1) = gt_odom_data{i}.Pose.Pose.Position.X;
    gt_odom(i, 2) = gt_odom_data{i}.Pose.Pose.Position.Y;
    gt_odom(i, 3) = gt_odom_data{i}.Pose.Pose.Position.Z;
end

for i = 1:length(nafnet_odom_data)
    nafnet_time(i) = double(nafnet_odom_data{i}.Header.Stamp.Sec) + double(nafnet_odom_data{1}.Header.Stamp.Nsec) * 1e-9;
    nafnet_odom(i, 1) = nafnet_odom_data{i}.Pose.Pose.Position.X;
    nafnet_odom(i, 2) = nafnet_odom_data{i}.Pose.Pose.Position.Y;
    nafnet_odom(i, 3) = nafnet_odom_data{i}.Pose.Pose.Position.Z;
end

for i = 1:length(reformer_odom_data)
    reformer_time(i) = double(reformer_odom_data{i}.Header.Stamp.Sec) + double(reformer_odom_data{1}.Header.Stamp.Nsec) * 1e-9;
    reformer_odom(i, 1) = reformer_odom_data{i}.Pose.Pose.Position.X;
    reformer_odom(i, 2) = reformer_odom_data{i}.Pose.Pose.Position.Y;
    reformer_odom(i, 3) = reformer_odom_data{i}.Pose.Pose.Position.Z;
end

%% draw 3d trajectory picture
figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
plot3(restormer_odom(:,1), restormer_odom(:,2), restormer_odom(:,3), 'c', 'DisplayName', 'Restormer_Odom', 'Linewidth', 2);
hold on
plot3(fftfomer_odom(:,1), fftfomer_odom(:,2), fftfomer_odom(:,3), 'g', 'DisplayName', 'FFTformer_Odom', 'Linewidth', 2);
hold on
plot3(gt_odom(:,1), gt_odom(:,2), gt_odom(:,3), 'k--', 'DisplayName', 'GT', 'Linewidth', 2);
hold on
plot3(nafnet_odom(:,1), nafnet_odom(:,2), nafnet_odom(:,3), 'r', 'DisplayName', 'NAFNet_Odom', 'Linewidth', 2);
hold on
plot3(reformer_odom(:,1), reformer_odom(:,2), reformer_odom(:,3), 'b', 'DisplayName', 'proposed', 'Linewidth', 2);
legend('Location', 'northeast');
xlabel('\itx\rm/m');
ylabel('\ity\rm/m');
zlabel('\itz\rm/m');
set(gca, 'Fontsize', 24);
grid on



