clear;clc;
%% 读取数据
bag1 = rosbag('xiaolu_fastlio_sloam.bag');
bag2 = rosbag('xiaolu_aloam.bag');
bag3 = rosbag('xiaolu_sloam_ori.bag');

fastlio_odom_msgs = select(bag1,'Topic','/Odometry');
sloam_odom_msgs = select(bag1,'Topic','/sloam/debug/odom');
rtk_msgs = select(bag1,'Topic','/chattergps');
aloam_msgs = select(bag2,'Topic','/aft_mapped_to_init');
sloam_ori_msgs = select(bag3,'Topic','/sloam/debug/odom');

fastlio_odom_data = readMessages(fastlio_odom_msgs,'DataFormat','struct');
sloam_odom_data = readMessages(sloam_odom_msgs,'DataFormat','struct');
rtk_data = readMessages(rtk_msgs,'DataFormat','struct');
aloam_data = readMessages(aloam_msgs,'DataFormat','struct');
sloam_ori_data = readMessages(sloam_ori_msgs,'DataFormat','struct');

fastlio_odom = [];
fastlio_time = [];
sloam_odom = [];
sloam_time = [];
rtk_odom_llh = [];
rtk_odom_enu = [];
rtk_odom = [];
rtk_time = [];
aloam_odom = [];
aloam_time = [];
sloam_ori_odom = [];
sloam_ori_time = [];
fastlio_end = zeros(1,3);
sloam_end = zeros(1,3);
aloam_end = zeros(1,3);
earth_radius = 6378137.0;
rtk_time_offset = 200;
fastlio_time_offset = 0;

approxtime = fastlio_odom_data{1}.Header.Stamp.Sec;
baseTime =(fastlio_odom_data{1}.Header.Stamp.Sec-approxtime)*1000 + fastlio_odom_data{1}.Header.Stamp.Nsec/1e6;

for i = 1:length(rtk_data)
    rtk_time(i) = (rtk_data{i}.Header.Stamp.Sec-approxtime)*1000 + rtk_data{i}.Header.Stamp.Nsec/1e6 - rtk_time_offset;
    rtk_odom_llh(i,1) = deg2rad(rtk_data{i}.Effort(1));
    rtk_odom_llh(i,2) = deg2rad(rtk_data{i}.Effort(2));
    rtk_odom_llh(i,3) = deg2rad(rtk_data{i}.Effort(3));
    rtk_valid(i) = rtk_data{i}.Effort(7);
end

for i = 1:length(rtk_odom_llh)
    rtk_odom_enu(i,1) = earth_radius*cos(rtk_odom_llh(i,2))*sin(rtk_odom_llh(i,1)-rtk_odom_llh(1,1));
    rtk_odom_enu(i,2) = earth_radius*(sin(rtk_odom_llh(i,2))*cos(rtk_odom_llh(1,2))-cos(rtk_odom_llh(i,2))*sin(rtk_odom_llh(1,2))*cos(rtk_odom_llh(i,1)-rtk_odom_llh(1,1)));
    rtk_odom_enu(i,3) = rtk_odom_llh(i,3)-rtk_odom_llh(1,3);
end

% 真值和估计值之间的旋转
% 绕Z轴旋转角度
theta_rotation = deg2rad(174.4); 

% 构建绕Z轴的旋转矩阵
R = [cos(theta_rotation) -sin(theta_rotation) 0;
     sin(theta_rotation) cos(theta_rotation) 0;
     0 0 1];

% 将轨迹中的每个点绕Z轴旋转
rtk_odom = (R * rtk_odom_enu')';

for i = 1:length(fastlio_odom_data)
    fastlio_time(i) = (fastlio_odom_data{i}.Header.Stamp.Sec-approxtime)*1000 + fastlio_odom_data{i}.Header.Stamp.Nsec/1e6 - baseTime - fastlio_time_offset;
    fastlio_odom(i,1) = fastlio_odom_data{i}.Pose.Pose.Position.X;
    fastlio_odom(i,2) = fastlio_odom_data{i}.Pose.Pose.Position.Y;
    fastlio_odom(i,3) = fastlio_odom_data{i}.Pose.Pose.Position.Z;
    fastlio_end(1,1) = fastlio_odom_data{i}.Pose.Pose.Position.X;
    fastlio_end(1,2) = fastlio_odom_data{i}.Pose.Pose.Position.Y;
    fastlio_end(1,3) = fastlio_odom_data{i}.Pose.Pose.Position.Z;
end

for i = 1:length(sloam_odom_data)
    sloam_time(i) = (sloam_odom_data{i}.Header.Stamp.Sec-approxtime)*1000 + sloam_odom_data{i}.Header.Stamp.Nsec/1e6 - baseTime;
    sloam_odom(i,1) = sloam_odom_data{i}.Pose.Pose.Position.X;
    sloam_odom(i,2) = sloam_odom_data{i}.Pose.Pose.Position.Y;
    sloam_odom(i,3) = sloam_odom_data{i}.Pose.Pose.Position.Z;
    sloam_end(1,1) = sloam_odom_data{i}.Pose.Pose.Position.X;
    sloam_end(1,2) = sloam_odom_data{i}.Pose.Pose.Position.Y;
    sloam_end(1,3) = sloam_odom_data{i}.Pose.Pose.Position.Z;
end

for i = 1:length(aloam_data)
    aloam_time(i) = (aloam_data{i}.Header.Stamp.Sec-approxtime)*1000 + aloam_data{i}.Header.Stamp.Nsec/1e6 - baseTime;
    aloam_odom(i,1) = aloam_data{i}.Pose.Pose.Position.X;
    aloam_odom(i,2) = aloam_data{i}.Pose.Pose.Position.Y;
    aloam_odom(i,3) = aloam_data{i}.Pose.Pose.Position.Z;
    aloam_end(1,1) = aloam_data{i}.Pose.Pose.Position.X;
    aloam_end(1,2) = aloam_data{i}.Pose.Pose.Position.Y;
    aloam_end(1,3) = aloam_data{i}.Pose.Pose.Position.Z;
end

for i = 1:length(sloam_ori_data)
    sloam_ori_time(i) = (sloam_ori_data{i}.Header.Stamp.Sec-approxtime)*1000 + sloam_ori_data{i}.Header.Stamp.Nsec/1e6 - baseTime;
    sloam_ori_odom(i,1) = sloam_ori_data{i}.Pose.Pose.Position.X;
    sloam_ori_odom(i,2) = sloam_ori_data{i}.Pose.Pose.Position.Y;
    sloam_ori_odom(i,3) = sloam_ori_data{i}.Pose.Pose.Position.Z;
    sloam_ori_end(1,1) = sloam_ori_data{i}.Pose.Pose.Position.X;
    sloam_ori_end(1,2) = sloam_ori_data{i}.Pose.Pose.Position.Y;
    sloam_ori_end(1,3) = sloam_ori_data{i}.Pose.Pose.Position.Z;
end

%% 计算漂移率
fastlio_drift = sqrt(fastlio_end(1,1)*fastlio_end(1,1) + fastlio_end(1,2)*fastlio_end(1,2) + fastlio_end(1,3)*fastlio_end(1,3))/sum(sqrt(sum(diff(fastlio_odom, 1, 1).^2, 2)));
sloam_drift = sqrt(sloam_end(1,1)*sloam_end(1,1) + sloam_end(1,2)*sloam_end(1,2) + sloam_end(1,3)*sloam_end(1,3))/sum(sqrt(sum(diff(sloam_odom, 1, 1).^2, 2)));
aloam_drift = sqrt(aloam_end(1,1)*aloam_end(1,1) + aloam_end(1,2)*aloam_end(1,2) + aloam_end(1,3)*aloam_end(1,3))/sum(sqrt(sum(diff(aloam_odom, 1, 1).^2, 2)));
sloam_ori_drift = sqrt(sloam_ori_end(1,1)*sloam_ori_end(1,1) + sloam_ori_end(1,2)*sloam_ori_end(1,2) + sloam_ori_end(1,3)*sloam_ori_end(1,3))/sum(sqrt(sum(diff(sloam_ori_odom, 1, 1).^2, 2)));

%% 计算rmse
[fastlio_rmse_x, fastlio_rmse_y, fastlio_rmse_z, fastlio_rmse_xy, fastlio_rmse_xyz, error_fastlio] = compute_rmse(rtk_time, rtk_odom, rtk_valid, fastlio_time, fastlio_odom);
[sloam_rmse_x, sloam_rmse_y, sloam_rmse_z, sloam_rmse_xy, sloam_rmse_xyz, error_sloam] = compute_rmse(rtk_time, rtk_odom, rtk_valid, sloam_time, sloam_odom);
[aloam_rmse_x, aloam_rmse_y, aloam_rmse_z, aloam_rmse_xy, aloam_rmse_xyz, error_aloam] = compute_rmse(rtk_time, rtk_odom, rtk_valid, aloam_time, aloam_odom);
[sloam_ori_rmse_x, sloam_ori_rmse_y, sloam_ori_rmse_z, sloam_ori_rmse_xy, sloam_ori_rmse_xyz, error_sloam_ori] = compute_rmse(rtk_time, rtk_odom, rtk_valid, sloam_ori_time, sloam_ori_odom);

%% 三维轨迹图
figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
plot3(rtk_odom(:,1), rtk_odom(:,2), rtk_odom(:,3), 'k--', 'DisplayName', 'RTK', 'Linewidth', 2);
hold on
plot3(aloam_odom(:,1), aloam_odom(:,2), aloam_odom(:,3), 'c', 'DisplayName', 'A-LOAM', 'Linewidth', 2);
hold on
plot3(fastlio_odom(:,1), fastlio_odom(:,2), fastlio_odom(:,3), 'g', 'DisplayName', 'FAST-LIO', 'Linewidth', 2);
hold on
plot3(sloam_ori_odom(:,1), sloam_ori_odom(:,2), sloam_ori_odom(:,3), 'r', 'DisplayName', 'SLOAM', 'Linewidth', 2);
hold on
plot3(sloam_odom(:,1), sloam_odom(:,2), sloam_odom(:,3), 'b', 'DisplayName', 'proposed', 'Linewidth', 2);
legend('Location', 'northeast');
xlabel('\itx\rm/m');
ylabel('\ity\rm/m');
zlabel('\itz\rm/m');
set(gca, 'Fontsize', 24);
grid on
%% 轨迹俯视图
%figure;
figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
plot(rtk_odom(:,1), rtk_odom(:,2), 'k--', 'DisplayName', 'RTK');
hold on
plot(aloam_odom(:,1), aloam_odom(:,2), 'c', 'DisplayName', 'A-LOAM');
hold on
plot(fastlio_odom(:,1), fastlio_odom(:,2), 'g', 'DisplayName', 'FAST-LIO');
hold on
plot(sloam_ori_odom(:,1), sloam_ori_odom(:,2), 'r', 'DisplayName', 'SLOAM');
hold on
plot(sloam_odom(:,1), sloam_odom(:,2), 'b', 'DisplayName', 'proposed');
legend('Location', 'northeast');
xlabel('\itx\rm/m');
ylabel('\ity\rm/m');
set(gca, 'Fontsize', 24);
grid on

%% 三轴轨迹图
figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
plot(rtk_time/1000, rtk_odom(:,1), 'k--', 'DisplayName', 'RTK');
hold on
plot(aloam_time/1000, aloam_odom(:,1), 'c', 'DisplayName', 'A-LOAM');
hold on
plot(fastlio_time/1000, fastlio_odom(:,1), 'g', 'DisplayName', 'FAST-LIO');
hold on
plot(sloam_ori_time/1000, sloam_ori_odom(:,1), 'r', 'DisplayName', 'SLOAM');
hold on
plot(sloam_time/1000, sloam_odom(:,1), 'b', 'DisplayName', 'proposed');
legend('Location', 'northeast');
xlabel('\itt\rm/s');
ylabel('\itx\rm/m');
set(gca, 'Fontsize', 24);
grid on

figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
plot(rtk_time/1000, rtk_odom(:,2), 'k--', 'DisplayName', 'RTK');
hold on
plot(aloam_time/1000, aloam_odom(:,2), 'c', 'DisplayName', 'A-LOAM');
hold on
plot(fastlio_time/1000, fastlio_odom(:,2), 'g', 'DisplayName', 'FAST-LIO');
hold on
plot(sloam_ori_time/1000, sloam_ori_odom(:,2), 'r', 'DisplayName', 'SLOAM');
hold on
plot(sloam_time/1000, sloam_odom(:,2), 'b', 'DisplayName', 'proposed');
legend('Location', 'northeast');
xlabel('\itt\rm/s');
ylabel('\ity\rm/m');
set(gca, 'Fontsize', 24);
grid on
 
figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
plot(rtk_time/1000, rtk_odom(:,3), 'k--', 'DisplayName', 'RTK');
hold on
plot(aloam_time/1000, aloam_odom(:,3), 'c', 'DisplayName', 'A-LOAM');
hold on
plot(fastlio_time/1000, fastlio_odom(:,3), 'g', 'DisplayName', 'FAST-LIO');
hold on
plot(sloam_ori_time/1000, sloam_ori_odom(:,3), 'r', 'DisplayName', 'SLOAM');
hold on
plot(sloam_time/1000, sloam_odom(:,3), 'b', 'DisplayName', 'proposed');
legend('Location', 'northeast');
xlabel('\itt\rm/s');
ylabel('\itz\rm/m');
set(gca, 'Fontsize', 24);
grid on

%% 三轴误差图
%% xy水平误差
figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
plot(error_aloam(:,1)/1000, error_aloam(:,5), 'c', 'DisplayName', 'A-LOAM', 'Linewidth', 2);
hold on
plot(error_fastlio(:,1)/1000, error_fastlio(:,5), 'g', 'DisplayName', 'FAST-LIO', 'Linewidth', 2);
hold on
plot(error_sloam_ori(:,1)/1000, error_sloam_ori(:,5), 'r', 'DisplayName', 'SLOAM', 'Linewidth', 2);
hold on
plot(error_sloam(:,1)/1000, error_sloam(:,5), 'b', 'DisplayName', 'proposed', 'Linewidth', 2);
legend('Location', 'northeast');
set(gca, 'Fontsize', 24);
truncAxis('X',[270,360]);
xlabel('\itt\rm/s');
ylabel('\rm\Delta\itxy\rm/m');

%% z误差
figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
plot(error_aloam(:,1)/1000, error_aloam(:,4), 'c', 'DisplayName', 'A-LOAM', 'Linewidth', 2);
hold on
plot(error_fastlio(:,1)/1000, error_fastlio(:,4), 'g', 'DisplayName', 'FAST-LIO', 'Linewidth', 2);
hold on
plot(error_sloam_ori(:,1)/1000, error_sloam_ori(:,4), 'r', 'DisplayName', 'SLOAM', 'Linewidth', 2);
hold on
plot(error_sloam(:,1)/1000, error_sloam(:,4), 'b', 'DisplayName', 'proposed', 'Linewidth', 2);
legend('Location', 'northeast');
set(gca, 'Fontsize', 24);
truncAxis('X',[270,360]);
xlabel('\itt\rm/s');
ylabel('\rm\Delta\itz\rm/m');

%% x误差
figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
plot(error_aloam(:,1)/1000, error_aloam(:,2), 'c', 'DisplayName', 'A-LOAM', 'Linewidth', 2);
hold on
plot(error_fastlio(:,1)/1000, error_fastlio(:,2), 'g', 'DisplayName', 'FAST-LIO', 'Linewidth', 2);
hold on
plot(error_sloam_ori(:,1)/1000, error_sloam_ori(:,2), 'r', 'DisplayName', 'SLOAM', 'Linewidth', 2);
hold on
plot(error_sloam(:,1)/1000, error_sloam(:,2), 'b', 'DisplayName', 'proposed', 'Linewidth', 2);
legend('Location', 'northeast');
set(gca, 'Fontsize', 24);
truncAxis('X',[270,360]);
xlabel('\itt\rm/s');
ylabel('\rm\Delta\itx\rm/m');

%% y误差
figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
plot(error_aloam(:,1)/1000, error_aloam(:,3), 'c', 'DisplayName', 'A-LOAM', 'Linewidth', 2);
hold on
plot(error_fastlio(:,1)/1000, error_fastlio(:,3), 'g', 'DisplayName', 'FAST-LIO', 'Linewidth', 2);
hold on
plot(error_sloam_ori(:,1)/1000, error_sloam_ori(:,3), 'r', 'DisplayName', 'SLOAM', 'Linewidth', 2);
hold on
plot(error_sloam(:,1)/1000, error_sloam(:,3), 'b', 'DisplayName', 'proposed', 'Linewidth', 2);
legend('Location', 'northeast');
set(gca, 'Fontsize', 24);
truncAxis('X',[270,360]);
xlabel('\itt\rm/s');
ylabel('\rm\Delta\ity\rm/m');

%% 误差箱线图绘制
data1 = error_aloam(:,5)';
data2 = error_fastlio(:,5)';
data3 = error_sloam_ori(:,5)';
data4 = error_sloam(:,5)';
% 创建包含数据的cell数组
data = {data1, data2, data3, data4};

% 创建组号向量和标签向量
group = [];
labels = {'A-LOAM', 'FAST-LIO', 'SLOAM', 'proposed'};

% 循环遍历每个数据集
for i = 1:numel(data)
    % 获取当前数据集的长度
    len = length(data{i});
    
    % 创建对应的组号向量和标签向量
    group = [group, i*ones(1, len)];
%     labels = [labels, repmat({['Data Set ' num2str(i)]}, 1, len)];
end

% 创建箱线图
figure('DefaultAxesFontName', 'Times New Roman', 'DefaultAxesFontSize', 16);
colors = ['c', 'g', 'r', 'b'];
h = boxplot([data{:}], group, 'Labels', labels, 'Whisker', 3, 'OutlierSize', 3, 'BoxStyle', 'outline',  'MedianStyle', 'line', 'Widths', 0.3, 'Colors', colors);
set(h,'LineWidth',3)
set(gca, 'Fontsize', 24);

% 设置图形属性
ylabel('水平误差/m');

%% 计算rmse函数
function[rmse_x, rmse_y, rmse_z, rmse_xy, rmse_xyz, error_slamtortk] = compute_rmse(rtk_time, rtk_odom, rtk_valid, slam_time, slam_odom)
k = 1;
for i = 1 : length(rtk_time)
    for j = 1 : (length(slam_time)-1)
        if (rtk_time(i)>slam_time(j) && rtk_time(i)<slam_time(j+1) && rtk_valid(i)==4)
            error_slamtortk(k,1) =  rtk_time(i);
            error_slamtortk(k,2) = (rtk_time(i)-slam_time(j))*(slam_odom(j+1,1)-slam_odom(j,1))/(slam_time(j+1)-slam_time(j)) + slam_odom(j,1) - rtk_odom(i,1);
            error_slamtortk(k,3) = (rtk_time(i)-slam_time(j))*(slam_odom(j+1,2)-slam_odom(j,2))/(slam_time(j+1)-slam_time(j)) + slam_odom(j,2) - rtk_odom(i,2);
            error_slamtortk(k,4) = (rtk_time(i)-slam_time(j))*(slam_odom(j+1,3)-slam_odom(j,3))/(slam_time(j+1)-slam_time(j)) + slam_odom(j,3) - rtk_odom(i,3);
            error_slamtortk(k,5)= sqrt(error_slamtortk(k,2)*error_slamtortk(k,2)+error_slamtortk(k,3)*error_slamtortk(k,3));
            error_slamtortk(k,6)= sqrt(error_slamtortk(k,2)*error_slamtortk(k,2)+error_slamtortk(k,3)*error_slamtortk(k,3)+error_slamtortk(k,4)*error_slamtortk(k,4));
            k = k+1;
        end
    end
end
rmse_x = rms(error_slamtortk(:,2));
rmse_y = rms(error_slamtortk(:,3));
rmse_z = rms(error_slamtortk(:,4));
rmse_xy = rms(error_slamtortk(:,5));
rmse_xyz = rms(error_slamtortk(:,6));
end


