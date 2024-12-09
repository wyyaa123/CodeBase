% I = im2double(imread('./Lenna.png'));           %读入清晰原图像
% figure(1);
% imshow(I,[]);         %显示原图像
% title('原图像');
% 
% PSF = fspecial('motion', 100, 0);   %建立二维仿真线性运动滤波器PSF
% MF = imfilter(I,PSF,'conv', 'circular');  %用PSF产生退化图像
% imwrite(MF, "C:/Users/86311/Desktop/CodeBase/Python/restored/blur.png")
% noise = imnoise(zeros(size(I)),'gaussian',0,0.001);   %产生高斯噪声
% MFN = imadd(MF,noise);
% 
% 
% figure(2); 
% imshow(MFN,[]);         %显示模糊噪声后的图像
% title('运动模糊图像');
% 
% 
% NSR = sum(noise(:).^2)/sum(MFN(:).^2);  %计算信噪比
% figure(3);
% wnr = deconvwnr(MFN,PSF,NSR);
% imshow(wnr,[]);     %显示维纳滤波复原图像
% title('维纳滤波复原');
% imwrite(wnr, "C:/Users/86311/Desktop/CodeBase/Python/restored/wnr.png")
% 
% %NP = 0.001 * numel(I);
% reg = deconvreg(MF,PSF);
% figure(4);imshow(reg,[]);
% title('有约束的最小二乘滤波复原');
% imwrite(reg, "C:/Users/86311/Desktop/CodeBase/Python/restored/reg.png")
% 
% invert = deconvblind(MFN, PSF);
% figure(5);
% imshow(invert,[]);
% title('逆滤波复原');
% imwrite(invert, "C:/Users/86311/Desktop/CodeBase/Python/restored/invert.png")

x = [10, 20, 30 , 40, 50, 60, 70, 80, 90, 100];

P_b = [26.876, 23.106, 21.42, 20.283, 19.451, 18.795, 18.269, 17.849, 17.507, 17.219];
S_b = [0.731, 0.586, 0.566, 0.554, 0.545, 0.541, 0.538, 0.537, 0.536, 0.535];

P_i = [26.229, 25.004, 23.899, 22.875, 22.021, 21.526, 21.082, 20.399, 19.838, 19.555];
S_i = [0.538, 0.627, 0.511, 0.496, 0.485, 0.480, 0.472, 0.463, 0.456, 0.454];

P_r = [40.117, 37.877, 36.822, 35.695, 35.076, 34.987, 34.492, 33.277, 33.863, 33.457];
S_r = [0.970, 0.953, 0.943, 0.927, 0.920, 0.920, 0.912, 0.890, 0.902, 0.894];

P_w = [16.769, 16.637, 16.917, 17.16, 17.458, 17.771, 17.990, 18.183, 18.321, 18.194];
S_w = [0.152, 0.138, 0.139, 0.138, 0.141, 0.144, 0.147, 0.149, 0.152, 0.156];

figure;
plot(x, P_b, "-o", 'LineWidth', 2, 'DisplayName', "模糊图像");
hold on;
plot(x, P_i, "-o", 'LineWidth', 2, 'DisplayName', "维纳滤波");
hold on;
plot(x, P_r, "-o", 'LineWidth', 2, 'DisplayName', "约束最小二乘滤波");
hold on;
plot(x, P_w, "-o", 'LineWidth', 2, 'DisplayName', "逆滤波");
hold off;

xlabel('模糊尺度');
ylabel('PSNR');
legend show;
grid on;

figure;
plot(x, S_b, "-o", 'LineWidth', 2, 'DisplayName', "模糊图像");
hold on;
plot(x, S_i, "-o", 'LineWidth', 2, 'DisplayName', "维纳滤波");
hold on;
plot(x, S_r, "-o", 'LineWidth', 2, 'DisplayName', "约束最小二乘滤波");
hold on;
plot(x, S_w, "-o", 'LineWidth', 2, 'DisplayName', "逆滤波");
hold off;

xlabel('模糊尺度');
ylabel('SSIM');
legend show;
grid on;







