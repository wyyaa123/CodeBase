I=im2double(imread('./DJI_20240422150608_0132_Zenmuse-L1-mission.JPG'));           %读入清晰原图像
figure(1);imshow(I,[]);         %显示原图像
title('原图像');
PSF=fspecial('motion', 20, 30);   %建立二维仿真线性运动滤波器PSF
MF=imfilter(I,PSF,'conv', 'circular');  %用PSF产生退化图像
imwrite(MF, "./blur.png")
%noise=imnoise(zeros(size(I)),'gaussian',0,0.001);   %产生高斯噪声
%MFN=imadd(MF,im2uint8(noise));
figure(2);imshow(MF,[]);         %显示模糊噪声后的图像
title('运动模糊图像');
%NSR=sum(noise(:).^2)/sum(MFN(:).^2);  %计算信噪比
figure(3);
imshow(deconvwnr(MF,PSF),[]);     %显示维纳滤波复原图像
title('维纳滤波复原');  
%NP = 0.001 * numel(I);
reg1 = deconvreg(MF,PSF);
figure(4);imshow(reg1,[]);
title('有约束的最小二乘滤波复原');