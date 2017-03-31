clear all;
close all;
addpath('utils/');
folder1 = 'Set5_gt';
folder2 = 'Set5_sr';
folder3 = 'Set5_lr';
filepaths1 = dir(fullfile(folder1, '*.bmp'));
filepaths2 = dir(fullfile(folder2, '*.bmp'));
filepaths3 = dir(fullfile(folder3, '*.bmp'));
PSNR = 0;
SSIM = 0;
SF = 4; %test scale factors. can be 2, 3 or 4
if length(filepaths1) == length(filepaths2)
    for i = 1 : length(filepaths1)
        filename1 = filepaths1(i).name;
        filename2 = filepaths2(i).name;
        filename3 = filepaths3(i).name;
        imGT = imread(fullfile(folder1, filename1));
        imSR = imread(fullfile(folder2, filename2));
        imLR = imread(fullfile(folder3, filename3));
        figure;
        subplot(1,3,1);
        imshow(imLR);
        title('LR');
        subplot(1,3,2);
        imshow(imSR);
        title('SR');
        subplot(1,3,3);
        imshow(imGT);
        title('GT');
        [psnr, ssim]  = compute_diff(imGT, imSR, SF);
        PSNR = PSNR+psnr;
        SSIM = SSIM+ssim;
    end
    PSNR = PSNR/length(filepaths1);
    SSIM = SSIM/length(filepaths1);
end
fprintf('PSNR: %f\n',PSNR);
fprintf('SSIM: %f\n',SSIM);

