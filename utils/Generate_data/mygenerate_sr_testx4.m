clear;close all;
%% settings
folder = 'E:/bmp';
savepath = 'E:/test_75s.h5';
scale =4 ;
size_input = 75;
size_label = size_input * scale;
stride = size_label;

%% initialization
data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);
padding = 0;
count = 0;

%% generate data
filepaths = dir(fullfile(folder,'*.bmp'));


for i = 1 : length(filepaths)
        fprintf('scale:%d,%d\n',scale,i);

        image = imread(fullfile(folder,filepaths(i).name));
        [H, W, C] = size(image);
        if C == 1
            continue;
        end
        im_label = modcrop(image, scale);
        [hei,wid,~] = size(im_label);
        %im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');

        for x = 1 : stride : hei-size_label+1
            for y = 1 :stride : wid-size_label+1

                subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,:);
                subim_input = imresize(subim_label,1/scale,'bicubic');


                count=count+1;
                data(:, :, :, count) = subim_input;
                label(:, :, :, count) = subim_label;
            end
        end
end


order = randperm(count);
data = data(:, :, :, order);
label = label(:, :, :, order);

%% writing to HDF5
chunksz = 2;
created_flag = false;
totalct = 0;

for batchno = 1:floor(count/chunksz)
    last_read=(batchno-1)*chunksz;
    batchdata = data(:,:,:,last_read+1:last_read+chunksz);
    batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

    startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
    curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
    created_flag = true;
    totalct = curr_dat_sz(end);
end
h5disp(savepath);
