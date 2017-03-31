clear;close all;
% settings
folder = 'E:/sr_dataset';
savepath = 'E:/train_75s_8b_';
size_input = 75;
%scale 
scale = 4; 
size_label = size_input * scale;
stride = size_input * scale;



% initialization
data = zeros(size_input, size_input, 3, 1);
label = zeros(size_label, size_label, 3, 1);
%padding = abs(size_input - size_label)/2;
padding = 0;
count = 0;
do_write = 0;

totalct = 0;
created_flag = false;
change =0;
old =1;
% generate data
filepaths = dir(fullfile(folder,'*.bmp'));
rand_index = randperm(length(filepaths));


for index = 1 : length(filepaths)
        i = rand_index(index);
        image = imread(fullfile(folder,filepaths(i).name));
        [H, W, C] = size(image);
        if C == 1
            continue;
        end
        [add, im_name, type] = fileparts(filepaths(i).name);
        im_label = modcrop(image, scale);
        [hei,wid,~] = size(im_label);
        %generate ILR
        %im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');
        %im_input = imresize(im_label,1/scale,'bicubic');
        %im_input = im_label;
        %generate sub-ILR
        for x = 1 : stride : hei-size_label+1
            for y = 1 :stride : wid-size_label+1
                subim_label = im_label(x+padding : x+padding+size_label-1, y+padding : y+padding+size_label-1,:);
                subim_input = imresize(subim_label,1/scale,'bicubic');

                count=count+1;  
                do_write = do_write +1;
                data(:, :, :, do_write) = subim_input;
                label(:, :, :, do_write) = subim_label;
                fprintf('do_write: %d\n', do_write);
                if do_write == 800
                    order = randperm(do_write);
                    data = data(:, :, :, order);
                    label = label(:, :, :, order);

                    % writing to HDF5
                    chunksz = 8;
%                     created_flag = false;
%                     totalct = 0;

                    for batchno = 1:floor(do_write/chunksz)
                        last_read=(batchno-1)*chunksz;
                        batchdata = data(:,:,:,last_read+1:last_read+chunksz);
                        batchlabs = label(:,:,:,last_read+1:last_read+chunksz);

                        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
                        curr_dat_sz = store2hdf5([savepath num2str(change) '.h5'], batchdata, batchlabs, ~created_flag, startloc, chunksz);
                        created_flag = true;
                        totalct = curr_dat_sz(end);
                    end
                    h5disp([savepath num2str(change) '.h5']);
                    fprintf('count:%d, totalct:%d\n',count,totalct);
                    data = zeros(size_input, size_input, 3, 1);
                    label = zeros(size_label, size_label, 3, 1); 
                    do_write = 0;
                end
               change = floor(count/4800) +1 ;
               if change ~= old
                      created_flag = false;
                    totalct = 0;
                    do_write = 0;
                end
               old = change;
                
                
            end
        end
end


% order = randperm(count);
% data = data(:, :, :, order);
% label = label(:, :, :, order);
% 
% % writing to HDF5
% chunksz = 16;
% created_flag = false;
% totalct = 0;
% 
% for batchno = 1:floor(count/chunksz)
%     last_read=(batchno-1)*chunksz;
%     batchdata = data(:,:,:,last_read+1:last_read+chunksz);
%     batchlabs = label(:,:,:,last_read+1:last_read+chunksz);
% 
%     startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
%     curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
%     created_flag = true;
%     totalct = curr_dat_sz(end);
% end
h5disp([savepath num2str(change) '.h5']);
