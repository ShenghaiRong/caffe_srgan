clear;close all;
% settings
folder = 'E:/ImageNet';
savepath = 'E:/rongsh/train_74s_8b_';
size_input = 74;
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
filepaths = dir(fullfile(folder,'*.JPEG'));
rand_index = randperm(length(filepaths));


for index = 1 : length(filepaths)
    try
        i = rand_index(index);
        image = imread(fullfile(folder,filepaths(i).name));
        [H, W, C] = size(image);
        if C == 1 || H < size_label || W < size_label
            continue;
        end
        
        [add, im_name, type] = fileparts(filepaths(i).name);
        left = floor(( W - size_label)/2)+1 ;
        top = floor((H - size_label)/2) +1;
        im_label = imcrop(image, [left,top,size_label-1,size_label-1]);
        %generate ILR
        %im_input = imresize(imresize(im_label,1/scale,'bicubic'),[hei,wid],'bicubic');
        %im_input = imresize(im_label,1/scale,'bicubic');
        %im_input = im_label;
        %generate sub-ILR
        
        subim_label = im_label;
        subim_input = imresize(subim_label,1/scale,'bicubic');
%         figure;imshow(image);
%         figure;imshow(subim_label);
%         figure;imshow(subim_input);
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
       if count == 240000
           break;
       end
       change = floor(count/4800) +1 ;
       if change ~= old
              created_flag = false;
            totalct = 0;
            do_write = 0;
        end
       old = change;
       
    catch err
        continue;
    end

       


end

% h5disp([savepath num2str(change) '.h5']);
