close all
clc
%  ImageName='C:\Andrey\DeepLearning\ChartExtraction\dcal5X7w.jpeg';
%  ImageName='C:\Andrey\DeepLearning\ChartExtraction\iyNrgCxg.jpeg';
%  ImageName='C:\Andrey\DeepLearning\ChartExtraction\Eha5bkFA.jpeg';
% ImageName='C:\Andrey\DeepLearning\ChartExtraction\PXcaBvlw.jpeg';
% ImageName='C:\Andrey\DeepLearning\ChartExtraction\WtRGxpuw.jpeg';
ImageName='C:\Andrey\DeepLearning\ChartExtraction\S1t94cQQ.jpeg';

Im=imread(ImageName);
%% rgb to gray
Im_gray=rgb2gray(Im);
figure
imshow(Im_gray,[])
%% Convert to binary image

Threshold=252;
Ibin=Im_gray;
Ibin(Ibin<Threshold)=0;
Ibin(Ibin>=Threshold)=255;
figure
imshow(Ibin,[])
% median filter
Ibin=medfilt2(Ibin,[5,5]);
figure,imshow(Ibin,[])
Ibin=medfilt2(Ibin,[5,5]);
figure,imshow(Ibin,[])

se0  = strel('rectangle',[5,5]); % we assume that minimal size of the chart is 50x50
eroded0 = imerode(Ibin,se0);
figure, imshow(eroded0)
se1  = strel('rectangle',[10,50]); % we assume that minimal size of the chart is 50x50
closedI = imclose(Ibin,se1);
figure, imshow(closedI)

[L1, NUM1] = bwlabeln(imcomplement(closedI));
stats1 = regionprops(L1,'Area','BoundingBox');

%% Threshold regions by area
%% Full image area
FullArea=size(closedI,1)*size(closedI,2);
thrArea=0.01;% 2%
relevant_labels_ind=[];
for ind =1:NUM1
    if(stats1(ind).Area<FullArea*thrArea)
        closedI(L1==ind)=255;
    else
        relevant_labels_ind=[relevant_labels_ind;ind];
    end
end

se2  = strel('rectangle',[20,20]); % we assume that minimal size of the chart is 50x50
erodedI = imopen(closedI,se2);
figure, imshow(erodedI)
[L2, NUM2] = bwlabeln(imcomplement(erodedI));
stats2 = regionprops(L2,'BoundingBox');
Imasked=zeros(size(erodedI));
all_x_left=[];
all_y_up=[];
all_x_right=[];
all_y_down=[];

for ind=1:NUM2
    curr_bound_box=stats2(ind).BoundingBox;
    x_left=max(floor(curr_bound_box(1)),1);
    y_up=max(floor(curr_bound_box(2)),1);
    x_right=min(ceil(curr_bound_box(1)+curr_bound_box(3)),size(closedI,2));
    y_down=min(ceil(curr_bound_box(2)+curr_bound_box(4)),size(closedI,1));
    Imasked(y_up:y_down,x_left:x_right)=Im_gray(y_up:y_down,x_left:x_right);
    all_x_left=[all_x_left;x_left];
    all_y_up=[all_y_up;y_up];
    all_x_right=[all_x_right;x_right];
    all_y_down=[all_y_down;y_down];
    
    
end
figure, imshow(Imasked,[])


%% Horisontal and vertical Gradients detections

for ind=1:NUM2
    currRegion=Im_gray(all_y_up(ind):all_y_down(ind),all_x_left(ind):all_x_right(ind));
    kernel_length=round(size(currRegion,2)/5);
    horiz_kernel=-1*ones(3,kernel_length);
    horiz_kernel(2,:)=2*ones(1,kernel_length);
    
    
    
    figure, imshow(currRegion,[])
    %% Horizontal gradient
    BW_horiz = conv2(currRegion,horiz_kernel,'same');
    figure;imshow(BW_horiz,[]);
    %% find max value
    min_grad_horiz=min(BW_horiz(:));
    thresh_grad_horiz=0.3*min_grad_horiz;
    BW_horiz_thresh=BW_horiz;
    BW_horiz_thresh(BW_horiz<=thresh_grad_horiz)=1;
    BW_horiz_thresh(BW_horiz>thresh_grad_horiz)=0;
    figure;imshow(BW_horiz_thresh,[]);
    BW_horiz_thresh=medfilt2(BW_horiz_thresh,[1,round(size(currRegion,2)/3)]);
    figure;imshow(BW_horiz_thresh,[]);
    se0  = strel('line',round(size(currRegion,2)/3),0); 
    BW_horiz_thresh = imdilate(BW_horiz_thresh,se0);  
    
    figure;imshow(BW_horiz_thresh,[]);
    
    %% Vertical gradient
    kernel_length=round(size(currRegion,1)/5);
    vert_kernel=-1*ones(kernel_length,3);
    vert_kernel(:,2)=2*ones(kernel_length,1);
    
    
    BW_vert = conv2(currRegion,vert_kernel,'same');
    figure;imshow(BW_vert,[]);
    %% find max value
    min_grad_vert=min(BW_vert(:));
    thresh_grad_vert=0.3*min_grad_vert;
    BW_vert_thresh=BW_vert;
    BW_vert_thresh(BW_vert<=thresh_grad_vert)=1;
    BW_vert_thresh(BW_vert>thresh_grad_vert)=0;
    BW_vert_thresh=medfilt2(BW_vert_thresh,[round(size(currRegion,1)/3),1]);
    figure;imshow(BW_vert_thresh,[]);
    se0  = strel('line',round(size(currRegion,1)/3),90); 
    BW_vert_thresh = imdilate(BW_vert_thresh,se0);
    figure;imshow(BW_vert_thresh,[]);
    
    
  
    B_vert_and_horiz_edges=BW_horiz_thresh+BW_vert_thresh;
    figure;imshow(B_vert_and_horiz_edges,[]);
    
    stats = regionprops(not(B_vert_and_horiz_edges));
    
    total_area=size(B_vert_and_horiz_edges,1)*size(B_vert_and_horiz_edges,2);
    min_area=100*100;
    figure;imshow(currRegion,[]);hold on;
    
    for i = 1:numel(stats)
        if(stats(i).Area>min_area)
        % skip by area factor
        rectangle('Position', stats(i).BoundingBox, ...
            'Linewidth', 1, 'EdgeColor', 'r', 'LineStyle', '--');
        end
    end
    hold off
    ty=1;
end






