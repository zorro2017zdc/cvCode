% Image Stitching with Moving DLT.
close all;
clear all;
clc;

%-------
% Paths.
%-------
addpath(genpath('sparselm-1.3'));
addpath('modelspecific');
addpath('multigs');
addpath('GCMex');

%-------------------
% Compile Mex files.
%-------------------
cd multigs;
if exist('computeIntersection','file')~=3
    mex computeIntersection.c;
end
cd ..;

if exist('imageWarping','file')~=3 || exist('imageProjection','file')~=3  || exist('getLocation','file')~=3 ||...
   exist('getBlendMask','file')~=3 || exist('ceresRigidError','file')~=3 || exist('ceresNonrigidError','file')~=3
    mex imageWarping.cpp;
    mex imageProjection.cpp;
    mex getBlendMask.cpp;
    mex getLocation.cpp;
    
    % CERES solver and GLOG (both from google) should be installed in your
    % system in order to compile the following files.
    mex ceresRigidError.cpp /usr/local/lib/libceres_shared.so /usr/local/lib/libglog.so -I/usr/include/eigen3;
    mex ceresNonrigidError.cpp /usr/local/lib/libceres_shared.so /usr/local/lib/libglog.so -I/usr/include/eigen3;
end

%----------------------
% Setup VLFeat toolbox.
%----------------------
cd vlfeat-0.9.14/toolbox;
feval('vl_setup');
cd ../..;

%---------------------------------------------
% Check if we are already running in parallel.
%---------------------------------------------
poolsize = matlabpool('size');
if poolsize == 0 %if not, we attemp to do it:
    matlabpool open;
end

%-------------------------
% User defined parameters.
%-------------------------
% Global model specific function handlers.
clear global;
global fitfn resfn degenfn psize numpar
fitfn = 'homography_fit';
resfn = 'homography_res';
degenfn = 'homography_degen';
psize   = 4;
numpar  = 9;


% THERE ARE NO DEFAULT PARAMETERS IN THIS IMP.
% Tune gamma and sigma in order to improve results.

M     = 500; % Number of hypotheses for RANSAC.
thr   = 0.05; % RANSAC threshold (0.1 for railtracks).

beta  = 2;   % Seamcut pairwise potential parameter.

reso  = 100;   % Resolution for showing mapping function in MDLT.
gamma = 0.1; % Normalizer for Moving DLT. (0.05-0.15 are usually good numbers
              % in many cases).
sigma = 12;   % Bandwidth for Moving DLT. (between 8-12 are good numbers).

scale = 1.0;  % Scale of input images

projection = 'sphere';% Possible values are: 'plane'(default), 'cylinder', or 'sphere'.
                 % planar: does not pre-project the images.
                 % cylinder: pre-projects the images into a cylinder (following Szeliski's book, Pp. 385).
                 % sphere: pre-projects the images into a spere (following Szeliski's book, Pp. 386).

%------------------------------
% Folder with images to stitch.
%------------------------------
folder_name = 'images/glenelg/';
imgs_format = '*.JPG';

%----------------
% Reading images.
%----------------
fprintf('Reading images...');tic;
dir_folder = dir(strcat(folder_name,imgs_format));
num_imgs = length(dir_folder);  % Number of files found
if num_imgs == 0
    error('No images could be read from folder... verify the folder name and file extension.');
end
fprintf('done (%fs)\n',toc);

imgs_pairs = combnk(1:num_imgs,2); % <-- Generating image pairs with all available image idexes.
imgs_pairs = [imgs_pairs ones(size(imgs_pairs,1),1)]; % Last column indicates if the pair is valid or not.
                                
%-----------------------------------------------------
% Keypoint detection and matching between image pairs.
%-----------------------------------------------------
fprintf('Pairwise keypoint detection and matching\n');

% Memory allocation
images  = cell(num_imgs,1);
inliers = cell(size(imgs_pairs,1),1);
data_norm = cell(size(imgs_pairs,1),1);
data_orig = cell(size(imgs_pairs,1),1);
h  = cell(size(imgs_pairs,1),1);

% For each image pair we find positive image matches.
for i=1:size(imgs_pairs,1)
    
    % Read images.
    img1 = imresize(imread(sprintf('%s%s',folder_name,dir_folder(imgs_pairs(i,1)).name)),scale);
    img2 = imresize(imread(sprintf('%s%s',folder_name,dir_folder(imgs_pairs(i,2)).name)),scale);

    % Pre-warp images
    if exist('projection','var') && strcmp(projection,'cylinder') == 1
        fprintf('  (Pre-)Projecting images into a cylinder...');tic;
        [img1,encl_rect] = imageProjection(double(img1),size(img1,2),'cylinder');
        img1 = reshape(uint8(img1),size(img1,1),size(img1,2)/3,3);
        img1 = imcrop(img1,encl_rect); 

        [img2,encl_rect] = imageProjection(double(img2),size(img2,2),'cylinder');
        img2 = reshape(uint8(img2),size(img2,1),size(img2,2)/3,3);
        img2 = imcrop(img2,encl_rect);
        fprintf('done (%fs)\n',toc);   
    elseif exist('projection','var') && strcmp(projection,'sphere') == 1
        fprintf('  (Pre-)Projecting images into a sphere...');tic;
        [img1,encl_rect] = imageProjection(double(img1),size(img1,2),'sphere');
        img1 = reshape(uint8(img1),size(img1,1),size(img1,2)/3,3);
        img1 = imcrop(img1,encl_rect); 

        [img2,encl_rect] = imageProjection(double(img2),size(img2,2),'sphere');
        img2 = reshape(uint8(img2),size(img2,1),size(img2,2)/3,3);
        img2 = imcrop(img2,encl_rect);
        fprintf('done (%fs)\n',toc);
    elseif exist('projection','var') && strcmp(projection,'plane') == 1

    elseif exist('projection','var')
        fprintf('\n  ');
        warning('(Pre-)Projection option not recognised, continuing without pre-projecting the images...');
    end

    % Save images.
    images{imgs_pairs(i,1)} = img1;
    images{imgs_pairs(i,2)} = img2;

    % SIFT keypoint detection and matching.
    [kp1 ds1] = vl_sift(single(rgb2gray(img1)));
    [kp2 ds2] = vl_sift(single(rgb2gray(img2)));
    matches = vl_ubcmatch(ds1,ds2,1.6);
    
    % Normalise point distribution.
    data_orig{i} = [ kp1(1:2,matches(1,:)) ; ones(1,size(matches,2)) ; kp2(1:2,matches(2,:)) ; ones(1,size(matches,2)) ];
    [data_norm_img1 T1] = normalise2dpts(data_orig{i}(1:3,:));
    [data_norm_img2 T2] = normalise2dpts(data_orig{i}(4:6,:));
    data_norm{i} = [data_norm_img1;data_norm_img2];

    % Outlier removal - Multi-GS (RANSAC).
    rng(0);
    [ ~,res,~,~,err ] = multigsSampling(100,data_norm{i},M,10);
    con = sum(res<=thr);
    [ ~, maxinx ] = max(con);
    inliers{i} = find(res(:,maxinx)<=thr);    
    
    % Refine homography using DLT on inliers.
    h{i} = feval(fitfn,data_norm{i}(:,inliers{i}));
    h{i} = T2\(reshape(h{i},3,3)*T1);
    h{i} = h{i}(:); % <- these are the initial estimates for Levenberg-Marquardt.
    
    % Verify image match.
    if err || length(inliers{i}) <= (8 + 0.3 *length(matches)) % Invalid image match (following Brown and Lowe).
                                                      % conference paper version values: 5.9 + 0.22 * length(matches)
        % Mark image pair as invalid image match and clean up the
        % image pair's data.
        imgs_pairs(i,3) = 0;
    end
end
valid_pairs = find(imgs_pairs(:,3));
fprintf('done (%fs)\n',toc);

%---------------------------------------------
% Extract reference image and stitching order.
%---------------------------------------------
% Build image connection graph. This graph will be useful for 
% generating the stitching order and determining the reference image.
fprintf('Building image connection graph...');tic;
imgs_conn_graph = zeros(num_imgs,num_imgs);
for i=1:size(valid_pairs,1)
    % The weight of each arc in the graph is the number of (pairwise) keypoint matches.
    imgs_conn_graph(imgs_pairs(valid_pairs(i),1),imgs_pairs(valid_pairs(i),2)) = length(inliers{valid_pairs(i)});
    imgs_conn_graph(imgs_pairs(valid_pairs(i),2),imgs_pairs(valid_pairs(i),1)) = length(inliers{valid_pairs(i)});
end
fprintf('done (%fs)\n',toc);

% Choose reference image (based on image graph).
fprintf('Obtaining reference image and stitching order...');tic;
[~,ref_img] = max(sum(imgs_conn_graph,2));

% Find stitching order.
stitching_order = {ref_img};

% Already considered/ordered images.
ordered_imgs = ref_img; % The first ordered image is the 
                        % reference image.
% Unconsidered images.
unordered_imgs = setdiff(1:num_imgs,ordered_imgs);

graph_paths = cell(num_imgs,1); % Contains the path from one image 
                                % to the reference image.
graph_paths{ref_img} = ref_img;
while ~isempty(unordered_imgs)    
    % Select the next image (using the weighted image connection graph).
    edge_weight = 0;
    for i=unordered_imgs
        for j=ordered_imgs
            if imgs_conn_graph(i,j) > edge_weight
                edge_weight = imgs_conn_graph(i,j);
                next_img   = i;
                middle_img = j;
            end
        end
    end
    
    if edge_weight == 0 % There was no next image, we probably
                        % have a disconnected panorama.
        fprintf('\n  ');
        warning('Found a disconnected panorama...');
        break;
    end

    % Find the path from the next image to the reference image.
    graph_paths{next_img} = [graph_paths{middle_img} next_img];
    stitching_order = [stitching_order; {graph_paths{next_img}}];
    
    % Update variables.            
    ordered_imgs = [ordered_imgs next_img];
    unordered_imgs = setdiff(1:num_imgs,ordered_imgs);    
end
fprintf('done (%fs)\n',toc);

% Order data according to the stitching order (target image is always on the left).
fprintf('> Ordering data according to stitching order...');tic;
stitch_pairs = []; % Will save the (valid) image pairs that we need for generating 
                   % the panorama (following the stitching order).
for i=1:size(stitching_order,1)
    for j=1:size(stitching_order{i},2)-1        
        for k=1:size(valid_pairs,1)
            if stitching_order{i}(j) == imgs_pairs(valid_pairs(k),2) && stitching_order{i}(j+1) == imgs_pairs(valid_pairs(k),1)                
                aux = imgs_pairs(valid_pairs(k),1);
                imgs_pairs(valid_pairs(k),1) = imgs_pairs(valid_pairs(k),2);
                imgs_pairs(valid_pairs(k),2) = aux;

                aux = data_orig{valid_pairs(k)}(1:3,:);
                data_orig{valid_pairs(k)}(1:3,:) = data_orig{valid_pairs(k)}(4:6,:);
                data_orig{valid_pairs(k)}(4:6,:) = aux;

                aux = data_norm{valid_pairs(k)}(1:3,:);
                data_norm{valid_pairs(k)}(1:3,:) = data_norm{valid_pairs(k)}(4:6,:);
                data_norm{valid_pairs(k)}(4:6,:) = aux;
                
                h{valid_pairs(k)} = reshape(reshape(h{valid_pairs(k)},3,3)\eye(3),9,1);
                
                stitch_pairs = [stitch_pairs; valid_pairs(k)];
                break;
            elseif stitching_order{i}(j) == imgs_pairs(valid_pairs(k),1) && stitching_order{i}(j+1) == imgs_pairs(valid_pairs(k),2)
                stitch_pairs = [stitch_pairs; valid_pairs(k)];
                break;
            end
        end
    end
end
stitch_pairs = unique(stitch_pairs);
fprintf('done (%fs)\n',toc);

%------------------------------------------------------
% Image stitching with pairwise (chained) Homographies.
%------------------------------------------------------
fprintf('Image stitching with pairwise Homographies\n');
% Chaining the homographies H (obtained by means of
% pairwise DLT). These chained H's will be the initial 
% estimates for BA.
H = cell(size(stitching_order,1),1);
for i=1:size(stitching_order,1)
    H{i} = eye(3);
    for j=1:size(stitching_order{i},2)-1
        for k=1:size(stitch_pairs,1)
            if stitching_order{i}(j) == imgs_pairs(stitch_pairs(k),1) && stitching_order{i}(j+1) == imgs_pairs(stitch_pairs(k),2)
                H{i} = H{i} * reshape(h{stitch_pairs(k)},3,3);
                break;
            end
        end
    end
end

% sort the paramaters following the image order.
lm_params0  = [];
for i=1:num_imgs
    for j=1:size(stitching_order,1)
        if i == stitching_order{j}(end)
            lm_params0 = [lm_params0; H{j}(:)];
            break;
        end
    end
end

fprintf('  Getting canvas size...');tic;
% Map four corners of the right image.
TL = []; % top-left corner
BL = []; % bottom-left corner
TR = []; % top-right corner
BR = []; % bottom-right corner
pairwiseH = cell(num_imgs,1);
for i=1:size(stitching_order,1)
    pairwiseH{i} = reshape(lm_params0(((i-1)*9)+1:i*9),3,3);
    aux = pairwiseH{i}\[1;1;1];
    TL  = [TL; round(aux(1,1)/aux(3,1)) round(aux(2,1)/aux(3,1))];
    aux = pairwiseH{i}\[1;size(images{ref_img},1);1];
    BL  = [BL; round(aux(1,1)/aux(3,1)) round(aux(2,1)/aux(3,1))];
    aux = pairwiseH{i}\[size(images{ref_img},2);1;1];
    TR  = [TR; round(aux(1,1)/aux(3,1)) round(aux(2,1)/aux(3,1))];
    aux = pairwiseH{i}\[size(images{ref_img},2);size(images{ref_img},1);1];
    BR  = [BR; round(aux(1,1)/aux(3,1)) round(aux(2,1)/aux(3,1))];
end

% Canvas size.
cw = max([1 size(images{ref_img},2) TL(:,1)' BL(:,1)' TR(:,1)' BR(:,1)']) - min([1 size(images{ref_img},2) TL(:,1)' BL(:,1)' TR(:,1)' BR(:,1)']) + 1;
ch = max([1 size(images{ref_img},1) TL(:,2)' BL(:,2)' TR(:,2)' BR(:,2)']) - min([1 size(images{ref_img},1) TL(:,2)' BL(:,2)' TR(:,2)' BR(:,2)']) + 1;
fprintf('done (%fs)\n',toc);

% Offset for left image.
fprintf('  Getting offset...');tic;
off = [ 1 - min([1 size(images{ref_img},2) TL(:,1)' BL(:,1)' TR(:,1)' BR(:,1)']) + 1 ; 1 - min([1 size(images{ref_img},1) TL(:,2)' BL(:,2)' TR(:,2)' BR(:,2)']) + 1 ];
fprintf('done (%fs)\n',toc);

% Warping and blending the images.
fprintf('  Warping and blending the images...');tic;
canvas_img = uint8(zeros(ch,cw,3));
for i=1:num_imgs
    warped_img2 = imageWarping(double(ch),double(cw),double(images{i}),pairwiseH{i},double(off));
    warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);

    % Blending images by averaging
    canvas_img = imageBlending(canvas_img,warped_img2,'linear');
end
pairwiseH_fig = figure;imshow(canvas_img);
drawnow;


%-----------------------------------------
% Image stitching using Bundle Adjustment.
%-----------------------------------------
% Obtaining size of canvas.
fprintf('Image stitching with Bundle Adjustment\n');

% Obtain xk's on images
fprintf('> Getting xk''s on images...');tic;
all_kpts_per_img = cell(num_imgs,1);
for i=1:num_imgs
    for j=1:size(stitch_pairs,1)
        if imgs_pairs(stitch_pairs(j),1) == i
            all_kpts_per_img{i} = unique([all_kpts_per_img{i} data_orig{stitch_pairs(j)}(1:2,inliers{stitch_pairs(j)})]','rows')';
        elseif imgs_pairs(stitch_pairs(j),2) == i
            all_kpts_per_img{i} = unique([all_kpts_per_img{i} data_orig{stitch_pairs(j)}(4:5,inliers{stitch_pairs(j)})]','rows')';
        end
    end
end
fprintf('done (%fs)\n',toc);

% Obtain xi's on canvas
fprintf('  Getting xi''s on canvas...');tic;
xk = [];
for i=1:num_imgs
    for j=1:size(all_kpts_per_img{i},2)
        aux = NaN(2*(num_imgs),1);
        aux(((i-1)*2)+1:i*2) = all_kpts_per_img{i}(:,j);
        for k=1:size(stitch_pairs,1)
            if imgs_pairs(stitch_pairs(k),1) == i
                idx = getLocation(data_orig{stitch_pairs(k)}(1:2,inliers{stitch_pairs(k)})',all_kpts_per_img{i}(:,j)');
                if idx ~= 0
                    aux(((imgs_pairs(stitch_pairs(k),2)-1)*2)+1:imgs_pairs(stitch_pairs(k),2)*2) = data_orig{stitch_pairs(k)}(4:5,inliers{stitch_pairs(k)}(idx));
                end
            elseif imgs_pairs(stitch_pairs(k),2) == i
                idx = getLocation(data_orig{stitch_pairs(k)}(4:5,inliers{stitch_pairs(k)})',all_kpts_per_img{i}(:,j)');
                if idx ~= 0
                    aux(((imgs_pairs(stitch_pairs(k),1)-1)*2)+1:imgs_pairs(stitch_pairs(k),1)*2) = data_orig{stitch_pairs(k)}(1:2,inliers{stitch_pairs(k)}(idx));
                end
            end
        end
        if ~isempty(xk)
            found = 0;
            for k=1:num_imgs
                idx = getLocation(xk((k-1)*2+1:k*2,:)',aux((k-1)*2+1:k*2)');
                if idx ~= 0
                    found = 1;
                    for l=1:num_imgs
                        if ~isnan(aux((l-1)*2+1))
                            xk((l-1)*2+1:l*2,idx) = aux((l-1)*2+1:l*2);
                        end
                    end
                    break;
                end
            end
            if found == 0
                xk = [xk aux];
            end
        else
            xk = [xk aux];
        end
    end
end

lm_xi = zeros(3,size(xk,2));
xik_idx = cell(num_imgs,1);
num_xik = zeros(num_imgs,1);
for i=1:num_imgs
    for j=1:size(stitching_order,1)
        if i == stitching_order{j}(end)
            xik_idx{i} = find(~isnan(xk((i-1)*2+1,:)));
            aux = H{j} \ [xk((i-1)*2+1:i*2,xik_idx{i});ones(1,size(xik_idx{i},2))];
            lm_xi(1:3,xik_idx{i}) = lm_xi(1:3,xik_idx{i}) +...
                [aux(1,:)./aux(3,:); aux(2,:)./aux(3,:); ones(1,size(xik_idx{i},2))];
            break;
        end
    end
    num_xik(i) = size(xik_idx{i},2);
end

lm_xi(1,:) = lm_xi(1,:) ./ lm_xi(3,:);
lm_xi(2,:) = lm_xi(2,:) ./ lm_xi(3,:);
fprintf('done (%fs)\n',toc);

fprintf('  Performing Bundle Adjustment (Levenberg-Marquardt)...');
% Set of parameters to be optimised (H1,...,Hk,x1,...,xN)
lm_params0   = [lm_params0; reshape(lm_xi(1:2,:),size(lm_xi(1:2,:),2)*2,1)];

% Normalizer (lm_norm contains |M(i)| per point xi)
lm_norm = lm_xi(3,:)';
% Get the set of observations (xik)
lm_xi_x = [];
lm_xi_y = [];
for i=1:num_imgs
    lm_xi_x = [lm_xi_x xk((i-1)*2+1,xik_idx{i})];
    lm_xi_y = [lm_xi_y xk((i-1)*2+2,xik_idx{i})];    
end
lm_xk = [lm_xi_x lm_xi_y]';
    
% Perform Bundle Adjustment (sparse Levenberg-Marquardt).
lm_params = ceresRigidError(double(lm_params0),xik_idx,lm_norm,num_imgs,lm_xk);

%ba_xi = lm_params((9*num_imgs)+1:end);
%ba_xi = reshape(ba_xi,2,size(ba_xi,1)/2);
% hold on;plot(ba_xi(1,:)+off(1),ba_xi(2,:)+off(2),'rx');
fprintf('done (%fs)\n',toc);

fprintf('  Getting canvas size...');tic;
% Map four corners of the right image.
TL = []; BL = []; TR = []; BR = []; 
baH = cell(num_imgs,1);
for i=1:size(stitching_order,1)
    baH{i} = reshape(lm_params(((i-1)*9)+1:i*9),3,3);
    aux = baH{i}\[1;1;1]; 
    TL  = [TL; round(aux(1,1)/aux(3,1)) round(aux(2,1)/aux(3,1))];
    aux = baH{i}\[1;size(images{ref_img},1);1]; 
    BL  = [BL; round(aux(1,1)/aux(3,1)) round(aux(2,1)/aux(3,1))];
    aux = baH{i}\[size(images{ref_img},2);1;1];
    TR  = [TR; round(aux(1,1)/aux(3,1)) round(aux(2,1)/aux(3,1))];
    aux = baH{i}\[size(images{ref_img},2);size(images{ref_img},1);1];
    BR  = [BR; round(aux(1,1)/aux(3,1)) round(aux(2,1)/aux(3,1))];
end

% Canvas size.
cw = max([1 size(images{ref_img},2) TL(:,1)' BL(:,1)' TR(:,1)' BR(:,1)']) - min([1 size(images{ref_img},2) TL(:,1)' BL(:,1)' TR(:,1)' BR(:,1)']) + 1;
ch = max([1 size(images{ref_img},1) TL(:,2)' BL(:,2)' TR(:,2)' BR(:,2)']) - min([1 size(images{ref_img},1) TL(:,2)' BL(:,2)' TR(:,2)' BR(:,2)']) + 1;
fprintf('done (%fs)\n',toc);

% Offset for left image.
fprintf('  Getting offset...');tic;
off = [ 1 - min([1 size(images{ref_img},2) TL(:,1)' BL(:,1)' TR(:,1)' BR(:,1)']) + 1 ; 1 - min([1 size(images{ref_img},1) TL(:,2)' BL(:,2)' TR(:,2)' BR(:,2)']) + 1 ];
fprintf('done (%fs)\n',toc);

% Warping and blending the images.
fprintf('  Warping and blending the images...');tic;
canvas_img = uint8(zeros(ch,cw,3));
for i=1:num_imgs
    warped_img2 = imageWarping(double(ch),double(cw),double(images{i}),baH{i},double(off));
    warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);

    % Blending images by averaging
    canvas_img = imageBlending(canvas_img,warped_img2,'linear');
end
rigidBA_fig = figure;imshow(canvas_img);
drawnow;


%----------------------------------------------------
%  Image stitching using Non-Rigid Bundle Adjustment.
%----------------------------------------------------
fprintf('Image stitching with Non-Rigid Bundle Adjustment\n');
fprintf('> Generating mesh...');tic;
% Generating mesh for MDLT.
[X Y] = meshgrid(linspace(1,cw,reso),linspace(1,ch,reso));

% Mesh's vertices coordinates.
Mvts = [X(:)-off(1) Y(:)-off(2)];
fprintf('done (%fs)\n',toc);

D = pdist2(Mvts,[lm_xi(1,:)' lm_xi(2,:)']);
GaussK = exp(-D./sigma^2);
GaussK = max(gamma,GaussK);
W = GaussK./repmat(sum(GaussK,2),1,length(lm_xi));

% Perform Moving DLT
fprintf('  Moving DLT main loop...');tic;

nrbaH0 = cell(1,size(num_imgs,1));
A   = cell(size(num_imgs,1),1);
C1  = cell(size(num_imgs,1),1);
C2  = cell(size(num_imgs,1),1);
T1  = cell(size(num_imgs,1),1);
T2  = cell(size(num_imgs,1),1);

lm_xi_xy = [lm_xi(1,:)' lm_xi(2,:)'];

for i=1:num_imgs
    H = zeros(size(Mvts,1),9);

    auxdata_orig = [ lm_xi(1:2,xik_idx{i}) ; ones(1,size(xik_idx{i},2)) ; xk((i-1)*2+1:(i-1)*2+2,xik_idx{i}) ; ones(1,size(xik_idx{i},2)) ];
    [auxdata_norm_img1 auxT1] = normalise2dpts(auxdata_orig(1:3,:));
    [auxdata_norm_img2 auxT2] = normalise2dpts(auxdata_orig(4:6,:));
    auxdata_norm = [auxdata_norm_img1;auxdata_norm_img2];
    
    [auxh auxA auxC1 auxC2] = feval(fitfn,auxdata_norm);
    
    parfor j=1:size(Mvts,1)
        % Get Weights (w*) focused in current cell
        v = wsvd(W(j,xik_idx{i}),auxA);
        h = reshape(v,3,3)';        

        % De-condition
        h = auxC2\h*auxC1;

        % De-normalize
        h = auxT2\h*auxT1;

        H(j,:) = h(:);
    end
    nrbaH0{i}=H;
end

nrbaH0 = cell2mat(nrbaH0);
fprintf('done (%fs)\n',toc);

fprintf('  Warping and blending the images...');tic;
canvas_img = uint8(zeros(ch,cw,3));
for i=1:num_imgs
    warped_img2 = imageWarping(double(ch),double(cw),double(images{i}),nrbaH0(:,((i-1)*9)+1:(i*9)),double(off),X(1,:),Y(:,1)');
    warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);
       
    canvas_img = imageBlending(canvas_img,warped_img2,'linear');
end  
MDLT_fig = figure;imshow(canvas_img);
drawnow;


% Use Bundle Adjustment (Levenberg-Marquardt) for optimising the parameters.
fprintf('  Performing Non-Rigid Bundle Adjustment...');tic;
% Set of parameters to be optimised (H1,...,Hk,x1,...,xN)

nrbaH = cell(size(Mvts,1),1);
lm_xi_xy = [lm_xi(1,:)' lm_xi(2,:)'];
aux_lm_params0 = lm_params0((9*num_imgs)+1:end);
parfor i=1:size(Mvts,1)
    % Get weights for current cell i
    %Wi = getWeights(Mvts(i,:),lm_xi_xy,sigma,gamma);
    
    % perform (weighted) Bundle Adjustment
    lm_params00 = [nrbaH0(i,:)'; aux_lm_params0];
    lm_params = ceresNonrigidError(lm_params00,xik_idx,lm_norm,num_imgs,lm_xk,W(i,:));

    nrbaH{i} = lm_params(1:9*num_imgs)';
end
nrbaH = cell2mat(nrbaH);
fprintf('done (%fs)\n',toc);

%-----------------------------------
% Image stitching with Non-Rigid BA.
%-----------------------------------
fprintf('  Warping and blending the images...');tic;
canvas_img = uint8(zeros(ch,cw,3));
for i=1:num_imgs
    warped_img2 = imageWarping(double(ch),double(cw),double(images{i}),nrbaH(:,((i-1)*9)+1:(i*9)),double(off),X(1,:),Y(:,1)');
    warped_img2 = reshape(uint8(warped_img2),size(warped_img2,1),size(warped_img2,2)/3,3);
       
    canvas_img = imageBlending(canvas_img,warped_img2,'linear');
end  
nonrigidBA_fig = figure;imshow(canvas_img);
drawnow;


