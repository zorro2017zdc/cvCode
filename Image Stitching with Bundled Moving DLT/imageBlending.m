function [output_canvas,mask_img1,mask_img2] = imageBlending(varargin)
    % warped_img1 and warped_img2: (warped) RGB images to blend.
    % method: method to use for blending. Possible values are: 'linear'
    %         (default) or 'feathering'.
    warped_img1 = varargin{1};
    warped_img2 = varargin{2};
    method = varargin{3};
    
    mask_img1 = imfill(im2bw(uint8(warped_img1), 0),'holes');
    mask_img2 = imfill(im2bw(uint8(warped_img2), 0),'holes');
    


    if strcmp(method,'laplacian') == 1
        
        level = varargin{4}; 
        
        w = mask_img2&mask_img1;
        mask_img2 = (~w)&mask_img2;

        mask_img1 = mat2gray(mask_img1);
        mask_img2 = mat2gray(mask_img2);                       
        
        limga = genPyr(warped_img1,'lap',level); % the Laplacian pyramid
        limgb = genPyr(warped_img2,'lap',level);

        maska = zeros(size(warped_img1));
        maska(:,:,1) = mask_img1;
        maska(:,:,2) = mask_img1;
        maska(:,:,3) = mask_img1;                
        maskb = 1-maska;                       
        %blurh = fspecial('gauss',30,15); % feather the border
        %maska = imfilter(maska,blurh,'replicate');        
        %maskb = imfilter(maskb,blurh,'replicate');

        limgo = cell(1,level); % the blended pyramid
        for p = 1:level
            [Mp Np ~] = size(limga{p});
            maskap = imresize(maska,[Mp Np]);
            maskbp = imresize(maskb,[Mp Np]);
            limgo{p} = limga{p}.*maskap + limgb{p}.*maskbp;
        end
        output_canvas = pyrReconstruct(limgo);
        return;
    elseif strcmp(method,'feathering') == 1    
        mask_img1 = ~mask_img1;
        mask_img2 = ~mask_img2;
        if nargin == 3
            p = 4;
        elseif nargin == 4
            p = varargin{4};
        end
        if nargin == 5
            p = varargin{4};
            mask_img1 = bwdist(mask_img1,varargin{5});
            mask_img2 = bwdist(mask_img2,varargin{5});
            % the method for getting the distance transform of the binary image
            % can be one of:
            %   'chessboard',
            %   'cityblock',
            %   'euclidean' or
            %   'quasi-euclidean'
        else        
            mask_img1 = bwdist(mask_img1); % default method is Euclidean
            mask_img2 = bwdist(mask_img2);
        end
        mask_img1 = mat2gray(mask_img1).^p;
        mask_img2 = mat2gray(mask_img2).^p;
    else % linear (average) blending
        mask_img1 = mat2gray(mask_img1);
        mask_img2 = mat2gray(mask_img2);        
    end
    
    img1 = warped_img1;
    img2 = warped_img2;
    img1(:,:,1) = (double(warped_img1(:,:,1)).*mask_img1)./(mask_img1+mask_img2);
    img1(:,:,2) = (double(warped_img1(:,:,2)).*mask_img1)./(mask_img1+mask_img2);
    img1(:,:,3) = (double(warped_img1(:,:,3)).*mask_img1)./(mask_img1+mask_img2);
    img1 = uint8(img1);            
    
    img2(:,:,1) = (double(warped_img2(:,:,1)).*mask_img2)./(mask_img1+mask_img2);
    img2(:,:,2) = (double(warped_img2(:,:,2)).*mask_img2)./(mask_img1+mask_img2);
    img2(:,:,3) = (double(warped_img2(:,:,3)).*mask_img2)./(mask_img1+mask_img2);    
    img2 = uint8(img2);            
    
    output_canvas(:,:,1) = (double(warped_img1(:,:,1)).*mask_img1+double(warped_img2(:,:,1)).*mask_img2)./(mask_img1+mask_img2);
    output_canvas(:,:,2) = (double(warped_img1(:,:,2)).*mask_img1+double(warped_img2(:,:,2)).*mask_img2)./(mask_img1+mask_img2);
    output_canvas(:,:,3) = (double(warped_img1(:,:,3)).*mask_img1+double(warped_img2(:,:,3)).*mask_img2)./(mask_img1+mask_img2);
    output_canvas = uint8(output_canvas);            
    