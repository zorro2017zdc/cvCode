clear
close all
f = 'a';
ext = 'jpg';
img1 = imread('4.jpg');
img2 = imread('5.jpg');
% img3 = imread('b3.jpg');

img0 = imMosaic(img1,img2,1);
% img0 = imMosaic(img1,img0,1);
figure,imshow(img0)
imwrite(img0,['mosaic_' f '.' ext],ext)