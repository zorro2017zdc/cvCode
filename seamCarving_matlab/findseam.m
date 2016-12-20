%%%%%%%%%%%%%%%%
%
%使用动态规划的方法实现seam carving
%
%  崔振龙 2016.12.09 南京
%
%%%%%%%%%%%%%%%%
clc;
clear;
Im=imread('ori.jpg');
gauss=fspecial('gaussian',[3,3],2);
Blur=imfilter(Im,gauss,'same');

[m,n,c]=size(Im);
%灰度图
Gray=rgb2gray(Im);

%求梯度图
hy=fspecial('sobel');
hx=hy';
Iy=imfilter(double(Gray),hy,'replicate');
Ix=imfilter(double(Gray),hx,'replicate');
Gradient=sqrt(Ix.^2+Iy.^2);
%归一化
max1=max(max(Gradient)');
Gradient=Gradient./max1;

%能量值图
Energy=zeros(m,n);
%路径图
Path=zeros(m,n);
tmp=0;
for i=1:m
    for j=1:n
        if(i==1)
            Energy(i,j)=Gradient(i,j);
            Path(i,j)=0;
        else
            if(j==1)
                tmp=which_min2(Energy(i-1,j),Energy(i-1,j+1));
                Energy(i,j)=Gradient(i,j)+Energy(i-1,j+tmp);
                Path(i,j)=tmp;
            elseif(j==n)
                tmp=which_min2(Energy(i-1,j-1),Energy(i-1,j));
                Energy(i,j)=Gradient(i,j)+Energy(i-1,j-1+tmp);
                Path(i,j)=tmp-1;
            else
                tmp=which_min3(Energy(i-1,j-1),Energy(i-1,j),Energy(i-1,j+1));
                Energy(i,j)=Gradient(i,j)+Energy(i-1,j-1+tmp);
                Path(i,j)=tmp-1;
            end
        end
    end
end
%归一化
%归一化
max2=max(max(Energy)');
Energy=Energy./max2;

%能量最小路径的最后一行的纵坐标
lastCol=find(Energy(m,:)==min(Energy(m,:)));
col=lastCol(1);
%描画出分割线
Line=Im;
for i=m:-1:1
    Line(i,col,:)=[0,255,0];
    col=col+Path(i,col);
end

%消除路径上的点
Im=Im(:);
for i=m:-1:1
    Im(1*i+col)=[];
    Im(2*i+col)=[];
    Im(3*i+col)=[];
    col=col+Path(i,col);
end
Im=reshape(Im,m,n-1,3);
%Gradient是梯度图
figure,imshow(Gradient);title('Gradient Image');
%Energy是累加能量图
figure,imshow(Energy);title('Cumulative Energy Image');
%Line是标注了分割线的图
figure,imshow(Line);title('Image with Seam');
%在Im上将分割线切掉
figure,imshow(Im);title('after Cut Seam');