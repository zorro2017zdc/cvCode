# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/imgproc/imgproc.hpp"
#include"opencv2/nonfree/nonfree.hpp"
#include"opencv2/calib3d/calib3d.hpp"
#include<iostream>
using namespace cv;
using namespace std;

/**
* @function main
* @brief Main function
*/
int main(int argc, char** argv)
{
	//Mat img1 = imread("window_view_1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat img2 = imread("window_view_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat img1 = imread("sport0.png", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat img2 = imread("sport1.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img1 = imread("sport0.png");
	Mat img2 = imread("sport1.png");
	if (!img1.data || !img2.data)
	{
		printf(" --(!) Error reading images \n"); return -1;
	}

	//
	Mat imgGray;
	cvtColor(img1,imgGray,CV_RGB2GRAY);
	vector<Point> corners;
	//this function can only process Gray image
	goodFeaturesToTrack(imgGray,corners,20,0.01,10);
	Scalar color = Scalar(255,255,255);
	int radius{ 15 }, thickness{1};
	vector<Point>::const_iterator it = corners.begin();
	while (it!=corners.end())
	{
		circle(img1,*it,radius,color,thickness);
		++it;
	}
	imshow("dst",img1);

	vector<KeyPoint> keypoints1,keypoints2;
	//构造检测器
	//Ptr<FeatureDetector> detector = new ORB(120);
	Ptr<FeatureDetector> detector = new SIFT(80);
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);
	//构造描述子提取器
	Ptr<DescriptorExtractor> descriptor = detector;
	//提取描述子
	Mat descriptors1, descriptors2;
	
	descriptor->compute(img1,keypoints1,descriptors1);
	descriptor->compute(img2, keypoints2, descriptors2);
	//构造匹配器
	BFMatcher matcher(NORM_L2,true);
	//匹配描述子0
	vector<DMatch> matches;
	matcher.match(descriptors1,descriptors2,matches);

	//make the nth element at the nth position
	nth_element(matches.begin(),matches.begin()+20,matches.end());
	matches.erase(matches.begin()+20,matches.end());
	//新的关键点对
	vector<Point2f> selPoints1, selPoints2;
	vector<Point2f> temp1, temp2;
	vector<int> pointIndexes1, pointIndexes2;
	KeyPoint::convert(keypoints1,temp1,pointIndexes1);
	KeyPoint::convert(keypoints2, temp2, pointIndexes2);
	DMatch mach;
	for (int i=0; i<matches.size(); i++)
	{
		mach = matches.at(i);
		
		bool tt = i > 0 && (temp1.at(mach.queryIdx).x - temp1.at(matches.at(i-1).queryIdx).x) < 1;
		if (tt) continue;
		else
		{
			selPoints1.push_back(temp1.at(mach.queryIdx));
			selPoints2.push_back(temp2.at(mach.trainIdx));
		}	
	}
	vector<uchar> inliers(selPoints1.size(),0);
	Mat fundamental = findFundamentalMat(selPoints1,selPoints2,inliers,CV_FM_RANSAC,3.0,0.98);

	for (uchar i = 0; i < fundamental.rows;i++)
	{
		for (uchar j = 0; j < fundamental.cols; j++)
		{
			cout << fundamental.at<double>(j,i)<<'\t';
		}
		cout << '\n';
	}
	//计算对极线
	vector<Vec3f> lines1;
	computeCorrespondEpilines(selPoints1,1,fundamental,lines1);
	//遍历画出所有对极线
	for (vector<Vec3f>::const_iterator it = lines1.begin(); it != lines1.end();++it)
	{
		//calculate points's y coordinate value (x-coordinate value is 0,or cols,) by line formula a*x+b*y+c=0;
		line(img2,Point(0,-(*it)[2]/(*it)[1]),	Point(img2.cols,-((*it)[2]+(*it)[0]*img2.cols)/(*it)[1]),Scalar(0,255,0));
	}
	//根据RANSAC重新筛选匹配
	vector<DMatch> outMatches;
	vector<uchar>::const_iterator itIn = inliers.begin();
	vector<DMatch>::const_iterator itM = matches.begin();
	for (; itIn != inliers.end();++itIn,++itM)
	{
		if (*itIn)
		{
			outMatches.push_back(*itM);
		}

	}

	//画出匹配结果
	Mat matchImage;
	drawMatches(img1, keypoints1, img2, keypoints2, outMatches,matchImage,255,255);
	imshow("match",matchImage);
	
	//保存图像
	vector<int> cmprs_params;
	cmprs_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	cmprs_params.push_back(100);
	imwrite("Match&epiline1.jpg",matchImage,cmprs_params);

	waitKey(0);
	return 1;
}