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
	Mat img1_c = imread("window_view_2.jpg");
	Mat img2_c = imread("window_view_1.jpg");
	Mat img1,img2;
	cvtColor(img1_c,img1,CV_RGB2GRAY);
	cvtColor(img2_c, img2, CV_RGB2GRAY);

	cvNamedWindow("match",1);
	cvNamedWindow("perspective", 1);

	double t1 = (double)getTickCount();

	//Mat img1 = imread("frame2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	//Mat img2 = imread("frame1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	if (!img1.data || !img2.data)
	{
		printf(" --(!) Error reading images \n"); return -1;
	}

	vector<KeyPoint> keypoints1, keypoints2;
	//构造检测器
	//Ptr<FeatureDetector> detector = new ORB(120);
	Ptr<FeatureDetector> detector = new SIFT(80);
	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);
	//构造描述子提取器
	Ptr<DescriptorExtractor> descriptor = detector;
	//提取描述子
	Mat descriptors1, descriptors2;

	descriptor->compute(img1, keypoints1, descriptors1);
	descriptor->compute(img2, keypoints2, descriptors2);
	//构造匹配器
	BFMatcher matcher(NORM_L2, true);
	//匹配描述子
	vector<DMatch> matches;
	matcher.match(descriptors1, descriptors2, matches);
	
	vector<Point2f> selPoints1, selPoints2;
	vector<int> pointIndexes1, pointIndexes2;
	for (vector<DMatch>::const_iterator it = matches.begin(); it!=matches.end(); ++it)
	{
		selPoints1.push_back(keypoints1.at(it->queryIdx).pt);
		selPoints2.push_back(keypoints2.at(it->trainIdx).pt);
	}
	
	vector<uchar> inliers(selPoints1.size(), 0);
	Mat homography = findHomography(selPoints1, selPoints2, inliers, CV_FM_RANSAC,1.0);

	//根据RANSAC重新筛选匹配
	vector<DMatch> outMatches;
	vector<uchar>::const_iterator itIn = inliers.begin();
	vector<DMatch>::const_iterator itM = matches.begin();
	for (; itIn != inliers.end(); ++itIn, ++itM)
	{
		if (*itIn)
		{
			outMatches.push_back(*itM);
		}

	}

	//画出匹配结果
	Mat matchImage;
	drawMatches(img1, keypoints1, img2, keypoints2, outMatches, matchImage, 255, 255);
	imshow("match", matchImage);

	//保存图像
	/*vector<int> cmprs_params;
	cmprs_params.push_back(CV_IMWRITE_JPEG_QUALITY);
	cmprs_params.push_back(100);
	imwrite("Match&epiline1.jpg", matchImage, cmprs_params); */
	t1 = ((double)getTickCount()-t1) / getTickFrequency();
	cout <<"align compute time: "<< t1 << endl;

	double t3 = getTickCount();
	Mat result;
	warpPerspective(img1_c, result, homography, Size(2 * img1_c.cols-200, img1_c.rows));//200可调
	t3 = ((double)getTickCount() - t3) / getTickFrequency();
	cout << "perspective time: " << t3 << endl;
	//线性融合
	double t2 = (double)getTickCount();
	int d = 200;
	Mat half(result, Rect(0, 0, img2_c.cols-d , img2_c.rows));
	img2_c(Range::all(), Range(0, img2_c.cols-d )).copyTo(half);
	for (int i = 0; i < d; i++)
	{
		result.col(img2_c.cols - d + i) = (d - i) / (float)d*img2_c.col(img2_c.cols - d + i) + i / (float)d*result.col(img2_c.cols - d + i);
	}
	
	t2 = ((double)getTickCount() - t2) / getTickFrequency();
	imshow("perspective", result);

	cout << "blend time: " << t2 << endl;

	waitKey(0);
	return 1;
}