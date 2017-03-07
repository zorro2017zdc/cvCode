#pragma once
# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/imgproc/imgproc.hpp"
#include"opencv2/nonfree/nonfree.hpp"
#include"opencv2/calib3d/calib3d.hpp"
#include<iostream>
using namespace cv;
using namespace std;

class Homography
{
private:
	Mat img1;
	Mat img2;

	Ptr<FeatureDetector> detector;
	Ptr<DescriptorExtractor> extractor;
	Ptr<DescriptorMatcher> matcher;

	vector<KeyPoint> keyPoints1;
	vector<KeyPoint> keyPoints2;

	Mat descriptors1;
	Mat descriptors2;

	vector<DMatch> firstMatches;
	vector<DMatch> matches;

	vector<Point2f> selfPoints1;
	vector<Point2f> selfPoints2;

	vector<uchar> inliers;

	Mat homography;

public:
	Homography();
	Homography(Mat img1, Mat img2) ;

	void setFeatureDetector(string detectorName);
	void setDescriptorExtractor(string descriptorName);
	void setDescriptorMatcher(string matcherName);

	vector<KeyPoint> getKeyPoints1();
	vector<KeyPoint> getKeyPoints2();

	Mat getDescriptors1();
	Mat getDescriptors2();

	vector<DMatch> getMatches();
	void drawMatches();

	Mat getHomography();

	~Homography();

private:
	void detectKeyPoints();
	void computeDescriptors();
	void match();
	void matchesToSelfPoints();
	void findHomography();
	void matchesFilter();
};

