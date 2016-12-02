# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/features2d.hpp"
#include"opencv2/imgproc/imgproc.hpp"
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
	Mat img_1 = imread("window_view_2.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	equalizeHist(img_1,img_1);
	Mat img_2 = imread("window_view_1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	equalizeHist(img_2, img_2);
	if (!img_1.data || !img_2.data)
	{
		printf(" --(!) Error reading images \n"); return -1;
	}
	//-- Step 1: Detect the keypoints using SURF Detector
	int minHessian = 400;
	SurfFeatureDetector detector(minHessian);
	std::vector<KeyPoint> keypoints_1, keypoints_2;
	detector.detect(img_1, keypoints_1);
	detector.detect(img_2, keypoints_2);
	//-- Step 2: Calculate descriptors (feature vectors)
	SurfDescriptorExtractor extractor;
	Mat descriptors_1, descriptors_2;
	extractor.compute(img_1, keypoints_1, descriptors_1);
	extractor.compute(img_2, keypoints_2, descriptors_2);
	//-- Step 3: Matching descriptor vectors using FLANN matcher
	FlannBasedMatcher matcher;
	std::vector< DMatch > matches;
	matcher.match(descriptors_1, descriptors_2, matches);
	double max_dist = 0; double min_dist = 100;
	//-- Quick calculation of max and min distances between keypoints
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);
	//-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	//-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	//-- small)
	//-- PS.- radiusMatch can also be used here.
	std::vector< DMatch > good_matches;
	for (int i = 0; i < descriptors_1.rows; i++)
	{
		if (matches[i].distance <= max(2 * min_dist, 0.02))
		{
			good_matches.push_back(matches[i]);
		}
	}
	//计算单应矩阵
	vector<Point2f> selPoints1, selPoints2;
	//vector<int> pointIndexes1, pointIndexes2;
	for (vector<DMatch>::const_iterator it = matches.begin(); it != matches.end(); ++it)
	{
		selPoints1.push_back(keypoints_1.at(it->queryIdx).pt);
		selPoints2.push_back(keypoints_2.at(it->trainIdx).pt);
	}

	vector<uchar> inliers(selPoints1.size(), 0);
	Mat homography = findHomography(selPoints1, selPoints2, inliers, CV_FM_RANSAC, 1.0);

	//
	//拼合
	Mat result;
	warpPerspective(img_1, result, homography, Size(2 * img_1.cols, img_1.rows));
	Mat half(result, Rect(0, 0, img_2.cols, img_2.rows));
	img_2.copyTo(half);
	imshow("perspective", result);
	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),vector<char>(),DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//-- Show detected matches
	imshow("Good Matches", img_matches);
	//写入
	/*vector<int> cmpres_para;
	cmpres_para.push_back(CV_IMWRITE_JPEG_QUALITY);
	cmpres_para.push_back(100);
	imwrite("match.jpg",img_matches,cmpres_para);*/

	for (int i = 0; i < (int)good_matches.size(); i++)
	{
		printf("-- Good Match [%d] Keypoint 1: %d -- Keypoint 2: %d \n", i, good_matches[i].queryIdx, good_matches[i].trainIdx);
	}

	//显示单应矩阵
	cout << endl;
	for (int i = 0; i < homography.rows; i++)
	{
		for (int j = 0; j < homography.cols; j++)
		{
			cout << homography.at<double>(i, j) << '\t';
		}
		cout << endl;
	}
	waitKey(0);
	return 0;
}