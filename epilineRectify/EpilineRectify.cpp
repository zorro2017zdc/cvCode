# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/features2d.hpp"
#include"opencv2/calib3d/calib3d.hpp"
#include"opencv2/imgproc/imgproc.hpp"
using namespace cv;

int main()
{
	Mat img_1 = imread("trees_001.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_2 = imread("trees_002.jpg", CV_LOAD_IMAGE_GRAYSCALE);
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
	//-- Draw only "good" matches
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2,
		good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
		vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("image1", img_1);
	imshow("image2", img_2);
	//imshow("Good Matches", img_matches);

	//find two picture's fundamental mat
	std::vector<Point2f> pt1, pt2;
	for (int i = 0; i < good_matches.size(); i++)
	{
		//Get key point from good matches
		pt1.push_back(keypoints_1[good_matches[i].queryIdx].pt);
		pt2.push_back(keypoints_2[good_matches[i].trainIdx].pt);
	}
	Mat F = findFundamentalMat(pt1, pt2, CV_FM_RANSAC, 3.0, 0.99);

	//calculate the homography mat for calibration
	Mat H1, H2;
	stereoRectifyUncalibrated(pt1, pt2, F, Size(img_1.cols, img_1.rows), H1, H2, 5);



	//give a inner_parameter matrix freely
	Mat	K = Mat::eye(3, 3, CV_64F);
	Mat invK = K.inv(DECOMP_SVD);
	Mat R1 = invK*H1*K;//calculate first camera transform Matrix in space based on homography matrix
	Mat R2 = invK*H2*K;//calculate second camera transform matrix in space
	Mat Map11, Map12;
	Mat Map21, Map22;
	Mat Distort;//径向畸变为0，设为NULL matrix
	Size UndistSize(img_1.cols, img_2.rows);
	//计算左右图像的映射矩阵
	initUndistortRectifyMap(K, Distort, R1, K, UndistSize, CV_32FC1, Map11, Map12);
	initUndistortRectifyMap(K, Distort, R2, K, UndistSize, CV_32FC1, Map21, Map22);
	//把原始图像投影到新图像上，得到校正图像
	Mat RectyImage1;
	Mat RectyImage2;

	remap(img_1, RectyImage1, Map11, Map12, INTER_LINEAR);
	remap(img_2, RectyImage2, Map21, Map22, INTER_LINEAR);
	imshow("rectify1", RectyImage1);
	imshow("rectify2", RectyImage2);

	waitKey(0);
	return 0;
}