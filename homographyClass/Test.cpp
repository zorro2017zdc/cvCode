#include"Homography.h"

int main()
{
	string imgPath1 = "trees_000.jpg";
	string imgPath2 = "trees_001.jpg";
	string imgPath3 = "trees_002.jpg";
	string imgPath4 = "trees_003.jpg";

	Mat img1 = imread(imgPath1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(imgPath2, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img3 = imread(imgPath3, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img4 = imread(imgPath4, CV_LOAD_IMAGE_GRAYSCALE);

	Mat img1_color = imread(imgPath1, CV_LOAD_IMAGE_COLOR);
	Mat img2_color = imread(imgPath2, CV_LOAD_IMAGE_COLOR);
	Mat img3_color = imread(imgPath3, CV_LOAD_IMAGE_COLOR);
	Mat img4_color = imread(imgPath4, CV_LOAD_IMAGE_COLOR);

	Homography homo12(img1,img2);
	Homography homo23(img1, img2);
	Homography homo34(img1, img2);

	//h12.drawMatches();

	Mat h12 = homo12.getHomography();
	Mat h23 = homo23.getHomography();
	Mat h34 = homo34.getHomography();

	Mat h21;
	Mat h32;
	Mat h43;

	invert(h12, h21, DECOMP_LU);
	invert(h23,h32, DECOMP_LU);
	invert(h34, h43, DECOMP_LU);

	Mat h31 = h21*h32;
	Mat h41 = h21*h32 * h43;

	Mat warp2;
	warpPerspective(img2_color, warp2, h21, Size(img1.cols * 4, img1.rows));
	Mat warp3;
	warpPerspective(img3_color, warp3, h31, Size(img1.cols * 4, img1.rows));
	Mat warp4;
	warpPerspective(img4_color, warp4, h41, Size(img1.cols * 4, img1.rows));

	imshow("warp2",warp2);
	imshow("warp3", warp3);
	imshow("warp4", warp4);
	imwrite("warp2.jpg", warp2);
	imwrite("warp3.jpg", warp3);
	imwrite("warp4.jpg", warp4);

	int d = img1.cols/2;
	int x2 = h21.at<double>(0,2)+d;
	int x3 = h21.at<double>(0, 2) + h32.at<double>(0, 2)+d;
	int x4 = h21.at<double>(0, 2) + h32.at<double>(0, 2) + h43.at<double>(0, 2)+d;

	Mat canvas(img1.rows,img1.cols*4,CV_8UC3);
	img1_color.copyTo(canvas(Range::all(), Range(0, img1.cols)));
	warp2(Range::all(), Range(x2, x3)).copyTo(canvas(Range::all(), Range(x2, x3)));
	warp3(Range::all(), Range(x3, x4 )).copyTo(canvas(Range::all(), Range(x3, x4)));
	warp4(Range::all(), Range(x4, x4 + img4.cols)).copyTo(canvas(Range::all(), Range(x4, x4 + img4.cols)));
	

	imwrite("canvas.jpg",canvas);
	imshow("canvas",canvas);
	cout << "Hello world!" << endl;

	waitKey(0);
	return 0;
}