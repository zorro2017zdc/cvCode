#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;

int main()
{
	Mat img1 = imread("yard1.jpg");
	if (!img1.data)
	{
		cout << "oh,no.open failure" << endl;
		return -1;
	}
	namedWindow("picture1", CV_WINDOW_AUTOSIZE);
	namedWindow("picture2", CV_WINDOW_AUTOSIZE);

	imshow("picture1", img1);
	Mat img2 = imread("yard2.jpg");

	imshow("picture2", img2);
	//新建一个名为expanded的Mat容器。高度和img1相同，宽度为两倍
	Mat expanded(Size((img1.cols + img2.cols), img1.rows), CV_8UC3);
	cout << "expanded.cols" << expanded.cols << endl;
	cout << "expanded.rows" << expanded.rows << endl;

	Mat ROI = expanded(Rect(0, 0, img1.cols, img1.rows));
	cout << "ROI.cols" << ROI.cols << endl;
	cout << "ROI.rows" << ROI.rows << endl;

	Mat ROI1 = expanded(Rect(img1.cols, 0, img2.cols, img2.rows));
	//cout << "ROI1.cols" << ROI1.cols << endl;
	//cout << "ROI1.rows" << ROI1.rows << endl;
	addWeighted(ROI, 0, img1, 1, 0., ROI);
	addWeighted(ROI1, 0, img2, 1, 0., ROI1);

	namedWindow("picture");
	imshow("picture", expanded);
	waitKey(100000);
	return 0;
}
