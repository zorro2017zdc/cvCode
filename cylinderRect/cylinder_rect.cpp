// mapcyl.cpp : Defines the entry point for the console application.
//

#include <opencv/cv.h>
#include <opencv/highgui.h>

cv::Point2f convert_pt(cv::Point2f point, int w, int h)
{
	//center the point at 0,0
	cv::Point2f pc(point.x - w / 2, point.y - h / 2);

	//these are your free parameters
	//negative focal length since we are rear projecting
	float f = -w / 2;
	float r = w;

	float omega = w / 2;
	float z0 = f - sqrt(r*r - omega*omega);

	float zc = (2 * z0 + sqrt(4 * z0*z0 - 4 * (pc.x*pc.x / (f*f) + 1)*(z0*z0 - r*r))) / (2 * (pc.x*pc.x / (f*f) + 1));

	cv::Point2f final_point(pc.x*zc / f, pc.y*zc / f);
	final_point.x += w / 2;
	final_point.y += h / 2;

	return final_point;
}

int main()
{
	cv::Mat imgMat = cv::imread("lena.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	cv::copyMakeBorder(imgMat, imgMat, 175, 175, 175, 175, cv::BORDER_CONSTANT);

	cv::namedWindow("Original", CV_WINDOW_AUTOSIZE);
	cv::imshow("Original", imgMat);
	//cv::waitKey(0);

	cv::Mat destImgMat(imgMat.size(), CV_8U);

	for (int y = 0; y < imgMat.rows; y++)
	{
		for (int x = 0; x < imgMat.cols; x++)
		{
			cv::Point2f current_pos(x, y);

			current_pos = convert_pt(current_pos, imgMat.cols, imgMat.rows);

			cv::Point2i top_left((int)current_pos.x, (int)current_pos.y); //top left because of integer rounding

			//make sure the point is actually inside the original image
			if (top_left.x < 0 || top_left.x > imgMat.cols - 2 || top_left.y < 0 || top_left.y > imgMat.rows - 2)
			{
				continue;
			}

			//bilinear interpolation
			float dx = current_pos.x - top_left.x;
			float dy = current_pos.y - top_left.y;

			float weight_tl = (1.0 - dx) * (1.0 - dy);
			float weight_tr = (dx)* (1.0 - dy);
			float weight_bl = (1.0 - dx) * (dy);
			float weight_br = (dx)* (dy);

			uchar value = weight_tl * imgMat.at<uchar>(top_left) +
				weight_tr * imgMat.at<uchar>(top_left.y, top_left.x + 1) +
				weight_bl * imgMat.at<uchar>(top_left.y + 1, top_left.x) +
				weight_br * imgMat.at<uchar>(top_left.y + 1, top_left.x + 1);

			destImgMat.at<uchar>(y, x) = value;
		}
	}

	cv::namedWindow("Cylindrical", CV_WINDOW_AUTOSIZE);
	cv::imshow("Cylindrical", destImgMat);
	cv::waitKey(0);

	cv::imwrite("cyl_lena.jpg", destImgMat);

	return 0;
}

