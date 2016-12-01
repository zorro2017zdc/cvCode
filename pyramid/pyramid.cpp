#include"opencv2/opencv.hpp"
using namespace cv;

void buildGaussianPyramid(const Mat& base, vector<Mat>& pyr, int nOctaves);
void buildDoGPyramid(const vector<Mat>& gpyr, vector<Mat>& dogpyr);

int nOctaveLayers{ 3 };
double sigma{ 1.6 };

int main()
{
	Mat img = imread("ori.jpg");
	vector<Mat> gpyr,dogpyr;
	buildGaussianPyramid(img, gpyr, 2);
	buildDoGPyramid(gpyr,dogpyr);
	namedWindow("img", CV_WINDOW_AUTOSIZE);
	/* in this case gaussian pyramid has 12 layers,dog pyramid has 10 layers.
	For example,to access first layer DOG image,just use dogpyr[0] as follows */
	Mat img2 = dogpyr[0];
	imshow("img",img2);
	/**/
	waitKey(0);
	return 0;
}

// 构建nOctaves组（每组nOctaveLayers+3层）高斯金字塔  
void buildGaussianPyramid(const Mat& base, vector<Mat>& pyr, int nOctaves)
{
	vector<double> sig(nOctaveLayers + 3);
	pyr.resize(nOctaves*(nOctaveLayers + 3));

	// precompute Gaussian sigmas using the following formula:  
	//  \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2、  
	// 计算对图像做不同尺度高斯模糊的尺度因子  
	sig[0] = sigma;
	double k = pow(2., 1. / nOctaveLayers);
	for (int i = 1; i < nOctaveLayers + 3; i++)
	{
		double sig_prev = pow(k, (double)(i - 1))*sigma;
		double sig_total = sig_prev*k;
		sig[i] = std::sqrt(sig_total*sig_total - sig_prev*sig_prev);
	}

	for (int o = 0; o < nOctaves; o++)
	{
		// DoG金子塔需要nOctaveLayers+2层图像来检测nOctaves层尺度  
		// 所以高斯金字塔需要nOctaveLayers+3层图像得到nOctaveLayers+2层DoG金字塔  
		for (int i = 0; i < nOctaveLayers + 3; i++)
		{
			// dst为第o组（Octave）金字塔  
			Mat& dst = pyr[o*(nOctaveLayers + 3) + i];
			// 第0组第0层为原始图像  
			if (o == 0 && i == 0)
				dst = base;

			// base of new octave is halved image from end of previous octave  
			// 每一组第0副图像时上一组倒数第三幅图像隔点采样得到  
			else if (i == 0)
			{
				const Mat& src = pyr[(o - 1)*(nOctaveLayers + 3) + nOctaveLayers];
				resize(src, dst, Size(src.cols / 2, src.rows / 2),
					0, 0, INTER_NEAREST);
			}
			// 每一组第i副图像是由第i-1副图像进行sig[i]的高斯模糊得到  
			// 也就是本组图像在sig[i]的尺度空间下的图像  
			else
			{
				const Mat& src = pyr[o*(nOctaveLayers + 3) + i - 1];
				GaussianBlur(src, dst, Size(), sig[i], sig[i]);
			}
		}
	}
}

//构建DOG金字塔
void buildDoGPyramid(const vector<Mat>& gpyr, vector<Mat>& dogpyr)
{
	int nOctaves = (int)gpyr.size() / (nOctaveLayers + 3);
	dogpyr.resize(nOctaves*(nOctaveLayers + 2));

	for (int o = 0; o < nOctaves; o++)
	{
		for (int i = 0; i < nOctaveLayers + 2; i++)
		{
			// 第o组第i副图像为高斯金字塔中第o组第i+1和i组图像相减得到  
			const Mat& src1 = gpyr[o*(nOctaveLayers + 3) + i];
			const Mat& src2 = gpyr[o*(nOctaveLayers + 3) + i + 1];
			Mat& dst_abs = dogpyr[o*(nOctaveLayers + 2) + i];
			Mat dst;
			subtract(src2, src1, dst, noArray(), CV_16S);
			convertScaleAbs(dst,dst_abs);//将差分转换成正值，方便显示
			
		}
	}
}