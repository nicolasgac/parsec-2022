#include "opencv2/opencv.hpp"

// includes CUDA
#include <cuda_runtime.h>

using namespace cv;


int main(int, char**)
{
	cudaError_t error;
		float msecTotal = 0.0f;

		cudaEvent_t start,stop,start_compute,stop_compute;
		error = cudaEventCreate(&start);
		error = cudaEventCreate(&stop);
		error = cudaEventCreate(&start_compute);
		error = cudaEventCreate(&stop_compute);

		Mat vol_in;
		Mat vol_out;


		vol_in = imread("/home/parsec_01/TP_GPU/TP_seuillage/Images/ferrari.jpg");

		vol_out.create(vol_in.rows,vol_in.cols,CV_8UC(3));

		namedWindow("input",WINDOW_AUTOSIZE);
		namedWindow("output",WINDOW_AUTOSIZE);

		imshow("input", vol_in);

		for (int i=0;i<vol_in.rows;i++){
			for (int j=0;j<vol_in.cols;j++){
				uchar r,v,b;

				b=vol_in.at<Vec3b>(i,j)[0];
				v=vol_in.at<Vec3b>(i,j)[1];
				r=vol_in.at<Vec3b>(i,j)[2];

				vol_out.at<Vec3b>(i,j)[0]=0;
				vol_out.at<Vec3b>(i,j)[1]=0;
				vol_out.at<Vec3b>(i,j)[2]=r;

			}
		}

		imshow("output", vol_out);

		waitKey(0);

		return 0;
}
