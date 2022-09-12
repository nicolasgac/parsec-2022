#include "opencv2/opencv.hpp"

// includes CUDA
#include <cuda_runtime.h>

#include <omp.h>

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

	VideoCapture cap(0); // open the default camera
	if(!cap.isOpened())  // check if we succeeded
		return -1;


		int num_devices,num_procs,num_threads;
		omp_set_num_threads(4);
		num_procs=omp_get_num_procs();
		printf("nombre de procs accessibles par OPENMP : %d\n",num_procs);


#pragma omp parallel for
	for (int i=0;i<100;i++){
		printf("it�ration %d traité par le thread numéro : %d\n",i,omp_get_thread_num());
	}

	waitKey(0);


	for(;;)
	{
		Mat frame,frame_out;
		// Record the start event
		error = cudaEventRecord(start, NULL);
		cap >> frame; // get a new frame from camera
frame_out.create(frame.rows,frame.cols,frame.type());
		imshow("input", frame);
		// Record the start_compute event



		Mat framefiltremoyenne;
		framefiltremoyenne.create(5,5,CV_32F);

		int rayon_rows,rayon_cols;
		rayon_rows=(framefiltremoyenne.rows-1)/2;
		rayon_cols=(framefiltremoyenne.cols-1)/2;
		for(int l=0;l<2*rayon_rows+1;l++){
			for(int c=0;c<2*rayon_cols+1;c++){
				framefiltremoyenne.at<float>(l,c)=1.0/(framefiltremoyenne.rows*framefiltremoyenne.cols);
			}
		}
		error = cudaEventRecord(start_compute, NULL);
#pragma omp parallel for
for (int i=1;i<frame.rows-1;i++){
	for (int j=1;j<frame.cols-1;j++){
		float sum_b=0,sum_r=0,sum_g=0;
		       		for(int l=-rayon_rows;l<rayon_rows+1;l++){
		       	    for(int c=-rayon_cols;c<rayon_cols+1;c++){
		       	    	/*sum_b += framefiltremoyenne.at<float>(l,c)*(float)(frame.at<Vec3b>(i+l,j+c)[0]);
		       	    	sum_g += framefiltremoyenne.at<float>(l,c)*(float)(frame.at<Vec3b>(i+l,j+c)[1]);
		       	    	sum_r += framefiltremoyenne.at<float>(l,c)*(float)(frame.at<Vec3b>(i+l,j+c)[2]);*/
		       	    	sum_b += 1.0/(framefiltremoyenne.rows*framefiltremoyenne.cols)*(frame.at<Vec3b>(i+l,j+c)[0]);//framefiltremoyenne.at<float>(l,c)*(float)(frame->at<Vec3b>(i+l,j+c)[0]);
		       	    	sum_g += 1.0/(framefiltremoyenne.rows*framefiltremoyenne.cols)*(frame.at<Vec3b>(i+l,j+c)[1]);//framefiltremoyenne.at<float>(l,c)*(float)(frame->at<Vec3b>(i+l,j+c)[1]);
		       	    	sum_r += 1.0/(framefiltremoyenne.rows*framefiltremoyenne.cols)*(frame.at<Vec3b>(i+l,j+c)[2]);//framefiltremoyenne.at<float>(l,c)*(float)(frame->at<Vec3b>(i+l,j+c)[2]);
		       }
		       }
		       		frame_out.at<Vec3b>(i,j)[0]=sum_b;
		       		frame_out.at<Vec3b>(i,j)[1]=sum_g;
		       		frame_out.at<Vec3b>(i,j)[2]=sum_r;

	}
}
// Record the stop_compute event
error = cudaEventRecord(stop_compute, NULL);
//	// Wait for the stop event to complete
error = cudaEventSynchronize(stop_compute);
error = cudaEventElapsedTime(&msecTotal, start_compute, stop_compute);
printf("Compute Time : %f msec\n",msecTotal);


imshow("output", frame_out);

// Record the stop event
error = cudaEventRecord(stop, NULL);
//	// Wait for the stop event to complete
error = cudaEventSynchronize(stop);

error = cudaEventElapsedTime(&msecTotal, start, stop);
printf("Total Time : %f msec\n",msecTotal);

if(waitKey(1) >= 0) break;
	}
	// the camera will be deinitialized automatically in VideoCapture destructor
	return 0;




	return 0;
}
