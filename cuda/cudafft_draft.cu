/*************************************************************************************/
/********************continous wavelet transform**************************************/
/* This is the main program for the CWT calculation, it uses fft technique to compute
the cwt coefficients. 

Author: Manas Jyoti Das, July:02:2016 */

#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include "cv.h"
#include "highgui.h"
#include<cufft.h>
#include "filter.cuh"
#include "normfilter.cuh"
#include<sys/time.h>

float *filter_dx,*filter_dy;

int main(void)
{
	IplImage* img=cvLoadImage("sample.jpg",CV_LOAD_IMAGE_COLOR);
	IplImage* gray_img=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
	
	int i,temp=0,row,col,count=0;
	unsigned int img_dim;
	cufftReal *signal;
	float *filter_dx_rearrange,*filter_dy_rearrange,*inverse_dx,*inverse_dy;
	cufftReal *d_signal,*d_result;
	struct timeval t0,t1;
	long double elapsed;
	cufftComplex *d_inverse,*mult_filt_data,*data_inverse;
	float scale[16];
	float temp1;
	
	img_dim=gray_img->width*gray_img->height;

	cufftHandle plan_c2r,plan_r2c;

	cvCvtColor(img,gray_img,CV_RGB2GRAY);

	gettimeofday(&t0,0);	

	filter_dx=(float*)malloc(sizeof(float)*img_dim);
	filter_dy=(float*)malloc(sizeof(float)*img_dim);
	filter_dx_rearrange=(float*)malloc(sizeof(float)*(gray_img->height*(gray_img->width/2+1)));	
	filter_dy_rearrange=(float*)malloc(sizeof(float)*(gray_img->height*(gray_img->width/2+1)));	
	mult_filt_data=(cufftComplex*)malloc(sizeof(cufftComplex)*(gray_img->height*(gray_img->width/2+1)));
	inverse_dx=(float*)malloc(sizeof(float)*img_dim);
	inverse_dy=(float*)malloc(sizeof(float)*img_dim);
	data_inverse=(cufftComplex*)malloc(sizeof(cufftComplex)*(gray_img->height*(gray_img->width/2+1)));

	signal=(cufftReal*)malloc(sizeof(cufftReal)*img_dim);
	
	cudaMalloc((void**)&d_signal,sizeof(cufftReal)*img_dim);
	cudaMalloc((void**)&d_result,sizeof(cufftReal)*img_dim);
	cudaMalloc((void**)&d_inverse,sizeof(cufftComplex)*(gray_img->width)*(gray_img->height/2+1));
	
	for(temp1=1;temp1<4.1;temp1+=0.2)
	{
		scale[count]=pow(2,temp1); printf("scale is %f",scale[count]);
		count++;
	}
	count=0;
	cufftPlan2d(&plan_c2r,gray_img->width,gray_img->height,CUFFT_C2R);	
	cufftPlan2d(&plan_r2c,gray_img->width,gray_img->height,CUFFT_R2C);	

	for(row=0;row<gray_img->height;row++)
	{
		const uchar* ptr=(const uchar*)(gray_img->imageData+row*gray_img->widthStep);
		for(col=0;col<gray_img->width;col++)
		{
			signal[count]=*ptr++;
			count++;
		}
	}

	cudaMemcpy(d_signal,signal,sizeof(cufftReal)*img_dim,cudaMemcpyHostToDevice);	

	cufftExecR2C(plan_r2c, d_signal,d_inverse);

	cudaMemcpy(data_inverse,d_inverse,sizeof(cufftComplex)*(gray_img->width)*(gray_img->height/2+1),cudaMemcpyDeviceToHost);
	
	for(temp=0;temp<16;temp++)
	{

	filter(filter_dx,filter_dy,gray_img->height,gray_img->width,scale[temp]);	

	memcpy(filter_dx_rearrange,filter_dx,sizeof(int)*(gray_img->width/2+1));
	memcpy(filter_dy_rearrange,filter_dy,sizeof(int)*(gray_img->width/2+1));

	for(i=1;i<=(gray_img->height-1);i++)
	{
		memcpy(filter_dx_rearrange+((i-1)*(gray_img->width/2+1)+(gray_img->width/2+1)),filter_dx+i*gray_img->width,sizeof(int)*(gray_img->width/2+1));
		memcpy(filter_dy_rearrange+((i-1)*(gray_img->width/2+1)+(gray_img->width/2+1)),filter_dy+i*gray_img->width,sizeof(int)*(gray_img->width/2+1));
	}

	//********** filter dx multiplication with data start *********//
	
	for(i=0;i<(gray_img->height*(gray_img->width/2+1));i++)
	{
		mult_filt_data[i].x= -filter_dx_rearrange[i]*data_inverse[i].y;
		mult_filt_data[i].y= filter_dx_rearrange[i]*data_inverse[i].x;
	}

	cudaMemcpy(d_inverse,mult_filt_data,sizeof(cufftComplex)*(gray_img->height*(gray_img->width/2+1)),cudaMemcpyHostToDevice);

	cufftExecC2R(plan_c2r,d_inverse,d_signal);

	cudaMemcpy(signal,d_signal,sizeof(cufftReal)*img_dim,cudaMemcpyDeviceToHost);

	for(i=0;i<img_dim;i++)
	{
		inverse_dx[i]=signal[i]/img_dim;
	}
	//********** filter dx multiplication with data over *********//

	//********** filter dy multiplication with data start *********//

	for(i=0;i<(gray_img->height*(gray_img->width/2+1));i++)
	{
		mult_filt_data[i].x= -filter_dy_rearrange[i]*data_inverse[i].y;
		mult_filt_data[i].y= filter_dy_rearrange[i]*data_inverse[i].x;
	}

	cudaMemcpy(d_inverse,mult_filt_data,sizeof(cufftComplex)*(gray_img->height*(gray_img->width/2+1)),cudaMemcpyHostToDevice);

	cufftExecC2R(plan_c2r,d_inverse,d_signal);

	cudaMemcpy(signal,d_signal,sizeof(cufftReal)*img_dim,cudaMemcpyDeviceToHost);

	for(i=0;i<img_dim;i++)
	{
		inverse_dy[i]=signal[i]/img_dim;
	}
	//********** filter dx multiplication with data over *********//

	//reusing filter_dx to hold magnitude and filter_dy to hold the angle, or i can use aliasing which is a good idea//
	for(i=0;i<img_dim;i++)
	{
		signal[i]=sqrt(pow(abs(inverse_dx[i]),2)+pow(abs(inverse_dy[i]),2));
	}
	
printf(" the complex number is %fi\n",signal[0]);

	}

	gettimeofday(&t1, 0);
	elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
	printf("\n time wall %Lf\n",elapsed/1000000);
	//free(filter_dx);
	//free(filter_dy);
	
	//cvReleaseImage(&img);
	//cvReleaseImage(&gray_img);
	
	return 0;
	
}
