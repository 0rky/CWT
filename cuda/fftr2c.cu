#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>


#include<cufft.h>

#include "cv.h"
#include "highgui.h"

int main()
{
	int i,row,col,count=0;;
	cufftReal *signal;
	cufftReal *d_signal,*d_result;
	cufftComplex *d_inverse;
	cufftHandle plan_c2r,plan_r2c;	
	unsigned int img_dim;

	IplImage* img=cvLoadImage("sample.jpg",CV_LOAD_IMAGE_COLOR);
	IplImage* gray_img=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);

	img_dim=gray_img->width*gray_img->height;
	cvCvtColor(img,gray_img,CV_RGB2GRAY);

	signal=(cufftReal*)malloc(sizeof(cufftReal)*img_dim);

	for(row=0;row<gray_img->height;row++)
	{
		const uchar* ptr=(const uchar*)(gray_img->imageData+row*gray_img->widthStep);
		for(col=0;col<gray_img->width;col++)
		{
			signal[count]=*ptr++;
			count++;
		}
	}

	cudaMalloc((void**)&d_signal,sizeof(cufftReal)*img_dim);
	cudaMalloc((void**)&d_result,sizeof(cufftReal)*img_dim);
	cudaMalloc((void**)&d_inverse,sizeof(cufftComplex)*(gray_img->width)*(gray_img->height/2+1));

	//inverse=(cufftComplex*)malloc(sizeof(cufftComplex)*NUM_POINTS);

	cufftPlan2d(&plan_c2r,gray_img->width,gray_img->height,CUFFT_C2R);	
	cufftPlan2d(&plan_r2c,gray_img->width,gray_img->height,CUFFT_R2C);

	cudaMemcpy(d_signal,signal,sizeof(cufftReal)*img_dim,cudaMemcpyHostToDevice);

	cufftExecR2C(plan_r2c, d_signal,d_inverse);
	
	//forward_fft<<<NUM_POINTS,1>>>(d_signal,d_inverse);

	//cudaMemcpy(inverse,d_inverse,(cufftComplex)*NUM_POINTS,cudaMemcpyDeviceToHost);

	for(i=0;i<64;i++)
		printf("[%f]",signal[i]);

	cufftExecC2R(plan_c2r,d_inverse,d_result);

	cudaMemcpy(signal,d_result,sizeof(cufftReal)*img_dim,cudaMemcpyDeviceToHost);	
printf("\n______________________________________________\n");
		
	for(i=0;i<64;i++)
		printf("[%f]",signal[i]/img_dim);	

	cufftDestroy(plan_c2r);
	cufftDestroy(plan_r2c);
	cudaFree(d_signal);
	cudaFree(d_result);
	cudaFree(d_inverse);
	cudaFree(signal);
	//free(inverse);
	return 0;
}
