#include "cv.h"
#include "cxmisc.h"
#include "highgui.h"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cvaux.h"
#include "CompConfidence.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <iostream>  
#include <fstream>  
#include <iterator>  

using namespace std;
using namespace cv;

/* This file is modified OpenCV stereo calibration sample */
/***************************
 Image rectification, disparity map and confidence cues computation 
 Input:
  - Config.xml : Configuration file
  - ImgList.xml : Video or image list
  - intrinsics.yml extrinsics.yml : Calibration parameters
  - sgbmPara.yml : SGBM parameters for disparity map computation
***************************/
/* Data: 3/17/2014 */


/* This is sample from the OpenCV book. The copyright notice is below */

/* *************** License:**************************
   Oct. 3, 2008
   Right to use this code in any way you want without warranty, support or any guarantee of it working.

   BOOK: It would be nice if you cited it:
   Learning OpenCV: Computer Vision with the OpenCV Library
     by Gary Bradski and Adrian Kaehler
     Published by O'Reilly Media, October 3, 2008

   AVAILABLE AT:
     http://www.amazon.com/Learning-OpenCV-Computer-Vision-Library/dp/0596516134
     Or: http://oreilly.com/catalog/9780596516130/
     ISBN-10: 0596516134 or: ISBN-13: 978-0596516130

   OTHER OPENCV SITES:
   * The source code is on sourceforge at:
     http://sourceforge.net/projects/opencvlibrary/
   * The OpenCV wiki page (As of Oct 1, 2008 this is down for changing over servers, but should come back):
     http://opencvlibrary.sourceforge.net/
   * An active user group is at:
     http://tech.groups.yahoo.com/group/OpenCV/
   * The minutes of weekly OpenCV development meetings are at:
     http://code.opencv.org/projects/opencv/wiki/Meeting_notes
   ************************************************** */

static int print_help()
{
#if 0
    cout <<
            " Given a list of chessboard images, the number of corners (nx, ny)\n"
            " on the chessboards, and a flag: useCalibrated for \n"
            "   calibrated (0) or\n"
            "   uncalibrated \n"
            "     (1: use cvStereoCalibrate(), 2: compute fundamental\n"
            "         matrix separately) stereo. \n"
            " Calibrate the cameras and display the\n"
            " rectified results along with the computed disparity images.   \n" << endl;
    cout << "Usage:\n ./stereo_calib -w board_width -h board_height [-nr /*dot not view results*/] <image list XML/YML file>\n" << endl;
#endif
    return 0;
}


static void StereoDisparity()
{
    // ARRAY AND VECTOR STORAGE:

    Size imageSize;
    int i, j, k;
	Mat cameraMatrix[2], distCoeffs[2];
    cameraMatrix[0] = Mat::eye(3, 3, CV_64F);
    cameraMatrix[1] = Mat::eye(3, 3, CV_64F);
    Mat R, T, E, F;
	Mat R1, R2, P1, P2, Q;
    Rect validRoi[2];
	char c;

	// Read data from yml files
	// Intrinsic
	FileStorage fs_in("intrinsics.yml", CV_STORAGE_READ);
	if(!fs_in.isOpened())
	{
		cout<<"Can not open intrinsics.yml!"<<endl;
		return;
	}
	fs_in["M1"]>>cameraMatrix[0];
	fs_in["D1"]>>distCoeffs[0];
	fs_in["M2"]>>cameraMatrix[1];
	fs_in["D2"]>>distCoeffs[1];
	fs_in.release();

	// Extrinsic
	FileStorage fs_ex("extrinsics.yml", CV_STORAGE_READ);
	if(!fs_ex.isOpened())
	{
		cout<<"Can not open extrinsics.yml!"<<endl;
		return;
	}
	fs_ex["R"]>>R;
	fs_ex["T"]>>T;
	fs_ex["ImageSizeHeight"]>>imageSize.height;
	fs_ex["ImageSizeWidth"]>>imageSize.width;
	fs_ex.release();

	stereoRectify(cameraMatrix[0], distCoeffs[0],
                  cameraMatrix[1], distCoeffs[1],
                  imageSize, R, T, R1, R2, P1, P2, Q,
                  CALIB_ZERO_DISPARITY, 0, imageSize, &validRoi[0], &validRoi[1]);

	int FLAG_SHOW_RECTIRY, FLAG_SHOW_DISPARITY, FLAG_SAVE_DISPARITY, FLAG_CONFIDENCE;
	string SavePath;
	FileStorage fs("Config.xml", CV_STORAGE_READ);
	if(!fs.isOpened())
	{
		cout<<"Can not open Config.xml!"<<endl;
		return;
	}
	fs["ShowRectify"]>>FLAG_SHOW_RECTIRY;
	fs["ShwoDisparity"]>>FLAG_SHOW_DISPARITY;
	fs["SaveDisparity"]>>FLAG_SAVE_DISPARITY;
	fs["ComputeConfidence"]>>FLAG_CONFIDENCE;
	fs["SavePath"]>>SavePath;
//	cout<<FLAG_SHOW_RECTIRY<<" "<<FLAG_SHOW_DISPARITY<<" "<<FLAG_SAVE_DISPARITY<<" "<<FLAG_CONFIDENCE<<endl;

	fs.open("sgbmPara.yml", CV_STORAGE_READ);
	if(!fs.isOpened())
	{
		cout<<"Can not open sgbmPare.yml!"<<endl;
		return;
	}
	StereoSGBM_COST sgbm;
	fs["preFilterCap"]>>sgbm.preFilterCap;
	fs["SADWindowSize"]>>sgbm.SADWindowSize;
	fs["P1"]>>sgbm.P1;
	fs["P2"]>>sgbm.P2;
	fs["minDisparity"]>>sgbm.minDisparity;
	fs["numberOfDisparities"]>>sgbm.numberOfDisparities;
	fs["uniquenessRatio"]>>sgbm.uniquenessRatio;
	fs["speckleWindowSize"]>>sgbm.speckleWindowSize;
	fs["speckleRange"]>>sgbm.speckleRange;
	fs["disp12MaxDiff"]>>sgbm.disp12MaxDiff;
	fs["fullDP"]>>sgbm.fullDP;

		Mat left_im,right_im,disp_sgbm,disp_cost,disp_mincost,vdisp;

    // OpenCV can handle left-right
    // or up-down camera arrangements
    bool isVerticalStereo = fabs(P2.at<double>(1, 3)) > fabs(P2.at<double>(0, 3));

// COMPUTE AND DISPLAY RECTIFICATION
    Mat rmap[2][2];

    //Precompute maps for cv::remap()
    initUndistortRectifyMap(cameraMatrix[0], distCoeffs[0], R1, P1, imageSize, CV_16SC2, rmap[0][0], rmap[0][1]);
    initUndistortRectifyMap(cameraMatrix[1], distCoeffs[1], R2, P2, imageSize, CV_16SC2, rmap[1][0], rmap[1][1]);

    Mat canvas;
    double sf;
    int w, h;
    if( !isVerticalStereo )
    {
        sf = 600./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h, w*2, CV_8UC3);
    }
    else
    {
        sf = 300./MAX(imageSize.width, imageSize.height);
        w = cvRound(imageSize.width*sf);
        h = cvRound(imageSize.height*sf);
        canvas.create(h*2, w, CV_8UC3);
    }

    vector<string> ImgList;
	string ImgFormat,ImgPath,left_name,right_name,left_video_name,right_video_name;

	int ImageOrVideo;

	fs.open("ImgList.xml", CV_STORAGE_READ);
	if(!fs.isOpened())
	{
		cout<<"Can not open ImgList.xml!"<<endl;
		return;
	}
	fs["imagelist"]>>ImgList;
	fs["imageformat"]>>ImgFormat;
	fs["inputpath"]>>ImgPath;
	fs["leftname"]>>left_name;
	fs["rightname"]>>right_name;
	fs["imageorvideo"]>>ImageOrVideo;
	fs["leftvideo"]>>left_video_name;
	fs["rightvideo"]>>right_video_name;
	string ImgName;
	VideoCapture left_v,right_v;
	int NumImg = (int)ImgList.size();

	// Loading videos
	if(!ImageOrVideo)
	{
		ImgName = ImgPath + left_video_name;
		left_v.open(ImgName);
		cout<<"Load video: "<<ImgName<<endl;
		
		ImgName = ImgPath + right_video_name;
		right_v.open(ImgName);
		cout<<"Load video: "<<ImgName<<endl;
	}

	
	bool sign = true;
	i = 0;
	string ImageID;
    while(1)
    {
		if(!ImageOrVideo)
		{
			ImageID = to_string((long long)i);
			cout<<ImageID<<endl;
		}else
		{
			ImageID = ImgList[i];
		}
        for( k = 0; k < 2; k++ )
        {
			Mat rimg, cimg, img;

			if(ImageOrVideo)
			{
				if(k==0)
				{
					ImgName = ImgPath + left_name + ImageID + "." + ImgFormat;
					img = imread(ImgName, 0);
				}else
				{
					ImgName = ImgPath + right_name + ImageID + "." + ImgFormat;
					img = imread(ImgName, 0);
				}
				cout<<"Load image: "<<ImgName<<endl;
			}else
			{
				if(k==0)
				{
					sign = left_v.read(img);
				}else
				{
					sign = right_v.read(img);
				}	
				
			}
			if(!sign)
			{
				cout<<"end"<<endl;
				break;
			}
            if(img.empty())
                break;
			// Make sure image is grayscale
			if(img.channels()>1)
			{
				cvtColor(img, img, CV_RGB2GRAY);
			}
            if( imageSize == Size() )
                imageSize = img.size();
            else if( img.size() != imageSize )
            {
                cout << "The image " << ImgName << " has the size different from the first image size. Skipping the pair\n";
                break;
            }
			remap(img, rimg, rmap[k][0], rmap[k][1], CV_INTER_LINEAR);
	
			if(FLAG_SHOW_RECTIRY)
			{
				cvtColor(rimg, cimg, CV_GRAY2BGR);
				Mat canvasPart = !isVerticalStereo ? canvas(Rect(w*k, 0, w, h)) : canvas(Rect(0, h*k, w, h));
				resize(cimg, canvasPart, canvasPart.size(), 0, 0, CV_INTER_AREA);
				if( 1 )
				{
					Rect vroi(cvRound(validRoi[k].x*sf), cvRound(validRoi[k].y*sf),
							  cvRound(validRoi[k].width*sf), cvRound(validRoi[k].height*sf));
					rectangle(canvasPart, vroi, Scalar(0,0,255), 3, 8);
				}
			}
			if(k==1)
			{
				left_im = rimg;
//				imwrite("left_crop.png",left_im);
			}else
			{
				right_im = rimg;
//				imwrite("right_crop.png",right_im);
			}
        }
		sgbm(right_im,left_im,disp_sgbm,disp_cost,disp_mincost); // Computing disparity map using SGBM
		
		normalize(disp_sgbm, vdisp, 0, 256, NORM_MINMAX,CV_8UC1 );
		if(FLAG_SHOW_DISPARITY)
		{
			imshow("disparity", vdisp);
		}
		if(FLAG_SAVE_DISPARITY)
		{
			string disparity_name;
			disparity_name = SavePath + "disparity_" + ImageID + ".txt";
			ofstream outFilemindis(disparity_name, ios_base::out);
			for (int r=0; r<disp_sgbm.rows; r++)
			{  
				for (int c = 0; c < disp_sgbm.cols; c++) 
				{
						short int data = disp_sgbm.at<short>(r,c); 
						outFilemindis << data << "\t" ;    
				}
				outFilemindis << endl;    
			} 
			ImgName = SavePath + "disparity_" + ImageID + ".png";
			imwrite(ImgName,vdisp);
			cout<<"Disparity map is saved in "<<ImgName<<" and "<<disparity_name<<endl;
		}

		// Computing confidence cue
		if(FLAG_CONFIDENCE)
		{
			string cost1_name, cost2_name, costmin_name, costMLM_name, costLC_name;
			
			cost1_name = SavePath + "cost1.txt";
			cost2_name = SavePath + "cost2.txt";
			costMLM_name = SavePath + "costMLM.txt";
			costLC_name = SavePath + "costLC.txt";

			ofstream outFilemin1(cost1_name, ios_base::out);
			ofstream outFilemin2(cost2_name, ios_base::out);
			ofstream outFileMLM(costMLM_name, ios_base::out);
			ofstream outFileLC(costLC_name, ios_base::out);
			
			int max_cost = 10000000;
			int minD = sgbm.minDisparity, maxD = minD + sgbm.numberOfDisparities;
			int D = maxD-minD;
			int min_cost,sed_min_cost,min_left,min_right;
			int data;
			double sum_exp,min_exp,LC_val;
			double theta = 10, gama = 480;
			int row_num = disp_cost.rows/disp_sgbm.rows,cnt_row=0;

			for(int r = 0; r<disp_cost.rows; r++)
			{
				short* costptr = disp_cost.ptr<short>(r);
				min_cost = max_cost;
				sed_min_cost = max_cost;
        
				sum_exp = 0;
				for(int d = 0; d<D; d++)
				{
					data = costptr[d];
					sum_exp = sum_exp + exp(-(double)data/(2*theta*theta));
					if((data<min_cost))
					{
						sed_min_cost = min_cost;
						min_cost = data;
						min_left = data;
						min_right = data;
						if(d>0)
						{
							min_left = costptr[d-1];
						}
						if(d<(D-1))
						{
							min_right = costptr[d+1];
						}
					 }
					 else if((data<sed_min_cost))
					 {
						sed_min_cost = data;
					 }
				 }
				 if(min_left>min_right)
				 {
					LC_val = ((double)min_left-(double)min_cost)/gama;
				 }else
				 {
					LC_val = ((double)min_right-(double)min_cost)/gama;
				 }
                            
				 min_exp = exp(-(double)min_cost/(2*theta*theta));
				 min_exp = min_exp/sum_exp;
                            
				 outFileLC << LC_val << "\t" ;
				 outFilemin1 << min_cost << "\t" ;
				 outFilemin2 << sed_min_cost << "\t" ;
				 outFileMLM << min_exp << "\t" ;
                            
				 if(cnt_row == (row_num-1) )
				 {
					outFileMLM << endl;
					outFilemin2 << endl;
					outFilemin1 << endl;
					outFileLC << endl;
					cnt_row = 0;
				 }else
				 {
					cnt_row++;
				 }
			}
			cout<<"Confidence cues are saved in "<<SavePath<<endl;
		}
		if(FLAG_SHOW_RECTIRY)
		{
			if( !isVerticalStereo )
				for( j = 0; j < canvas.rows; j += 16 )
					line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
			else
				for( j = 0; j < canvas.cols; j += 16 )
					line(canvas, Point(j, 0), Point(j, canvas.rows), Scalar(0, 255, 0), 1, 8);
			imshow("rectified", canvas);
			c = (char)waitKey();

		}
		if( c == 27 || c == 'q' || c == 'Q' )
			break;
		i++;
		if((ImageOrVideo)&&(i >= NumImg))
			break;
    }
}

int main(int argc, char** argv)
{
    for( int i = 1; i < argc; i++ )
    {
        if( string(argv[i]) == "--help" )
            return print_help();
        else if( argv[i][0] == '-' )
        {
            cout << "invalid option " << argv[i] << endl;
            return 0;
        }
    }
    StereoDisparity();
    return 0;
}