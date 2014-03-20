#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#ifdef HAVE_CVCONFIG_H
#include "cvconfig.h"
#endif

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/internal.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <vector>

#ifdef HAVE_TEGRA_OPTIMIZATION
#include "opencv2/calib3d/calib3d_tegra.hpp"
#else
#define GET_OPTIMIZED(func) (func)
#endif

#endif


using namespace cv;

class CV_EXPORTS_W StereoSGBM_COST
{
public:
    enum { DISP_SHIFT=4, DISP_SCALE = (1<<DISP_SHIFT) };

    //! the default constructor
    CV_WRAP StereoSGBM_COST();

    //! the full constructor taking all the necessary algorithm parameters
    CV_WRAP StereoSGBM_COST(int minDisparity, int numDisparities, int SADWindowSize,
               int P1=0, int P2=0, int disp12MaxDiff=0,
               int preFilterCap=0, int uniquenessRatio=0,
               int speckleWindowSize=0, int speckleRange=0,
               bool fullDP=false);
    //! the destructor
    virtual ~StereoSGBM_COST();

    //! the stereo correspondence operator that computes disparity map for the specified rectified stereo pair
    CV_WRAP_AS(compute) virtual void operator()(InputArray left, InputArray right,
                                                OutputArray _disp, OutputArray _disp2, OutputArray _dispMinCost);

    CV_PROP_RW int minDisparity;
    CV_PROP_RW int numberOfDisparities;
    CV_PROP_RW int SADWindowSize;
    CV_PROP_RW int preFilterCap;
    CV_PROP_RW int uniquenessRatio;
    CV_PROP_RW int P1;
    CV_PROP_RW int P2;
    CV_PROP_RW int speckleWindowSize;
    CV_PROP_RW int speckleRange;
    CV_PROP_RW int disp12MaxDiff;
    CV_PROP_RW bool fullDP;

protected:
    Mat buffer;
};