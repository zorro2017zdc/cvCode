#include "mex.h"
#include <time.h>
#include <math.h>
#include <string.h>

#ifndef DBL_MAX
    #define DBL_MAX  9999999999
#endif
#ifndef DBL_MIN
    #define DBL_MIN -9999999999
#endif

/*Round function.*/
double round(double x) { return (x-floor(x))>0.5 ? ceil(x) : floor(x); }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Input/output variables. */
    double ch;      /* Canvas height. */
    double cw;      /* Canvas width. */
    double *img2;   /* Image to be stitched (source image). */
    double *H;      /* Image to be stitched (source image). */
    double *off;    /* 2x1 matrix which contains the offset of the source and target images. */
    double *orig_arr;
    
    double *X;      /* X positions of the moving dlt (or moving ls) grid. */
    double *Y;      /* Y positions of the moving dlt (or moving ls) grid. */    
          
    double *Hinv, det;
    
    double *warped_img2;     /* Warped img2 (img2 after being warped with either Global Homography (projective, DLT) 
                              * or Moving DLT. */
    double *warped_arr;
    double *encl_rect;      /* Coordinates of the top-left and bottom-right corners 
                            * the minimum enclosing rectangle of the warped image. */  
    
    /* Intermediate variables.*/
    int canvm,canvn;    /* 'm' and 'n': size of the canvas image. */
    int img2m,img2n;    /* 'm' and 'n': size of the target image. */    
    int Hm;
    int xn;
    int yn;    
    int arrm, arridx;
    
    double minx = DBL_MAX, maxx = DBL_MIN;
    double miny = DBL_MAX, maxy = DBL_MIN;
    double dist;
    
    int posa, posb; /*contain the position of the source image after applying the homography H (with DLT or MDLT)*/
    int cidx, sidx; /*contain the indexes of the target and source images respectively (after homography transform)*/
    
    /* Since MATLAB works with 2D matrices and C works with 1D arrays we need the following variables for generating the 
     * final stitched RGB 2D matrix for MATLAB, this image/matrix contains the warped img2. */
    int ch1canv; /* Displacement for channel 1 in image 1 (R). */
    int ch2canv; /* Displacement for channel 2 in image 1 (G). */
    int ch3canv; /* Displacement for channel 3 in image 1 (B). */
    
    int ch1img2; /* Displacement for channel 1 in image 2 (R). */
    int ch2img2; /* Displacement for channel 2 in image 2 (G). */
    int ch3img2; /* Displacement for channel 3 in image 2 (B). */
    
    /* Auxiliary variables and counters*/
    int xinx,yinx,inx;
    int i, j;
    
    /* Check for proper number of arguments. */    
    if (nrhs < 5 && nrhs > 8)
    {
        mexErrMsgTxt("Wrong number of input arguments.");
    }
    else if (nlhs > 2)
    {
        mexErrMsgTxt("Wrong number of output arguments.");
    }
  
    /* Assign scalars to inputs. */
    ch  = mxGetScalar(prhs[0]);
    cw  = mxGetScalar(prhs[1]);
    
    /* Assign pointers to inputs. */    
    img2  = mxGetPr(prhs[2]);
    H     = mxGetPr(prhs[3]);
    off   = mxGetPr(prhs[4]);

    /* Get sizes of input matrices (images, transformations, etc.).*/
    canvm = (int)ch;
    canvn = (int)cw;

    img2m = mxGetM(prhs[2]);
    img2n = mxGetN(prhs[2]) / 3; /* It is an RGB image, which means that for MATLAB it is an Mx(Nx3) image. So, we have  
                                  * to divide N by 3 in order to get the proper size of the image in C. */
    Hm = mxGetM(prhs[3]); /* Number of H matrixes in H. */
    
    if (nrhs == 7 || nrhs == 8){ /* Get input parameters for MDLT warping. */
        X     = mxGetPr(prhs[5]);
        Y     = mxGetPr(prhs[6]);
                
        xn    = mxGetN(prhs[5]);    /*length of X*/
        yn    = mxGetN(prhs[6]);    /*length of Y*/
    }    
    
    /* Create matrix for the return arguments. */
    plhs[0] = mxCreateDoubleMatrix(canvm,canvn*3,mxREAL);  
    plhs[1] = mxCreateDoubleMatrix(1,4,mxREAL);     
    
    /* Assign pointers to output canvas (warped image2). */
    warped_img2  = mxGetPr(plhs[0]);       
    encl_rect = mxGetPr(plhs[1]);
    
    /* Initialize displacements. */
    ch1canv = 0;
    ch2canv = canvn*canvm;
    ch3canv = canvn*canvm*2;
    
    ch1img2 = 0;
    ch2img2 = img2n*img2m;
    ch3img2 = img2n*img2m*2;
    
    /* Start computations. */
    if (nrhs == 5 || nrhs == 6){ /*We stitch Global Homography (DLT). */
        
        /* For each pixel in the target image... */
        for(i=0;i<canvm;i++)
        {
            for(j=0;j<canvn;j++)
            {
                /* Get projective transformation for current source point (i,j). */
                posa = round(((H[(Hm*0)+0] * (j-off[0]+1)) + (H[(Hm*1)+0] * (i-off[1]+1)) + H[(Hm*2)+0]) / ((H[(Hm*0)+2] * (j-off[0]+1)) + (H[(Hm*1)+2] * (i-off[1]+1)) + H[(Hm*2)+2]));
                posb = round(((H[(Hm*0)+1] * (j-off[0]+1)) + (H[(Hm*1)+1] * (i-off[1]+1)) + H[(Hm*2)+1]) / ((H[(Hm*0)+2] * (j-off[0]+1)) + (H[(Hm*1)+2] * (i-off[1]+1)) + H[(Hm*2)+2]));

                /* Find if the current pixel/point in the (warped) source image falls inside canvas. */
                if ((posa > 1)&&(posa < img2n)&&(posb>1)&&(posb<img2m))
                {
                    /* If the current pixel in the source image hits the target (i.e., is inside the canvas) 
                     * we get its corresponding position in the 2D array. */
                    cidx = ((j-1)*canvm)+(i-1);
                    sidx = ((posa-1)*img2m)+(posb-1);
                    
                    if (j < minx)
                        minx = j;
                    if (j > maxx)
                        maxx = j;
                    if (i < miny)
                        miny = i;
                    if (i > maxy)
                        maxy = i;
                    
                    /* Warping pixel in source image to canvas. */
                    if (cidx+ch1canv >= 0 && cidx+ch3canv < canvm*canvn*3 && sidx+ch1img2 >= 0 && sidx+ch3img2 < img2m*img2n*3)
                    {                    
                        warped_img2[cidx+ch1canv] = img2[sidx+ch1img2];
                        warped_img2[cidx+ch2canv] = img2[sidx+ch2img2];
                        warped_img2[cidx+ch3canv] = img2[sidx+ch3img2];
                    }
                }
            }
        }
        if(nrhs == 6)
        {            
            /* Assign array to input.*/
            orig_arr = mxGetPr(prhs[5]);
            arrm     = mxGetM(prhs[5]);
                        
            /* Create matrix for the return arguments. */
            plhs[1] = mxCreateDoubleMatrix(arrm,2,mxREAL);
            
            /* Assign pointers to output image. */
            warped_arr   = mxGetPr(plhs[1]);             
            
            Hinv = (double*)malloc(9);            
            det=H[Hm*0]*(H[Hm*4]*H[Hm*8]-H[Hm*5]*H[Hm*7])-H[Hm*3]*(H[Hm*1]*H[Hm*8]-H[Hm*7]*H[Hm*2])+H[Hm*6]*(H[Hm*1]*H[Hm*5]-H[Hm*4]*H[Hm*2]);
            Hinv[0]= (H[Hm*4]*H[Hm*8]-H[Hm*5]*H[Hm*7])/det;
            Hinv[1]=-(H[Hm*1]*H[Hm*8]-H[Hm*7]*H[Hm*2])/det;
            Hinv[2]= (H[Hm*1]*H[Hm*5]-H[Hm*2]*H[Hm*4])/det;
            Hinv[3]=-(H[Hm*3]*H[Hm*8]-H[Hm*6]*H[Hm*5])/det;
            Hinv[4]= (H[Hm*0]*H[Hm*8]-H[Hm*6]*H[Hm*2])/det;
            Hinv[5]=-(H[Hm*0]*H[Hm*5]-H[Hm*2]*H[Hm*3])/det;
            Hinv[6]= (H[Hm*3]*H[Hm*7]-H[Hm*6]*H[Hm*4])/det;
            Hinv[7]=-(H[Hm*0]*H[Hm*7]-H[Hm*1]*H[Hm*6])/det;
            Hinv[8]= (H[Hm*0]*H[Hm*4]-H[Hm*1]*H[Hm*3])/det;
            
            for(arridx=0;arridx<arrm;arridx++)
            {
                /* Warp each element in array. */
                j = orig_arr[arridx];
                i = orig_arr[arridx+arrm];
                    
                /* Get projective transformation of this grid point and warp the current pixel with it. */
                posa = round(((Hinv[0] * (j-off[0]+1)) + (Hinv[3] * (i-off[1]+1)) + Hinv[6]) / ((Hinv[2] * (j-off[0]+1)) + (Hinv[5] * (i-off[1]+1)) + Hinv[8]));
                posb = round(((Hinv[1] * (j-off[0]+1)) + (Hinv[4] * (i-off[1]+1)) + Hinv[7]) / ((Hinv[2] * (j-off[0]+1)) + (Hinv[5] * (i-off[1]+1)) + Hinv[8]));
                
                warped_arr[arridx]      = posa;
                warped_arr[arridx+arrm] = posb;
            }
        
        }
    }
    else if(nrhs == 7 || nrhs == 8)
    {
        /* For each point in the grid. */        
        for(i=0;i<canvm;i++)
        {
            for(j=0;j<canvn;j++)
            {            
                /* Get grid point for current pixel. */
                for(xinx=0; xinx < xn && j >= X[xinx]; xinx++);
                for(yinx=0; yinx < yn && i >= Y[yinx]; yinx++);
                
                if(j-X[xinx]>0.5)
                    xinx = xinx+1;

                if(i-Y[yinx]>0.5)
                    yinx = yinx+1;

                inx = yinx + xinx*xn;

                /* Get projective transformation of this grid point and warp the current pixel with it. */
                posa = round(((H[inx+(Hm*0)] * (j-off[0]+1)) + (H[inx+(Hm*3)] * (i-off[1]+1)) + H[inx+(Hm*6)]) / ((H[inx+(Hm*2)] * (j-off[0]+1)) + (H[inx+(Hm*5)] * (i-off[1]+1)) + H[inx+(Hm*8)]));
                posb = round(((H[inx+(Hm*1)] * (j-off[0]+1)) + (H[inx+(Hm*4)] * (i-off[1]+1)) + H[inx+(Hm*7)]) / ((H[inx+(Hm*2)] * (j-off[0]+1)) + (H[inx+(Hm*5)] * (i-off[1]+1)) + H[inx+(Hm*8)]));
                              
                /* Find if the current pixel/point in the (warped) source image falls inside canvas. */
                if ((posa>0)&&(posa<img2n)&&(posb>0)&&(posb<img2m))
                {
                    /* If the current pixel in the source image hits the target (i.e., if it is inside the canvas) 
                     * we get its corresponding position in the 2D array. */
                    cidx = ((j-1)*canvm)+(i-1);
                    sidx = ((posa-1)*img2m)+(posb-1);
                          
                    if (j < minx)
                        minx = j;
                    if (j > maxx)
                        maxx = j;
                    if (i < miny)
                        miny = i;
                    if (i > maxy)
                        maxy = i;
                    
                    /* Warping pixel in source image to canvas. */
                    if ((cidx+ch1canv > 0) && (cidx+ch3canv < canvm*canvn*3) && (sidx+ch1img2 > 0) && (sidx+ch3img2 < img2m*img2n*3))
                    {
                        warped_img2[cidx+ch1canv] = img2[sidx+ch1img2];
                        warped_img2[cidx+ch2canv] = img2[sidx+ch2img2];
                        warped_img2[cidx+ch3canv] = img2[sidx+ch3img2];
                    }
                }
            }
        }
        if (nrhs == 8)
        {
            Hinv = (double*)malloc(9);
            
            /* Assign array to input.*/
            orig_arr = mxGetPr(prhs[7]);
            arrm     = mxGetM(prhs[7]);
                        
            /* Create matrix for the return arguments. */
            plhs[1] = mxCreateDoubleMatrix(arrm,2,mxREAL);
            
            /* Assign pointers to output image. */
            warped_arr   = mxGetPr(plhs[1]);             
/*            for(arridx=0;arridx<arrm;arridx++)
            {
                // Warp each element in array. 
                j = orig_arr[arridx];
                i = orig_arr[arridx+arrm];
                
                // Get grid point for current pixel. 
                for(xinx=0; xinx < xn && j >= X[xinx]; xinx++);
                for(yinx=0; yinx < yn && i >= Y[yinx]; yinx++);
                
                if(j-X[xinx]>0.5)
                    xinx = xinx+1;

                if(i-Y[yinx]>0.5)
                    yinx = yinx+1;

                inx = yinx + xinx*xn;

                det=H[inx+(Hm*0)]*(H[inx+(Hm*4)]*H[inx+(Hm*8)]-H[inx+(Hm*5)]*H[inx+(Hm*7)])-H[inx+(Hm*3)]*(H[inx+(Hm*1)]*H[inx+(Hm*8)]-H[inx+(Hm*7)]*H[inx+(Hm*2)])+H[inx+(Hm*6)]*(H[inx+(Hm*1)]*H[inx+(Hm*5)]-H[inx+(Hm*4)]*H[inx+(Hm*2)]);
                Hinv[0]= (H[inx+(Hm*4)]*H[inx+(Hm*8)]-H[inx+(Hm*5)]*H[inx+(Hm*7)])/det;
                Hinv[1]=-(H[inx+(Hm*1)]*H[inx+(Hm*8)]-H[inx+(Hm*7)]*H[inx+(Hm*2)])/det;
                Hinv[2]= (H[inx+(Hm*1)]*H[inx+(Hm*5)]-H[inx+(Hm*2)]*H[inx+(Hm*4)])/det;
                Hinv[3]=-(H[inx+(Hm*3)]*H[inx+(Hm*8)]-H[inx+(Hm*6)]*H[inx+(Hm*5)])/det;
                Hinv[4]= (H[inx+(Hm*0)]*H[inx+(Hm*8)]-H[inx+(Hm*6)]*H[inx+(Hm*2)])/det;
                Hinv[5]=-(H[inx+(Hm*0)]*H[inx+(Hm*5)]-H[inx+(Hm*2)]*H[inx+(Hm*3)])/det;
                Hinv[6]= (H[inx+(Hm*3)]*H[inx+(Hm*7)]-H[inx+(Hm*6)]*H[inx+(Hm*4)])/det;
                Hinv[7]=-(H[inx+(Hm*0)]*H[inx+(Hm*7)]-H[inx+(Hm*1)]*H[inx+(Hm*6)])/det;
                Hinv[8]= (H[inx+(Hm*0)]*H[inx+(Hm*4)]-H[inx+(Hm*1)]*H[inx+(Hm*3)])/det;
    
                // Get projective transformation of this grid point and warp the current pixel with it. 
                posa = round(((Hinv[0] * (j-off[0]+1)) + (Hinv[3] * (i-off[1]+1)) + Hinv[6]) / ((Hinv[2] * (j-off[0]+1)) + (Hinv[5] * (i-off[1]+1)) + Hinv[8]));
                posb = round(((Hinv[1] * (j-off[0]+1)) + (Hinv[4] * (i-off[1]+1)) + Hinv[7]) / ((Hinv[2] * (j-off[0]+1)) + (Hinv[5] * (i-off[1]+1)) + Hinv[8]));
                
                warped_arr[arridx]      = posa;
                warped_arr[arridx+arrm] = posb;
            }*/
            
            for(arridx=0;arridx<arrm;arridx++)
            {  
                dist = DBL_MAX;
                for(i=0;i<canvm;i++)
                {
                    for(j=0;j<canvn;j++)
                    {            
                        /* Get grid point for current pixel. */
                        for(xinx=0; xinx < xn && j >= X[xinx]; xinx++);
                        for(yinx=0; yinx < yn && i >= Y[yinx]; yinx++);

                        if(j-X[xinx]>0.5)
                            xinx = xinx+1;

                        if(i-Y[yinx]>0.5)
                            yinx = yinx+1;

                        inx = yinx + xinx*xn;

                        /* Get projective transformation of this grid point and warp the current pixel with it. */
                        posa = round(((H[inx+(Hm*0)] * (j-off[0]+1)) + (H[inx+(Hm*3)] * (i-off[1]+1)) + H[inx+(Hm*6)]) / ((H[inx+(Hm*2)] * (j-off[0]+1)) + (H[inx+(Hm*5)] * (i-off[1]+1)) + H[inx+(Hm*8)]));
                        posb = round(((H[inx+(Hm*1)] * (j-off[0]+1)) + (H[inx+(Hm*4)] * (i-off[1]+1)) + H[inx+(Hm*7)]) / ((H[inx+(Hm*2)] * (j-off[0]+1)) + (H[inx+(Hm*5)] * (i-off[1]+1)) + H[inx+(Hm*8)]));

                        /* Find if the current pixel/point in the (warped) source image falls inside canvas. */
                        if (posa==orig_arr[arridx] && posb==orig_arr[arridx+arrm])
                        {
                            warped_arr[arridx]      = j;
                            warped_arr[arridx+arrm] = i;
                            i = canvm;
                            j = canvn;
                        }
                        else if(dist > sqrt(pow(posa-orig_arr[arridx],2)+pow(posb-orig_arr[arridx+arrm],2)))
                        {
                            dist = sqrt(pow(posa-orig_arr[arridx],2)+pow(posb-orig_arr[arridx+arrm],2));
                            warped_arr[arridx]      = j;
                            warped_arr[arridx+arrm] = i;                        
                        }
                    }            
                }
            }
        }
    }
    
    encl_rect[0] = minx;
    encl_rect[1] = miny;
    encl_rect[2] = maxx-minx;
    encl_rect[3] = maxy-miny;    
    /* Note: I decided to perform the previous IF/ELSE statement for performing either Homography
     * or Moving DLT stitching and (as you can see) I "repeated" most of  the code inside the IF 
     * and the ELSE statements. I did this in order to avoid performing an IF/ELSE inside the 2
     * nested for loops and, therefore, perform such IF/ELSE comparisons an MxN number of times. 
     * This way I just have one IF/ELSE statement at the really small price of repeating a small
     * chunk of code instead of performing a huge number of IF/ELSE comparisons inside the loops :) */
    
    /* Bye bye.*/
    return;
}
