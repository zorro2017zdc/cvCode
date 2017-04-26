#include "mex.h"
#include <math.h>
#include <string.h>

#define pi 3.14159265
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
    double *orig_img;   /* Image to be projected into the sphere. */
    double *kps;   /* Image to be projected into the sphere. */
    double f;      /* Y positions of the moving dlt (or moving ls) grid. */        
    double *orig_arr;
      
    double *proj_img;     /* Warped image (image after being warped to a sphere). */
    double *encl_rect;         /* Coordinates of the top-left and bottom-right corners 
                                * the minimum enclosing rectangle of the warped image. */  
    double *proj_arr;        
    
    /* Intermediate variables.*/
    int imgm,imgn;    /* 'm' and 'n': size of the target image. */    
    int arrm;
    
    /* Since MATLAB works with 2D matrices and C works with 1D arrays we need the following variables for generating the 
     * final stitched RGB 2D matrix for MATLAB, this image/matrix contains the warped img2. */
    int ch1img; /* Displacement for channel 1 in image 2 (R). */
    int ch2img; /* Displacement for channel 2 in image 2 (G). */
    int ch3img; /* Displacement for channel 3 in image 2 (B). */   
    double dist;
    
    int cidx, sidx; /* Contain the indexes of the image after projection. */
    
    double minx = DBL_MAX, maxx = DBL_MIN;
    double miny = DBL_MAX, maxy = DBL_MIN;
    double dif;
    int status;
    int arridx;
    
    /* Auxiliary variables and counters*/
    int yproj, xproj;
    double theta, h, phi, xh, yh, zh, xc, yc, x, y;
    
    /* Check for proper number of arguments. */    
    if (nrhs != 3 && nrhs != 4)
    {
        mexErrMsgTxt("Three or four inputs required.");
    }
    else if (nlhs > 3)
    {
        mexErrMsgTxt("Wrong number of output arguments.");
    }
    
    /* Assign pointers to inputs.*/    
    orig_img  = mxGetPr(prhs[0]);
    
    /* Assign scalars to inputs.*/
    f  = mxGetScalar(prhs[1]);
    
    /* Get projection type string. */    
    /* Get the length of the projection string. */
    char *projection_type = (char*)mxCalloc((mxGetM(prhs[2]) * mxGetN(prhs[2])) + 1, sizeof(char));
    status = mxGetString(prhs[2], projection_type, (mxGetM(prhs[2]) * mxGetN(prhs[2])) + 1);
    
    /* Get sizes of input matrices.*/
    imgm = mxGetM(prhs[0]);
    imgn = mxGetN(prhs[0]) / 3; /* It is an RGB image, which means that for MATLAB it is an Mx(Nx3) image. So, we have  
                                 * to divide N by 3 in order to get the proper size of the image in C. */

    /* Initialize original image displacements. */
    ch1img = 0;
    ch2img = imgn*imgm;
    ch3img = imgn*imgm*2;
    
    /* Get image center. */
    xc = (double)imgn/2;
    yc = (double)imgm/2;    
    
    /* Create matrix for the return arguments. */
    plhs[0] = mxCreateDoubleMatrix(imgm,imgn*3,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,4,mxREAL);     
    
    /* Assign pointers to output image. */
    proj_img   = mxGetPr(plhs[0]);
    encl_rect = mxGetPr(plhs[1]);
       
    /* Start computations. */         
    /* Projecting image to cylinder. */
    if(strcmp(projection_type,"cylinder") == 0)
    {       
        for(yproj=0;yproj<imgm;yproj++) /* For each pixel in the image... */
        {
            for(xproj=0;xproj<imgn;xproj++)
            {
                /* Project image to cylinder (following  Szeliski's book). */
                theta = ((double)xproj - xc) / f;
                h  = ((double)yproj - yc) / f;
                xh = sin(theta);
                yh = h;
                zh = cos(theta);

                x = round(f*xh/zh + xc);
                y = round(f*yh/zh + yc);

                /* Find if the current pixel/point in the source image 'hits' the target. */
                if ((x>0)&&(y>0)&&(x<imgn)&&(y<imgm))
                {
                    /* If the current pixel in the source image hits the target we get its corresponding position in the 2D array. */                
                    cidx = ((xproj-1)*imgm)+(yproj-1);
                    sidx = ((x-1)*imgm)+(y-1);
                    if (xproj < minx)
                        minx = xproj;
                    if (xproj > maxx)
                        maxx = xproj;
                    if (yproj < miny)
                        miny = yproj;
                    if (yproj > maxy)
                        maxy = yproj;                

                    /* Projecting image. */
                    if ((cidx+ch1img > 0) && (cidx+ch1img < imgn*imgm*3) && (sidx+ch1img > 0) && (sidx+ch1img < imgm*imgn*3))
                    {                
                        proj_img[cidx+ch1img] = orig_img[sidx+ch1img];
                        proj_img[cidx+ch2img] = orig_img[sidx+ch2img];
                        proj_img[cidx+ch3img] = orig_img[sidx+ch3img];
                    }
                }
            }   
        }
        if(nrhs==4)
        {             
            // Assign array to input.
            orig_arr = mxGetPr(prhs[3]);
            arrm = mxGetM(prhs[3]);
            
            // Create matrix for the return arguments. 
            plhs[2] = mxCreateDoubleMatrix(arrm,2,mxREAL);
            
            // Assign pointers to output image. 
            proj_arr   = mxGetPr(plhs[2]);
            
            for(arridx=0;arridx<arrm;arridx++)
            {
                dist = 9999999999;
                
                // Project each element in array to cylinder (following  Szeliski's book).
                for(yproj=0;yproj<imgm;yproj++) // For each pixel in the image... 
                {
                    for(xproj=0;xproj<imgn;xproj++)
                    {
                        // Project image to sphere (following  Szeliski's book). 
                        theta = ((double)xproj - xc) / f;
                        h  = ((double)yproj - yc) / f;
                        xh = sin(theta);
                        yh = h;
                        zh = cos(theta);

                        x = round(f*xh/zh + xc);
                        y = round(f*yh/zh + yc);

                        if ((x>0)&&(y>0)&&(x<imgn)&&(y<imgm))
                        {
                            // If the current pixel in the source image hits the target we get its corresponding position in the 2D array. 
                            cidx = ((xproj-1)*imgm)+(yproj-1);
                            sidx = ((x-1)*imgm)+(y-1);

                            // Projecting image.
                            if ((cidx+ch1img > 0) && (cidx+ch1img < imgn*imgm*3) && (sidx+ch1img > 0) && (sidx+ch1img < imgm*imgn*3))
                            {                                
                                proj_img[cidx+ch1img] = orig_img[sidx+ch1img];
                                proj_img[cidx+ch2img] = orig_img[sidx+ch2img];
                                proj_img[cidx+ch3img] = orig_img[sidx+ch3img];
                                
                                if (sqrt(pow(x-orig_arr[arridx],2)+pow(y - orig_arr[arridx+arrm],2)) < dist)
                                {
                                    dist = round(sqrt(pow(x-orig_arr[arridx],2)+pow(y - orig_arr[arridx+arrm],2)));
                                    proj_arr[arridx]      = xproj-1;
                                    proj_arr[arridx+arrm] = yproj-1;                                     
                                }
                            }
                        }
                    }
                }                               
            }                        
        }
    }    
    else if(strcmp(projection_type,"sphere") == 0)
    {            
        /* Projecting image to sphere. */
        for(yproj=0;yproj<imgm;yproj++) /* For each pixel in the image... */
        {
            for(xproj=0;xproj<imgn;xproj++)
            {
                /* Project image to sphere (following  Szeliski's book). */
                theta = ((double)xproj - xc) / f;
                phi  = ((double)yproj - yc) / f;
                xh = sin(theta)*cos(phi);
                yh = sin(phi);
                zh = cos(theta)*cos(phi);

                x = round(f*xh/zh + xc);
                y = round(f*yh/zh + yc);

                if ((x>0)&&(y>0)&&(x<imgn)&&(y<imgm))
                {
                    /* If the current pixel in the source image hits the target we get its corresponding position in the 2D array. */
                    cidx = ((xproj-1)*imgm)+(yproj-1);
                    sidx = ((x-1)*imgm)+(y-1);
                    if (xproj < minx)
                        minx = xproj;
                    if (xproj > maxx)
                        maxx = xproj;
                    if (yproj < miny)
                        miny = yproj;
                    if (yproj > maxy)
                        maxy = yproj;

                    /* Projecting image. */
                    if ((cidx+ch1img > 0) && (cidx+ch1img < imgn*imgm*3) && (sidx+ch1img > 0) && (sidx+ch1img < imgm*imgn*3))
                    {                                
                        proj_img[cidx+ch1img] = orig_img[sidx+ch1img];
                        proj_img[cidx+ch2img] = orig_img[sidx+ch2img];
                        proj_img[cidx+ch3img] = orig_img[sidx+ch3img];
                    }
                }
            }
        }
        if(nrhs==4)
        {  
            
            // Assign array to input.
            orig_arr = mxGetPr(prhs[3]);
            arrm = mxGetM(prhs[3]);
            
            // Create matrix for the return arguments. 
            plhs[2] = mxCreateDoubleMatrix(arrm,2,mxREAL);
            
            // Assign pointers to output image. 
            proj_arr   = mxGetPr(plhs[2]);
            
            for(arridx=0;arridx<arrm;arridx++)
            {
                dist = 9999999999;
                
                // Project each element in array to cylinder (following  Szeliski's book).
                for(yproj=0;yproj<imgm;yproj++) // For each pixel in the image... 
                {
                    for(xproj=0;xproj<imgn;xproj++)
                    {
                        // Project image to sphere (following  Szeliski's book). 
                        theta = ((double)xproj - xc) / f;
                        phi  = ((double)yproj - yc) / f;
                        xh = sin(theta)*cos(phi);
                        yh = sin(phi);
                        zh = cos(theta)*cos(phi);

                        x = round(f*xh/zh + xc);
                        y = round(f*yh/zh + yc);

                        if ((x>0)&&(y>0)&&(x<imgn)&&(y<imgm))
                        {
                            // If the current pixel in the source image hits the target we get its corresponding position in the 2D array. 
                            cidx = ((xproj-1)*imgm)+(yproj-1);
                            sidx = ((x-1)*imgm)+(y-1);

                            // Projecting image.
                            if ((cidx+ch1img > 0) && (cidx+ch1img < imgn*imgm*3) && (sidx+ch1img > 0) && (sidx+ch1img < imgm*imgn*3))
                            {                                
                                proj_img[cidx+ch1img] = orig_img[sidx+ch1img];
                                proj_img[cidx+ch2img] = orig_img[sidx+ch2img];
                                proj_img[cidx+ch3img] = orig_img[sidx+ch3img];
                                
                                if (sqrt(pow(x-orig_arr[arridx],2)+pow(y - orig_arr[arridx+arrm],2)) < dist)
                                {
                                    dist = round(sqrt(pow(x-orig_arr[arridx],2)+pow(y - orig_arr[arridx+arrm],2)));
                                    proj_arr[arridx]      = xproj-1;
                                    proj_arr[arridx+arrm] = yproj-1;                                     
                                }
                            }
                        }
                    }
                }                               
            }                        
        }    
    }
    else
    {
        mexErrMsgTxt("Projection types can be 'cylinder' or 'sphere' only.\n");
    }
    encl_rect[0] = minx;
    encl_rect[1] = miny;
    encl_rect[2] = maxx-minx;
    encl_rect[3] = maxy-miny;

    /* Bye bye.*/
    return;
}