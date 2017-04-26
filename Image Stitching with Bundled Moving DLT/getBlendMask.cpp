#include "mex.h"
#include "math.h"
#include "time.h"

/*Round function.*/
double round(double x) { return (x-floor(x))>0.5 ? ceil(x) : floor(x); }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Input variables. */
    double *img1;
    double *img2;
    double *H;
    double *off;    /* 2x1 matrix which contains the offset of the source and target images. */
    double *encl_rect;    /* 2x1 matrix which contains the offset of the source and target images. */

    double *X;      /* X positions of the moving dlt (or moving ls) grid. */
    double *Y;      /* Y positions of the moving dlt (or moving ls) grid. */    
    
    /* Output variables. */
    double *mask;
    double *inten1;
    double *inten2;
  
    /* Intermediate variables.*/
    int Hm;
    int xn;
    int yn;
    double img1m, img1n, img2m,img2n;
    
    int d1,d2,m1,n1,m2,n2;
    int i,j,k;
    double a,b,c;
    int p,q;
    int count;
    long int auxidx1, auxidx2;
    int xinx,yinx,inx;
    mwSize *dimsize;
    
    /* Check for proper number of arguments. */    
    /*if (nrhs!=4 && nrhs!=6)
    {
        mexErrMsgTxt("Four of six inputs required!");
    }
    else if (nlhs > 3)
    {
        mexErrMsgTxt("Too many output arguments!");
    }*/
  
    /* Assign pointers to inputs.*/
    img1 = mxGetPr(prhs[0]);
    img2 = mxGetPr(prhs[1]);
    H = mxGetPr(prhs[2]);
    
    /* Get sizes of input matrices.*/
    d2 = mxGetNumberOfDimensions(prhs[1]);
    dimsize = (mwSize*)mxGetDimensions(prhs[1]);
    m2 = dimsize[0];
    n2 = dimsize[1];
    d1 = mxGetNumberOfDimensions(prhs[0]);
    dimsize = (mwSize*)mxGetDimensions(prhs[0]);
    m1 = dimsize[0];
    n1 = dimsize[1];
    
    img1n    = mxGetN(prhs[0]);    /*length of X*/
    img2n    = mxGetN(prhs[1]);    /*length of Y*/                
    img1m    = mxGetM(prhs[0]);    /*length of X*/
    img2m    = mxGetM(prhs[1]);    /*length of Y*/                
        
    if (d1!=d2){
        mexErrMsgTxt("Number of colour channels not the same!");
    }
    if ((d1!=2)&&(d1!=3)){
        mexErrMsgTxt("Wrong image dimensions!");
    }
    
    if (nrhs == 6){ /* Get input parameters for MDLT warping. */
        Hm = mxGetM(prhs[2]); /* Number of H matrixes in H. */
        
        X     = mxGetPr(prhs[3]);
        Y     = mxGetPr(prhs[4]);
                
        xn    = mxGetN(prhs[3]);    /*length of X*/
        yn    = mxGetN(prhs[4]);    /*length of Y*/                
        off = mxGetPr(prhs[5]);
    }    
    else{
        off = mxGetPr(prhs[3]);
        /*encl_rect = mxGetPr(prhs[4]);*/
    }
    
    /* Create matrix for the return arguments. */
    plhs[0] = mxCreateNumericMatrix(m1,n1,mxDOUBLE_CLASS,mxREAL);
    plhs[1] = mxCreateNumericArray(d1,dimsize,mxDOUBLE_CLASS,mxREAL);
    plhs[2] = mxCreateNumericArray(d2,dimsize,mxDOUBLE_CLASS,mxREAL);
    
    /* Assign pointers to outputs.*/
    mask = mxGetPr(plhs[0]);
    inten1 = mxGetPr(plhs[1]);
    inten2 = mxGetPr(plhs[2]);
    
    /* Start computations.*/
    count = 0;
    auxidx1 = m1*n1;
    auxidx2 = m2*n2;
    if (nrhs == 4){
        for(i=0;i<n1;i++)
        {
            for(j=0;j<m1;j++)
            {
                /* Map left to right.*/
                a = H[0]*(i-off[0]+1) + H[3]*(j-off[1]+1) + H[6];
                b = H[1]*(i-off[0]+1) + H[4]*(j-off[1]+1) + H[7];
                c = H[2]*(i-off[0]+1) + H[5]*(j-off[1]+1) + H[8]; 
                  if (c!=0)
                {
                    p = round(a/c);
                    q = round(b/c);
                }
                else{
                    continue;
                }
                /* Check if within boundary of right image.*/
                if ((p>1)&&(p<n2)&&(q>1)&&(q<m2))
                {        
                    ++count;
                    mask[i*m1+j] = count;
                    for (k=0;k<3;k++)
                    {
                        if ((k*(auxidx1)+i*m1+j)>=0 && (k*(auxidx1)+i*m1+j) < (d1*(auxidx1)) && (k*(auxidx1)+i*m1+j) >= 0 && (k*(auxidx1)+i*m1+j) < (img1m*img1n) && (k*(auxidx2)+(p-1)*m2+(q-1)) >=0 && (k*(auxidx2)+(p-1)*m2+(q-1)) < (img2m*img2n))
                        {
                            inten1[k*(auxidx1)+i*m1+j] = img1[k*(auxidx1)+i*m1+j];
                            inten2[k*(auxidx1)+i*m1+j] = img2[k*(auxidx2)+(p-1)*m2+(q-1)];
                        }
                    }                
                }
            }
        }
    }
    else{
        for(i=0;i<n1;i++)
        {
            for(j=0;j<m1;j++)
            {
                /* Get grid point for current pixel. */
                for(xinx=0; xinx < xn && i >= X[xinx]; xinx++);
                for(yinx=0; yinx < yn && j >= Y[yinx]; yinx++);
                
                if(i-X[xinx]>0.5)
                    xinx = xinx+1;

                if(j-Y[yinx]>0.5)
                    yinx = yinx+1;

                inx = yinx + xinx*xn;
                
                /* Map left to right.*/
                a = H[inx+(Hm*0)]*(i-off[0]+1) + H[inx+(Hm*3)]*(j-off[1]+1) + H[inx+(Hm*6)];
                b = H[inx+(Hm*1)]*(i-off[0]+1) + H[inx+(Hm*4)]*(j-off[1]+1) + H[inx+(Hm*7)];
                c = H[inx+(Hm*2)]*(i-off[0]+1) + H[inx+(Hm*5)]*(j-off[1]+1) + H[inx+(Hm*8)];
             
                if (c!=0)
                {
                    p = round(a/c);
                    q = round(b/c);
                }
                
                /* Check if within boundary of right image.*/
                if ((p>=1)&&(p<n2)&&(q>=1)&&(q<m2))
                {                    
                    ++count;
                    mask[i*m1+j] = count;

                    for (k=0;k<3;k++)
                    {
                        if ((k*(auxidx1)+i*m1+j)>=0 && (k*(auxidx1)+i*m1+j) < (d1*(auxidx1)) && (k*(auxidx1)+i*m1+j) >= 0 && (k*(auxidx1)+i*m1+j) < (img1m*img1n) && (k*(auxidx2)+(p-1)*m2+(q-1)) >=0 && (k*(auxidx2)+(p-1)*m2+(q-1)) < (img2m*img2n))
                        {
                            inten1[k*(auxidx1)+i*m1+j] = img1[k*(auxidx1)+i*m1+j];
                            inten2[k*(auxidx1)+i*m1+j] = img2[k*(auxidx2)+(p-1)*m2+(q-1)];
                        }
                    }                   
                }
            }
        }    
    }
    /* Bye bye.*/
    return;
}