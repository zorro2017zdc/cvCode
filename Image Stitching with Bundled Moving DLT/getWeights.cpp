#include "mex.h"
#include <time.h>
#include <math.h>
#include <string.h>

/*Round function.*/
double round(double x) { return (x-floor(x))>0.5 ? ceil(x) : floor(x); }

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Input/output variables. */
    double *mvts;
    double *data;
    double sigma;
    double gamma;
    
    double *Wi; 
    double *sum;
  
    /* Intermediate variables.*/
    int datam,datan;      
    int i, j;
    
    /* Check for proper number of arguments. */    
    if (nrhs != 4)
    {
        mexErrMsgTxt("Four inputs required.");
    }
    else if (nlhs > 2)
    {
        mexErrMsgTxt("Wrong number of output arguments.");
    }
  
    /* Assign pointers to inputs. */
    mvts = mxGetPr(prhs[0]);
    data = mxGetPr(prhs[1]);
    sigma   = mxGetScalar(prhs[2]);
    gamma   = mxGetScalar(prhs[3]);
    
    /* Get sizes of input matrices (images, transformations, etc.).*/
    datam = mxGetM(prhs[1]);
    datan = mxGetN(prhs[1]);
    
    /* Create matrix for the return arguments. */
    plhs[0] = mxCreateDoubleMatrix(1,datam,mxREAL);
    plhs[1] = mxCreateDoubleMatrix(1,1,mxREAL);
         
    /* Assign pointers to output canvas (warped image2). */
    Wi  = mxGetPr(plhs[0]);
    sum = mxGetPr(plhs[1]);
    
    /* Start computations. */
    sigma = pow(sigma,2);
    sum[0] = 0;
    for(i=0;i<datam;i++)
    {
        Wi[i] = exp(-sqrt(pow(data[i]-mvts[0],2) + pow(data[i+datam]-mvts[1],2)) / sigma);
        if (Wi[i] < gamma)
            Wi[i] = gamma;
        sum[0] += Wi[i];
    }
    for(i=0;i<datam;i++)
        Wi[i] /= sum[0];

    /* Bye bye.*/
    return;
}
