#include "mex.h"
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <iostream>

#include "ceres/ceres.h"
#include "glog/logging.h"

using namespace std;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solver;
using ceres::Solve;
    
/*
mex ceresError.cpp /usr/local/lib/libceres_shared.so /usr/local/lib/libglog.so
 */

struct projectionError {
  projectionError(double observed_x, double observed_y, double normaliser, double weight)
      : observed_x(observed_x), observed_y(observed_y), normaliser(normaliser), weight(weight) {}

  template <typename T>
  bool operator()(const T* const H, const T* const point, T* residuals) const {
               
    T aux = point[0]*H[2] + point[1]*H[5] + H[8];
    T predicted_x = (point[0]*H[0] + point[1]*H[3] + H[6]) / aux;
    T predicted_y = (point[0]*H[1] + point[1]*H[4] + H[7]) / aux;
    
    aux = T(weight/normaliser);
    // The error is the (weighted) difference between the predicted and observed position.
    residuals[0] = aux * (predicted_x - T(observed_x));
    residuals[1] = aux * (predicted_y - T(observed_y));
    
    return true;
  }

  // Factory to hide the construction of the CostFunction object from
  // the client code.
  static ceres::CostFunction* Create(const double observed_x,
                                     const double observed_y,
                                     const double normaliser,
                                     const double weight) {
    return (new ceres::AutoDiffCostFunction<projectionError, 2, 9, 2>(
                new projectionError(observed_x, observed_y, normaliser, weight)));
  }

  double observed_x;
  double observed_y;
  double normaliser;
  double weight;
};

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    /* Input/output variables. */
    double *origba_params;    
    double *arr_xik_idx;
    int num_homographies;
    double *observations;            
    double *normaliser;
    double *weights;
    
    double *finalba_params;

    /* Intermediate variables.*/
    double *ba_params;
    mxArray *xik_idx[1];
    int ba_paramsm,observationsm_db2;
    int i,j,c=0;
    int id_jc,id_xik,id_i;
    int num_Hparams,lenarr_xik_idx;
    
    /* Assign pointers to inputs.*/
    origba_params = mxGetPr(prhs[0]);
    ba_paramsm    = mxGetM(prhs[0]);
    normaliser       = mxGetPr(prhs[2]);
    num_homographies = mxGetScalar(prhs[3]);
    observations = mxGetPr(prhs[4]);
    observationsm_db2 = mxGetM(prhs[4]) / 2;
    weights = mxGetPr(prhs[5]);
    num_Hparams = 9*num_homographies;
       
    /* Get sizes of input matrices (images, transformations, etc.).*/
    ba_params = (double*)malloc(ba_paramsm*sizeof(double));                
    memcpy(ba_params,origba_params,ba_paramsm*sizeof(double));
    
    plhs[0] = mxCreateDoubleMatrix(ba_paramsm,1,mxREAL);
    finalba_params = mxGetPr(plhs[0]);    
    
    /* Start computations.*/        
    Problem problem;          
    for(i=0;i<num_homographies;i++)
    {
        xik_idx[0]  = mxGetCell(prhs[1],i);
        arr_xik_idx = mxGetPr(xik_idx[0]);
        lenarr_xik_idx = mxGetN(xik_idx[0]);
        
        id_i = i*9;
        for(j=0;j<lenarr_xik_idx;j++)
        {
            id_xik = arr_xik_idx[j]-1;
            id_jc = j+c;

            ceres::CostFunction* cost_function =
            projectionError::Create(observations[id_jc],
                                             observations[id_jc+observationsm_db2],
                                             normaliser[id_xik],
                                             weights[id_xik]);
            
            problem.AddResidualBlock(cost_function,
                             NULL /* squared loss */,
                             ba_params+id_i,
                             ba_params+num_Hparams+(id_xik*2));
        }
        c += lenarr_xik_idx;
    }
  
    // Make Ceres automatically detect the bundle structure. Note that the
    // standard solver, SPARSE_NORMAL_CHOLESKY, also works fine but it is slower
    // for standard bundle adjustment problems.
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 10;
        
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    /*std::cout << summary.FullReport() << "\n";*/

    memcpy(finalba_params,ba_params,ba_paramsm*sizeof(double));
    
    return;
}
