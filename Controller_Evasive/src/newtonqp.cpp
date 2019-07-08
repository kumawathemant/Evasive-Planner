/*
IMPLEMENTATION OF PROJECTED NEWTON QP OPTIMIZER

BASED ON PAPER :Tassa, Yuval, Nicolas Mansard, and Emo Todorov. "Control-limited differential dynamic programming." Robotics and Automation (ICRA), 2014 IEEE International Conference on. IEEE, 2014.

*/



#include "iLQR.h"
using namespace std;

#define CHECKPARAM 1

///========================================================================
/// RETURNS THE CONTROL OUTPUT IN CONTROL LIMITS;
/// ONLY CLAMPS, DOES NOT CHECK MINIMUM AMOUNT
VecXd iLQR::clamp_to_limits(VecXd &u){
  VecXd u_clamped(ncontrols);
  u_clamped(0)= min(control_limits(0,1), max(control_limits(0,0),u(0)));
  u_clamped(1)= min(control_limits(1,1),max(control_limits(1,0),u(1)));

  return u_clamped;
}
///=========================================================================
///return sub vector of vec only if index of thta value is greater than zero
///?????????????????????????????????????????????????????????????????????????????????????
///TODO : OPTIMIZE this function
VecXd iLQR::subvec_w_ind(VecXd &vec, VecXd &indices){
  //VecXd subvec;
  std::vector<double> store_var;
  for(int i=0;i<vec.size();i++){
    if(indices(i)>0){
        store_var.push_back(var(i));
    }
  }
  VecXd subvec(store_var.size());
  for(int i=0;i<store_var.size();i++){
      subvec(i) = store_var[i];
  }
  return subvec;
}
///=======================================
///
int iLQR::boxQP(Matxd &H,VecXd &g, VecXd &x0, VecXd &x, MatXd& Hfree, VecXd &free_v){
    /*

        Minimize  0.5x'*H*x + x'*G such that lower <= x <= upper

        H == positive definite matrix from QuuF
        g == bias vector from Qu
        X0 == initial state
        Lower and Upper bounds from control limits


    */
    free_v.resize(nstates);
    int maxIter     =100;// Max no of iterations
    double minGrad  =1e-8;//min norm of non fixed gradient
    double minRelImprove =1e-8;//min relative improvement
    double stepDec =0.6; //factor for decreasing step size;
    double minStep = 1e-22; //minimal stepsize for linesearch
    double Armijo =0.1; // Armijo param // fraction of linear improvement

    VecXd lower = control_limits.col(0);
    VecXd upper = control_limits.col(1);

    //Initial State
    x = clamp_to_limits(x0);

    //Intial Objective value
    double value = x.dot(g) + 0.5x.transpose()*H*x;// Intial start value for functions
    if(CHECKPARAM) {
        std::cout << "================\nLets start BoxQP \n Dimension::"<<m<<" Initial value :: "<<value<< '\n';
    }
    int result = 0;
    int nfactor =0;
    double oldvalue =0;
    bool factorize = false;

    VecXd clamped(nstates);
    VecXd old_clamped(nstates);
    for(int iter=1;i<=maxIter;iter++){

        //RESULT CHECK
        if(result!=0){
            break;
        }
        //=================================

        //Result improvement CHECK
        if(iter>1 && (oldvalue - value)<minRelImprove*std::abs(oldvalue)){
            result =4;
            break;
        }
        //===================================

        //===================================
        oldvalue =value;
        VecXd grad = g+ H*x;

        old_clamped = clamped;
        clamped.setZero();
        free_v.setOnes();
        for(int i=0;i<nstates;i++){
            if(x(i)==lower(i)&&grad(i)>0){
                clamped(i)=1;
                free_v(i)=0;
            }

            else if(x(i)==upper(i) && grad(i)<0){
                clamped(i) =1;
                free_v(i) =0;
            }
        }
        if(clamped.all() == nstates){
            result =6;
            break;
        }

        if(iter ==1){
            factorize = true;
        }

        else if((old_clamped-clamped).sum()!= 0){
            factorize = true;
        }

        else {
            factorize = false;
        }

        if(factorize){
            int n_free = free_v.sum();
            MatXd Hf;

            if(free_v[0]==1){
                Hf = H.block(0,0,n_free,n_free);
            }
            else{
                Hf = H.block(1,1,n_free,n_free);
            }
            ///Cholesky Decomposition of Hf
            Eigen::LLT<MatXd> lltOfHf(Hf);
            Hfree = lltOfHf.matrixL().transpose;
            nfactor++;
        }

        ///=====================================
        ///Gradient norm Check
        double gnorm = grad.cwiseProduct(free_v).norm();
        if(gnorm < minGrad){
            result =5;
            break;
        }
        ///Let the search begin
        VectorXd grad_clamped = g+ H*(x.cwiseProduct(clamped));
        VecXd search(nstates);
        search.setZero();

        if(free_v[0]==1 && free_v[1]==1){
            search = -Hfree.inverse()*(Hfree.transpose().inverse*subvec_w_ind(grad_clamped,free_v) - subvec_w_ind(x,free_v));
        }
        else if(free_v[0]==1){
            search(0) = (-Hfree.inverse() * (Hfree.transpose().inverse()*subvec_w_ind(grad_clamped, free_v)) - subvec_w_ind(x, free_v))(0);
        }
        else if(free_v[1]==1){
            search(1)=(-Hfree.inverse() * (Hfree.transpose().inverse()*subvec_w_ind(grad_clamped, free_v)) - subvec_w_ind(x, free_v))(0);
        }

        /// descent direction Check
        double sdotg = (search.cwiseProduct(grad)).sum();
        ///================================================
        ///SHOULD NEVER HAPPEN;

        if(sdotg >=0){
            break;
        }
        //==================================================

        ///LINE search
        double step =1;
        int nstep =0;
        VecXd reach = x+ step*search;
        VecXd xc = clamp_to_limits(reach);
        double vc = xc.dot(g) + 0.5*xc.dot(H*xc);

        ///Search Loop starts
        while((vc - oldvalue)/(step*sdotg)<Armijo){
            step *= stepDec;
            nstep++;
            reach = x+ step*search;
            xc = clampe_to_limits(reach);
            vc = xc.dot(g) + 0.5*xc.dot(H*xc);
            if(step < minStep){
                result =2;
                break;
            }
        }

        if(CHECKPARAM){
            printf("iter %-3d  value % -9.5g |g| %-9.3g  reduction %-9.3g  linesearch %g^%-2d  n_clamped %d\n",
				iter, vc, gnorm, oldvalue-vc, stepDec, nstep, int(clamped.sum()));
        }

        x = xc;
        value = vc;

    }
    return result;

}
