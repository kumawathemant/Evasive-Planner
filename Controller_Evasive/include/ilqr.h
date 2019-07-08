/*

Github: kumawathemant553
========================================
ILQR IMPLEMENTATION 

*/


#ifndef _ILQR_H_
#define _ILQR_H_

#include "libfunctions.h"

class iLQR{
	const double tolFun;
	const double tolGrad;
	const double max_iter;
	double lambda;
	double dlambda;

	const double lambdaFactor;
	const double lambdaMax;
	const double lambdaMin;
	const double zMin;

	const int nstates;  //dimension of state vector
	const int ncontrols;  // dimension of control vector

	int T; // number of state transition (#Timesteps -1)
	VecXd Alpha;

	///=============================================
	///Tacking Process
	VecOfVecXd xs; //states from last trajectory
	VecOfVecXd us; //control from last trajectory
	VecOfVecXd ls;
	VecOfVecXd Ls;

	///Misc Params
	MatXd control_limits;
	VecXd x_current;
	VecXd u_current;
	double cost_s;

	///=============================================
	///VIRTUAL FUNCTION 
	/// To be chnaged in SeDriCa car param file
	virtual VecXd dynamic(const VecXd &x, const VecXd &u) =0; /// Gonna compute my dynamics
	virtual VecXd Integrated_dynamics(const VecXd &x, const VecXd &u) =0;
	virtual double cost(const VecXd &x, const VecXd &u) =0;
	virtual double final_cost(const VecXd &x) = 0;

	///=============================================
	/// FORWARD and BACKWARD Pass
    void forward_pass(const VecXd &x0, const VecOfVecXd &u,
  					VecOfVecXd &xnew, VecOfVecXd &unew, double &new_cost,
  					const VecOfVecXd &x, const VecOfMatXd &L);
    int backward_pass(const VecOfVecXd &cx, const VecOfVecXd &cu, const VecOfMatXd &cxx, const VecOfMatXd &cxu,
  					const VecOfMatXd &cuu, const VecOfMatXd &fx, const VecOfMatXd &fu, const VecOfVecXd &u,
  					VecOfVecXd &Vx, VecOfMatXd &Vxx, VecOfVecXd &k, VecOfMatXd &K, Vec2d &dV);

    int boxQP(MatXd &H, VecXd &g, VecXd &x0, VecXd &x, MatXd &Hfree, VecXd &free);
    VecXd clamp_to_limits(VecXd &u);

    MatXd row_w_ind(MatXd &mat, VecXd &rows);
    VecXd subvec_w_ind(VecXd &vec, VecXd &indices);
    VecOfVecXd adjust_u(VecOfVecXd &u, VecOfVecXd &l, double alpha);

    ///=================================================
    ///Derivatives Calculation

    void compute_derivatives(const VecOfVecXd &x, const VecOfVecXd &u, VecOfMatXd &fx, 
    						 VecOfVecXd &fu, VecOfVecXd &cx, VecOfVecXd &cu,
    						 VecOfMatXd &cxx, VecOfMatXd &cxu, VecOfVecXd &cuu);
    void get_dynamics_drivatives(const VecOfMatXd &x, const VecOfMatXd &u,
    							 VecOfMatXd &fx, VecOfMatXd &fu);
    void get_cost_derivatives(const VecOfVecXd &x, const VecOfVecXd &u,
    						  VecOfVecXd &cx, VecOfVecXd &cu);
    void get_cost_2nd_derivatives(const VecOfVecXd &x, const VecOfVecXd &u, 
    								VecOfMatXd &cxx, VecOfMatXd &cxu, VecOfMatXd &cuu);

public:
	const double timeDelta; //dt for euler integration
	VecXd x_d;

  iLQR(): tolFun(1e-6), tolGrad(1e-6), maxIter(30), lambda(1),
                dlambda(1), lambdaFactor(1.6), lambdaMax(1e11), lambdaMin(1e-8),
                zMin(0), n(6), m(2), T(50), timeDelta(0.05)
    {
    	Alpha.resize(11);
    	Alpha<<1.0000, 0.5012, 0.2512, 0.1259, 0.0631, 0.0316, 0.0158, 0.0079, 0.0040, 0.0020, 0.0010;

    	control_limits.resize(2,2);
    	control << -1, 4, // FOr throttle
    				-0.76, 0.68; // For steering /min / max
    }

    double init_traj(VecXd &x_0, VecOfVecXd &u_0);
    void generate_trajectory(const VecXd &x_0, init trajectoryLength);

   	///Funtions to Test
   	void demoQp();
   	void output_to_csv();	





};

#endif