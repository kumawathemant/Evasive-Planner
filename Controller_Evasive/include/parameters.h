/*

Github: kumawathemant553

========================================
Car Parameters to be used later in lqr

*/

/*
Some functions are copied from LOCO implemenmntation

Additional drift control functions introduced

*/


#ifndef _PARAMETERS_H_
#define _PARAMETERS_H_

#include "libfunctions.h"
#include "ilqr.h"

class SeDriCa : public iLQR{
  ///============================================
  ///Car Parameters

  const double m; // mass of car
  const double g; //gravity
  const double L ; // wheelbase in metres
  const double b; //CoG to rear axle
  const double a; //cog to front axle
  const double G_front; //Calculate load
  const double G_rear;
  const double C_x;   // logitudinal stiffness
  const double C_alpha; //lateral stiffness
  const double Iz; //Inertia
  const double mu; //G_rear;
  const double mu_spin;// G_rear

  ///==============================================
  ///CONTROL COST PARAMS
  Vec2d cu; //Control COst
  Vec2d cdu; //change in control cost  
  VecXd cx; //running cost for velocites
  VecXd px; //smoothness sclaes for running cost
  double c_drift; //Reward for drifting
  double kp_obs; //Obstacle avoidance costs
  double kv_obs;
  double dist_thres;

  Vec2d tire_dynamics(double Ux, double Ux_cmd, double mu, double mu_slide,
                      double Fz, double C_x, double C_alpha, double alpha);
  VecXd dynamics(const VecXd &x, const VecXd &u);
  double cost(const VecXd &x, const VecXd &u);
  double final_cost(const VecXd &x);
  VecXd Integrated_dynamics(const VecXd &x, const VecXd u);

public:
  Vec2d obstacle;
  SeDriCa():  m(2.35), g(9.81), L(0.257), a(0.11372), b(0.14328),
              C_x(65), C_alpha(55), Iz(0.025), mu(0.45), mu_spin(0.2),
              G_front(12.852550506), G_rear(10.200949494),
              c_drift(-0.001), kp_obs(0.5), kv_obs(0.1), dist_thres(0.5)
  {
    cu << 1e-3, 1e-3;
    cdu<< 5 , 5e-1;
    cx.resize(6);
    px.resize(6);
    cx << 0.5, 0.1, 0.4, 0.05, 0.005, 0.002;
    px << 0.01, 0.01, 0.1, 0.01, 0.01, 0.1;

  }

}

#endif
