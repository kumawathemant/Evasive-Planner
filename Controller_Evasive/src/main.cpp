/*
Main file for iLQR implemnetation
*/

#include "parameters.h"
using namespace std;

int main(){
    SeDriCa batmobile;
    VecXd x0(6);
    x0<< 0,0,0,2,0,0;
    VecXd x_d(6);
    x_d<< 1,1,0,0,0,0;
    VecXd obs(2);
    obs<< 1, 0;

    batmobile.obstacle = obs;
    batmobile.x_d = x_d;
    int T =50;

    VecOfVecXd u0;
    Vec2d u_int(1,0.3);
    for(int i=0;i<T;i++){
        u0.push_back(u_init);
    }

    std::clock_t start;
    start = std::clock();

    batmobile.init_traj(x0,u0);
    batmobile.generate_trajectory(x0, T);
    batmobile.output_to_csv();

    double time_elapsed = (std::clock() - start) / (double)(CLOCKS_PER_SEC);
    std::cout << "Took " << time_elapsed << " seconds.\n";

}
