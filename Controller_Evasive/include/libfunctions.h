/*

Github: kumawathemant553

========================================
///COntains some important functions to be used later


*/
#ifndef _libfunctions_INCLUDES_H_
#define _libfunctions_INCLUDES_H_


///====================================
///Dependencies
#include <vector>
#include <eigen/Eigen/Core>
#include <eigen/Eigen/Eigenvalues>
#include <eigen/Eigen/StdVector>
#include <iostream>
#include <math.h>
#include <time.h>
#define Eye2 Eigen::Matrix2d::Identity(2,2);
#define EIGEN_USE_NEW_STDVECTOR
using namespace std;


///========================================
///Eigen definations
typedef Eigen::Vector2d Vec2d;
typedef Eigen::VectorXd VecXd;
typedef Eigen::MatrixXd MatXd;

typedef std::vector<VecXd, Eigen::aligned_allocator<VecXd> > VecOfVecXd;
typedef std::vector<MatXd, Eigen::aligned_allocator<MatXd> >  VecOfMatXd;


///==========================================
///Funtions defination
///Some of the functions copied from LOCO team implemnetation

double pi = M_PI;

template<typename T>
inline T sqr(const T &val){
	return val*val;
}

template<typename T>
inline T cube(const T&val){
	return val*val*val;
}

template<typename>
inline int sgn(T&val){
	return ((T(0)<val)-(val <T(0)));
}

inline double sabs(double x, double y){
	return sqrt(sqr(x) + sqr(y)) - y;
}

///==========================================
///IMPLEMENTATION OF MATLAB Wrap TO Pi function
template<typename T>
inline T Mod(T x, T y)
{
    static_assert(!std::numeric_limits<T>::is_exact , "Mod: floating-point type expected");

    if (0. == y)
        return x;
    double m= x - y * floor(x/y);
    // handle boundary cases resulted from floating-point cut off:
    if (y > 0)              // modulo range: [0..y)
    {
        if (m>=y)           // Mod(-1e-16             , 360.    ): m= 360.
            return 0;

        if (m<0 )
        {
            if (y+m == y)
                return 0  ; // just in case...
            else
                return y+m; // Mod(106.81415022205296 , _TWO_PI ): m= -1.421e-14
        }
    }
    else                    // modulo range: (y..0]
    {
        if (m<=y)           // Mod(1e-16              , -360.   ): m= -360.
            return 0;

        if (m>0 )
        {
            if (y+m == y)
                return 0  ; // just in case...
            else
                return y+m; // Mod(-106.81415022205296, -_TWO_PI): m= 1.421e-14
        }
    }

    return m;
}

inline double wrap_to_pi(double angle)
{
  return Mod(angle+pi, 2*pi) - pi;
}


///==============================================
///Helper functions
template <typename T>
inline void print_vec(T vec){
	for(int i =0;i<vec.size();i++){
		cout<< vec(i)<<' ';
	}
	cout<<'\n';
}


inline VecXd elem_square(const VecXd &vec){
	return vec.array().square().matrix();
}

inline VecXd elem_sqrt(const VecXd &vec){
	return vec.array().sqrt().matrix();
}

inline VecXd sabs(const VecXd &vec, const VecXd &p){
	VecXd var_sum = elem_sqrt(elem_square(vec)+ elem_square(p));
	return var_sum - p;
}
/// Will push one new element to the array
/// To0 slow
///WARNING : DO not use this much
/// changed now
inline void push_back(VecXd& vec, const double &val){
	int old_length = vec.size();
	vec.resize(old_length+1);
	vec(old_length) = val;

}



#endif
