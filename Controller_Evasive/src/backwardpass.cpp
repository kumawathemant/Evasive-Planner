#include "iLQR.h"
using namespace std;


/// CHnages the row onoly if the inmdices i is greater than zero;
MatXd iLQR::rows_w_ind(MatXd &mat, VecXd &indices){
	MatXd submat;
	if(mat.rows()!= indices.size()){
		cout<<" Mat.rows size not equal to indices size \n";
		return submat;
	}

	for(int i =0;i<indices.size(); i++){
		if(indices(i)>0){
			submat.conservativeResizeLike(MatXd(submat.rows()+1, mat.cols()));
			submat.row(submat.rows()-1) = mat.row(i);
		}
	}
	return submat;
}

/// Will return Contorl gains K and k for control estimation
int iLQR::backward_pass(const VecOfVecXd &cx, const VecOfVecXd cu, const VecOfMatXd &cxx,
												const VecOfMatXd &cxu, const VecOfMatXd &cuu, const VecOfMatXd &fx,
												const VecOfMatXd &fu, const VecOfVecXd &u, VecOfVecXd &Vx, VecOfMatXd &Vxx,
												VecOfVecXd &k, VecOfMatXd &K, Vec2d &dV)
{
	///==================================================================
	/// Backward Pass inputs
	/* cx : n * (T+1)  				cu: 2*(T+1)
	   cxx: n*n*(T+1)   			cxx: n*n*(T+1) cuu: 2*2*(T)
		  c
	*/
	///=====================================================================
	///Cost -to - go end
	Vx[T] = cx[T];
	Vxx[T] = cxx[T];

	///===================================================================
	/// Initialize Q matriz;
	VecXd Qx(nstates);
	VecXd Qu(ncontrols);
	MatXd Qxx(nstates,ncontrols);
	MatXd Qux(ncontrols,nstates);
	MatXd Quu(ncontrols,ncontrols);
	VecXd k_i(ncontrols);
	MatXd K_i(ncontrols, nstates);

	for(int i =T-1; i>=0; i--){
		Qx = cx[i] + fx[i].transpose()*Vx[i+1];
		Qu = cu[i] + fu[i].transpose()*Vx[i+1];


		/// Printing some values
		///========================
		std::cout << "fu:\n" << fu[i] << '\n';
		std::cout << "Vx: \n" << Vx[i+1] << '\n';
		std::cout << "cu: \n" << cu[i] << '\n';
		///=============================

		Qxx = cxx[i] + fx[i].transpose()*Vxx[i+1]*fx[i];

		Qux = cxu[i].transpose() + fu[i].transpose()*Vxx[i+1].fx[i];

		Quu = cuu[i] + fu[i].transpose()*Vxx[i+1]*fu[i];

		/// We have already regularized the Vxx ;

		MatXd Vxx_reg = Vxx[i+1];
		MatXd Qux_reg = cxu[i].transpose() + fu[i].transpose()*Vxx_reg*fx[i];
		// Eye2 is 2*2indentity Matrix
		MatXd QuuF = cuu[i] + fu[i].transpose()*Vxx_reg*fu[i] + lambda*Eye2;

		///===================================
		/// Impose limits based on boxQP

		VecXd k_i(ncontrols);
		VecXd R(ncontrols,ncontrols);
		VecXd free_v(ncontrols);






	}

}
