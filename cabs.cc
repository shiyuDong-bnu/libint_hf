#include <string>
#include <vector>
#include <libint2.hpp>
#include <algorithm>
#include <iomanip>
#include<iostream>
#include<fstream>
#include <typeinfo>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <Eigen/SparseCholesky>
using namespace std;

using real_t = libint2::scalar_type;
typedef Eigen::Matrix<real_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>
        Matrix;
typedef Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::RowMajor> Vector;
Matrix ao_overlap(libint2::BasisSet obs,libint2::BasisSet abs);
int main(int argc, char* argv[]){
    libint2::initialize();
    // reading input molecule
    string xyzfile ="Ne.xyz";
    ifstream inputfile(xyzfile);
    vector<libint2::Atom> atoms= libint2::read_dotxyz(inputfile);
    std::cout <<"Number of atoms is "<<atoms.size()<<endl;
    // load basis 
    libint2::BasisSet obs("aug-cc-pVDZ",atoms);
    libint2::BasisSet abs("aug-cc-pVDZ-optri",atoms);
    // get information of all basis set
    int n_obs=obs.nbf();
    int n_abs=abs.nbf();
    int n_union=n_obs+n_abs;
    std::cout<<" nobs "<<n_obs;
    std::cout<<" n_abs "<<n_abs;
    //calculate overlap matrix of all.
    // S_obs_abs
    Matrix S_obs_abs=ao_overlap(obs,abs);
    // S_obs_obs 
    Matrix S_obs_obs=ao_overlap(obs,obs);
    // S_abs_abs
    Matrix S_abs_abs=ao_overlap(abs,abs);
    // Now show the rank of the union basis set
    Matrix S_union=Matrix::Zero(n_union, n_union);
    S_union.topLeftCorner(n_obs,n_obs)=S_obs_obs;
    S_union.bottomRightCorner(n_abs,n_abs)=S_abs_abs;
    S_union.topRightCorner(n_obs,n_abs)=S_obs_abs;
    S_union.bottomLeftCorner(n_abs,n_obs)=S_obs_abs.transpose();
    real_t lin_dep=1e-8;
    Eigen::SelfAdjointEigenSolver<Matrix> es(S_union);
    std::cout<< es.eigenvalues();
    // check linear dependece in uniton basis set
    int n_redundancy=0;
    for (auto eig:es.eigenvalues()){
        if (eig<lin_dep){
            n_redundancy+=1;
        }
        else{
            break;
        }
    }
    std::cout<<"There are "<<n_redundancy<<" redundancy basis."<<std::endl;
    // test ri orbitals are orthonormal
    auto eigv=es.eigenvalues();
    auto sqrt_eigv=eigv.cwiseSqrt();
    auto sqrt_inv_eigv=sqrt_eigv.cwiseInverse();
    Matrix diag=es.eigenvalues().asDiagonal();
    Matrix eigvec=es.eigenvectors();
    Matrix S_union_b=eigvec* diag*eigvec.transpose();
    Matrix diag_invsq=(eigv).cwiseSqrt().cwiseInverse().asDiagonal();
    Matrix S_invsqrt=eigvec*diag_invsq*eigvec.transpose();
    // project to RI basis
    // project to MO basis
    Matrix RI_coef=S_invsqrt;
    std::cout<<(RI_coef.transpose()*S_union*RI_coef-Matrix::Identity(n_union,n_union)).norm();
    std::cout<<"above is norm expected zero .";
    Matrix S_obs_union=S_union.topLeftCorner(n_obs,n_union);
    // transform from raw ao_ao to ao_ri 
    Matrix S_obs_ri=S_obs_union*RI_coef;


    // Now calculate cabs
    

    Eigen::JacobiSVD<Matrix> svd(S_obs_ri, Eigen::ComputeFullU | Eigen::ComputeFullV);
    //  check svd
    std::cout<<"Check svd."<<std::endl;
    std::cout << "matrixU shape: " << svd.matrixU().rows() << "x" << svd.matrixU().cols() << std::endl;
    // Output the number of rows and columns of matrixV
    std::cout << "matrixV shape: " << svd.matrixV().rows() << "x" << svd.matrixV().cols() << std::endl;
    // Output the singular values
    std::cout << "Singular values: " << svd.singularValues().size() << std::endl;

    Matrix svdeigenvalue=Matrix::Zero(n_obs,n_union);
    svdeigenvalue.topLeftCorner(n_obs,n_obs)=svd.singularValues().asDiagonal();
    std::cout << "m1 shape: " << svd.matrixU().rows() << "x" << svd.matrixU().cols() << std::endl;
    std::cout << "m2 shape: " << svdeigenvalue.rows() << "x" << svdeigenvalue.cols() << std::endl;
    std::cout << "m3 shape: " << svd.matrixV().rows() << "x" << svd.matrixV().cols() << std::endl;
    Matrix svd_recover=svd.matrixU()*svdeigenvalue*svd.matrixV().transpose();
    int n_cabs=n_union-svd.singularValues().size();
    std::cout<<(S_obs_ri-svd_recover).norm()<<std::endl;
    // svd done 
    // now ectract coeff.
    std::cout<<"Number of CABS basis: "<<n_cabs<<std::endl;
    Matrix V=svd.matrixV();
    Matrix Coeff_ri_cabs=V.topRightCorner(n_union,n_cabs);
    std::cout<< (S_obs_ri*Coeff_ri_cabs).norm()<<std::endl;

    // transform from ri to ao
    Matrix Coeff_ao_cabs=RI_coef*Coeff_ri_cabs;
    // test overlap between cabs and cabs
    Matrix S_cabs_cabs=Coeff_ao_cabs.transpose()*S_union*Coeff_ao_cabs;
    std::cout << "S_cabs_cabs shape: " << S_cabs_cabs.rows() << "x" << S_cabs_cabs.cols() << std::endl;
    Matrix idendity=Matrix::Identity(n_cabs,n_cabs);
    std::cout<<(idendity-S_cabs_cabs).norm()<<std::endl;
    //test orthogonal of cabs and ao
    std::cout<< (S_obs_union*Coeff_ao_cabs).norm()<<std::endl;
}
Matrix ao_overlap(libint2::BasisSet obs,libint2::BasisSet abs){
    int n_obs=obs.nbf();
    int n_abs=abs.nbf();
    int maxl_obs=obs.max_l();
    int maxl_abs=abs.max_l();
    int max_n_prim_obs=obs.max_nprim();
    int max_n_prim_abs=abs.max_nprim();
    int n_obs_shell=obs.size();
    int n_abs_shell=abs.size();
    auto  obs_shell2bf=obs.shell2bf();
    auto  abs_shell2bf=abs.shell2bf();
    std::cout<<" nobs "<<n_obs;
    std::cout<<" n_abs "<<n_abs;
    //calculate overlap matrix of all.
    // set up S_engine
    int max_nprim=max_n_prim_obs;
    if (max_n_prim_obs<max_n_prim_abs){
        max_nprim=max_n_prim_abs;
    }
    int max_l=maxl_obs;
    if (maxl_obs<maxl_abs){
        max_l=maxl_abs;
    }
    libint2::Engine s_engine(libint2::Operator::overlap,
                max_nprim,
                max_l
                ); 
    const auto& buf = s_engine.results();
    // S_obs_abs
    Matrix S_obs_abs = Matrix::Zero(n_obs, n_abs);
    for (int i=0;i<n_obs_shell;i++){
        for (int j=0;j<n_abs_shell;j++){
            auto first=obs[i]; 
            auto second=abs[j]; 
            s_engine.compute(first ,second);
            // transfer into eigen buf
            auto n1=first.size();
            auto n2=second.size();
           // std::cout<<"N1 " <<n1<<" N2"<<n2<<std::endl;
            Eigen::Map<const Matrix> buf_mat(buf[0],n1,n2);
            // check block index using shell2bf
            int bind1=obs_shell2bf[i];
            int bind2=abs_shell2bf[j];
           // std::cout<<"Buffer is \n"<<buf_mat<<std::endl;
            S_obs_abs.block(bind1,bind2,n1,n2)=buf_mat;
        }
    }
    return S_obs_abs;
}