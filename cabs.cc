#include <string>
#include <vector>
#include <libint2.hpp>
#include <algorithm>
#include <iomanip>
#include<iostream>
#include<fstream>
#include <typeinfo>
#include <Eigen/Dense>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xcsv.hpp>
#include <xtensor/xnpy.hpp>
#include "xtensor-blas/xlinalg.hpp"
#include <stdexcept>
#include <xtensor/xnorm.hpp>
using namespace std;

using real_t = libint2::scalar_type;
typedef Eigen::Matrix<real_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>
        Matrix;
 using XTMatrix = xt::xarray<real_t, xt::layout_type::row_major>;
// write a general function to calculte one-body integral
// Matrix compute_1body_ints(const libint2::BasisSet basis,
//                           libint2::Operator t,
//                           const std::vector<libint2::Atom>& atoms = std::vector<libint2::Atom>());
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
    S_obs_abs=ao_overlap(obs,abs);
    // S_obs_obs 
    Matrix S_obs_obs=ao_overlap(obs,obs);
    // S_abs_abs
    Matrix S_abs_abs=ao_overlap(abs,abs);
        // save it .
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    ofstream file("S_obs_abs.csv");
	if (file.is_open())
	{
		file << S_obs_abs.format(CSVFormat);
		file.close();
	}
    ofstream file1("S_obs_obs.csv");
	if (file1.is_open())
	{
		file1 << S_obs_obs.format(CSVFormat);
		file1.close();
	}
    ofstream file2("S_abs_abs.csv");
	if (file2.is_open())
	{
		file2 << S_abs_abs.format(CSVFormat);
		file2.close();
	}
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