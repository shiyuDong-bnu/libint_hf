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
XTMatrix ao_eri(libint2::Engine eri_engine,libint2::BasisSet obs);
Matrix  ao_overlap(libint2::Engine s_engine,libint2::BasisSet obs);
int main(int argc, char* argv[]){
    libint2::initialize();
    // reading input molecule
    string xyzfile ="Ne.xyz";
    ifstream inputfile(xyzfile);
    vector<libint2::Atom> atoms= libint2::read_dotxyz(inputfile);
    std::cout <<"Number of atoms is "<<atoms.size()<<endl;
    auto nelectron = 0;
    for (auto i = 0; i < atoms.size(); ++i) nelectron += atoms[i].atomic_number;
    const auto ndocc = nelectron / 2;
    std::cout<<"docc obs number is "<<ndocc<<std::endl;
    // build basis
    libint2::BasisSet obs("aug-cc-pVDZ",atoms);
    libint2::BasisSet cabs("aug-cc-pVDZ-optri",atoms);
    // Get information of basis set
    int max_l=obs.max_l();
    int max_nprim= obs.max_nprim();
    std::cout<<"max l is "<<max_l<<endl;
    std::cout<<"max nprim is"<<max_nprim<<endl;
    int n_shell=obs.size();
    std::cout<<"The basis set has "<<n_shell<<" shells"<<endl;
    for (int i=0;i<n_shell;++i){
        int n_basis=obs[i].size();
        std::cout<<"shell "<< i<<" has "<<n_basis<<" basis"<<endl;
    }
    auto shell2bf=obs.shell2bf();
    // initial s_engine
     libint2::Engine s_engine(libint2::Operator::overlap,
                     obs.max_nprim(),
                     obs.max_l()
                     ); 
    const auto& buf = s_engine.results();
    int n_basis=obs.nbf();
    //std::cout<<"Number of basis is "<<n_basis<<endl;
    Matrix S = Matrix::Zero(n_basis, n_basis);
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            auto first=obs[i]; 
            auto second=obs[j]; 
            s_engine.compute(first ,second);
            // transfer into eigen buf
            auto n1=first.size();
            auto n2=second.size();
           // std::cout<<"N1 " <<n1<<" N2"<<n2<<std::endl;
            Eigen::Map<const Matrix> buf_mat(buf[0],n1,n2);
            // check block index using shell2bf
            int bind1=shell2bf[i];
            int bind2=shell2bf[j];
           // std::cout<<"Buffer is \n"<<buf_mat<<std::endl;
            S.block(bind1,bind2,n1,n2)=buf_mat;
        }
    }
    auto S_f=ao_overlap(s_engine,obs);
    std::cout<<"diff normal is "<<(S-S_f).norm()<<std::endl;
    // // test S correct.
    // // compute T correct
    // // step 1 build engine
    libint2::Engine t_engine(libint2::Operator::kinetic,  // will compute overlap ints
                obs.max_nprim(),    // max # of primitives in shells this engine will accept
                obs.max_l()         // max angular momentum of shells this engine will accept
               );
    const auto& buf_t = t_engine.results();
    //step 2 build final data matrix
    Matrix T = Matrix::Zero(n_basis, n_basis);
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            auto first=obs[i]; 
            auto second=obs[j]; 
            t_engine.compute(first ,second);   // heare need change
            // transfer into eigen buf
            auto n1=first.size();
            auto n2=second.size();
            //std::cout<<"N1 " <<n1<<" N2"<<n2<<std::endl;
            Eigen::Map<const Matrix> buf_mat(buf_t[0],n1,n2); // change buffer
            // check block index using shell2bf
            int bind1=shell2bf[i];
            int bind2=shell2bf[j];
            //std::cout<<"Buffer is \n"<<buf_mat<<std::endl;
            T.block(bind1,bind2,n1,n2)=buf_mat; // here need change
        }
    }
    auto T_f=ao_overlap(t_engine,obs);
    std::cout<<"diff normal is "<<(T-T_f).norm()<<std::endl;
    // // save T data
    // // const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    // ofstream file2("T.csv");
	// if (file2.is_open())
	// {
	// 	file2 << T.format(CSVFormat);
	// 	file2.close();
	// }   

    // // calculate V_nuc 
    // // step1 iniitalize engine
    libint2::Engine v_engine(libint2::Operator::nuclear	,  // will compute overlap ints
                obs.max_nprim(),    // max # of primitives in shells this engine will accept
                obs.max_l()         // max angular momentum of shells this engine will accept
               );
    // calculate nuclear potential 
    std::vector<std::pair<real_t,std::array<real_t,3>>>   q;
    std::pair<real_t,std::array<real_t,3>> nuclear_potential;
    for (const auto atom : atoms){
        std::array<real_t,3> coord;
         coord={atom.x,atom.y,atom.z};
       nuclear_potential=std::make_pair(
        static_cast<real_t> (atom.atomic_number),
        coord
       );
       q.push_back(nuclear_potential);
    };
    v_engine.set_params(q);
    const auto& buf_v = v_engine.results();
    // calculate 
    Matrix V = Matrix::Zero(n_basis, n_basis);
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            auto first=obs[i]; 
            auto second=obs[j]; 
            v_engine.compute(first ,second); // here need change 
            // transfer into eigen buf
            auto n1=first.size();
            auto n2=second.size();
            //std::cout<<"N1 " <<n1<<" N2"<<n2<<std::endl;
            Eigen::Map<const Matrix> buf_mat(buf_v[0],n1,n2); // here need change .
            // check block index using shell2bf
            int bind1=shell2bf[i];
            int bind2=shell2bf[j];
            //std::cout<<"Buffer is \n"<<buf_mat<<std::endl; // and here
            V.block(bind1,bind2,n1,n2)=buf_mat;
        }
    }
    auto V_f=ao_overlap(v_engine,obs);
    std::cout<<"diff normal is "<<(V-V_f).norm()<<std::endl;   
    // }
    // // save V data
    // // const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    // ofstream file3("V.csv");
	// if (file3.is_open())
	// {
	// 	file3 << V.format(CSVFormat);
	// 	file3.close();
	// } 
    // eri integral
    // initialize engine
    libint2::Engine eri_engine(libint2::Operator::coulomb	,  // will compute overlap ints
        obs.max_nprim(),    // max # of primitives in shells this engine will accept
        obs.max_l()         // max angular momentum of shells this engine will accept
        );
    const auto& buf_eri = eri_engine.results(); 
    // using xtensor to store eri results
    XTMatrix eri_tensor =xt::zeros<real_t>({n_basis, n_basis,n_basis,n_basis});
    // now fill the tensor
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            for(int k=0;k<n_shell;k++){
                for (int l=0;l<n_shell;l++){                
                    auto first=obs[i]; 
                    auto second=obs[j]; 
                    auto third=obs[k];
                    auto fourth=obs[l];
                    eri_engine.compute(first ,second,third,fourth); // here need change 
                    auto ints_shellset=buf_eri[0]; // location of the computed integrals
                    if (ints_shellset==nullptr)
                        continue; // nullprt retured if the entaire shell-set was screened out
                    // transfer into eigen buf
                    auto n1=first.size();
                    auto n2=second.size();
                    int n3=third.size();
                    int n4=fourth.size();
                     // here need change .
                    // check block index using shell2bf
                    int bind1=shell2bf[i];
                    int bind2=shell2bf[j];
                    int bind3=shell2bf[k];
                    int bind4=shell2bf[l];
                    // write to full tensor 
                    // this is not the most efficient way to write such a product,
                    // but is the only way possible with the current feature set
                    for (auto f1=0;f1!=n1;++f1){
                        for (auto f2=0; f2!=n2;++f2){
                            for(auto f3=0; f3!=n3;++f3){
                                for(auto f4=0;f4!=n4;++f4){
                                
                                //cout<<"  "<<bind1+f1<<" " <<bind2+f2<<" " <<bind3+f3 <<" "<<bind4+f4<<" "<<setprecision(15)<< ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4]<<endl;
                                        // print out results
                                eri_tensor(bind1+f1,bind2+f2,bind3+f3,bind4+f4)=ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4];
                                }
                            }

                        }
                    }
                    

                }
            }
        }
    }
    XTMatrix eri_tensor_from_function=ao_eri(eri_engine,obs);
    bool equal=xt::allclose(eri_tensor_from_function,eri_tensor);
    std::cout<<"Function is OK "<<equal<< "norm is "<< xt::norm_l2(eri_tensor_from_function-eri_tensor);
    // now do hartree fock
    Matrix H_zero= T+V; // h0 matrix
    Matrix F= Matrix::Zero(n_basis, n_basis); // fock matrix
    Matrix Coeff= Matrix::Zero(n_basis, n_basis); // coeff matrix

    // from coeff zero ,build f
    Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> ges;
    ges.compute(H_zero,S);
    auto epsilon=ges.eigenvalues();
    Coeff=ges.eigenvectors();
    real_t energy=0;
    real_t delta;
    // begin iteration
     do{
        // build D
        Matrix Density=Matrix::Zero(n_basis, n_basis);
        for (int i =0;i<n_basis;i++){
            for (int j=0;j<n_basis;j++){
                for (int k=0;k<ndocc;k++){
                    Density(i,j)+=Coeff(i,k)*Coeff(j,k);
                }
            }
        }
        // build J
        Matrix J=Matrix::Zero(n_basis, n_basis);
        for (int i =0;i<n_basis;i++){
            for (int j=0;j<n_basis;j++){
                for (int k=0;k<n_basis;k++){
                    for (int l=0;l<n_basis;l++){
                        J(i,j)+=eri_tensor(i,j,k,l)*Density(k,l);
                    }              
                }
            }
        }
        // build K
        Matrix K=Matrix::Zero(n_basis, n_basis);
        for (int i =0;i<n_basis;i++){
            for (int j=0;j<n_basis;j++){
                for (int k=0;k<n_basis;k++){
                    for (int l=0;l<n_basis;l++){
                        K(i,j)+=eri_tensor(i,k,j,l)*Density(k,l);
                    }              
                }
            }
        }
        F=H_zero+2*J-K;
        Eigen::GeneralizedSelfAdjointEigenSolver<Matrix> ges1;
        ges1.compute(F,S);
        auto epsilon1=ges1.eigenvalues();
        Coeff=ges1.eigenvectors();
        real_t old_energy=energy;
        energy=0;
        for (int i=0;i<ndocc;i++){
            energy+=epsilon1(i);
            for (int j=0;j<n_basis;j++){
                for (int k=0;k<n_basis;k++){
                    energy+=Coeff(j,i)*H_zero(j,k)*Coeff(k,i);    
                }
            }
        }
        std::cout<<"energy is "<<std::setprecision(15)<<energy<<"\t";
        delta=std::abs(energy-old_energy);
        std::cout <<"delta is "<<delta<<std::endl;
        if (delta<1e-9){
            // save mo coeff
        const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
        ofstream file("C_mo.csv");
        if (file.is_open())
        {
            file << Coeff.format(CSVFormat);
            file.close();
        }   
        }
    }while(delta>1e-9);
    return 0;
}
Matrix  ao_overlap(libint2::Engine s_engine,libint2::BasisSet obs){
    int n_basis=obs.nbf();
    int n_shell=obs.size();
    auto shell2bf=obs.shell2bf();
    // initial s_engine
    const auto& buf = s_engine.results();
    Matrix S = Matrix::Zero(n_basis, n_basis);
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            auto first=obs[i]; 
            auto second=obs[j]; 
            s_engine.compute(first ,second);
            // transfer into eigen buf
            auto n1=first.size();
            auto n2=second.size();
           // std::cout<<"N1 " <<n1<<" N2"<<n2<<std::endl;
            Eigen::Map<const Matrix> buf_mat(buf[0],n1,n2);
            // check block index using shell2bf
            int bind1=shell2bf[i];
            int bind2=shell2bf[j];
           // std::cout<<"Buffer is \n"<<buf_mat<<std::endl;
            S.block(bind1,bind2,n1,n2)=buf_mat;
        }
    }
    return S;
}
XTMatrix ao_eri(libint2::Engine eri_engine,libint2::BasisSet obs){
    int n_basis=obs.nbf();
    int n_shell=obs.size();
    libint2::Shell first,second,third,fourth; // four shell
    int n1,n2,n3,n4;                          // number of basis function in each shell.
    auto shell2bf=obs.shell2bf();              //  index of first bs function in total basis function.
    XTMatrix eri_tensor =xt::zeros<real_t>({n_basis, n_basis,n_basis,n_basis});
    const auto &buf_eri=eri_engine.results();
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            for(int k=0;k<n_shell;k++){
                for (int l=0;l<n_shell;l++){                
                    first=obs[i]; 
                    second=obs[j]; 
                    third=obs[k];
                    fourth=obs[l];
                    eri_engine.compute(first ,second,third,fourth); // here need change 
                    auto ints_shellset=buf_eri[0]; // location of the computed integrals
                    if (ints_shellset==nullptr)
                        continue; // nullprt retured if the entaire shell-set was screened out
                    // transfer into eigen buf
                    n1=first.size();
                    n2=second.size();
                    n3=third.size();
                    n4=fourth.size();
                     // here need change .
                    // check block index using shell2bf
                    int bind1=shell2bf[i];
                    int bind2=shell2bf[j];
                    int bind3=shell2bf[k];
                    int bind4=shell2bf[l];
                    // write to full tensor 
                    // this is not the most efficient way to write such a product,
                    // but is the only way possible with the current feature set
                    for (auto f1=0;f1!=n1;++f1){
                        for (auto f2=0; f2!=n2;++f2){
                            for(auto f3=0; f3!=n3;++f3){
                                for(auto f4=0;f4!=n4;++f4){
                                
                                //cout<<"  "<<bind1+f1<<" " <<bind2+f2<<" " <<bind3+f3 <<" "<<bind4+f4<<" "<<setprecision(15)<< ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4]<<endl;
                                        // print out results
                                eri_tensor(bind1+f1,bind2+f2,bind3+f3,bind4+f4)=ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4];
                                }
                            }

                        }
                    }
                    

                }
            }
        }
    }
    return   eri_tensor;
}