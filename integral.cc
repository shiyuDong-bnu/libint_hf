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
#include <xtensor-blas/xlinalg.hpp>
#include <stdexcept>
using namespace std;

using real_t = libint2::scalar_type;
typedef Eigen::Matrix<real_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>
        Matrix;
 using XTMatrix = xt::xarray<real_t, xt::layout_type::row_major>;
Matrix S_integral(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2
);
Matrix S_integral1(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2
);
Matrix T_integral1(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2
);
Matrix V_integral1(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
const vector<libint2::Atom>
);
Matrix one_body(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
    libint2::Engine engine
);  
XTMatrix eri_integral(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
    libint2::BasisSet obs3,
    libint2::BasisSet obs4
);
XTMatrix yukawa_integral(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
    libint2::BasisSet obs3,
    libint2::BasisSet obs4,
    double gamma
);
XTMatrix stg_integral(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
    libint2::BasisSet obs3,
    libint2::BasisSet obs4,
    double gamma
);
XTMatrix two_body(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
    libint2::BasisSet obs3,
    libint2::BasisSet obs4,
    libint2::Engine engine
);
void Save_Eigen_Matrix(std::string fname,Matrix matrix);
int main(int argc, char* argv[]){
    libint2::initialize();
    // reading input molecule
    string xyzfile ="Ne.xyz";
    ifstream inputfile(xyzfile);
    vector<libint2::Atom> atoms= libint2::read_dotxyz(inputfile);
    //cout <<"Number of atoms is "<<atoms.size()<<endl;
    auto nelectron = 0;
    for (auto i = 0; i < atoms.size(); ++i) nelectron += atoms[i].atomic_number;
    const auto ndocc = nelectron / 2;
    auto enuc = 0.0;
    for (auto i = 0; i < atoms.size(); i++)
      for (auto j = i + 1; j < atoms.size(); j++) {
        auto xij = atoms[i].x - atoms[j].x;
        auto yij = atoms[i].y - atoms[j].y;
        auto zij = atoms[i].z - atoms[j].z;
        auto r2 = xij * xij + yij * yij + zij * zij;
        auto r = sqrt(r2);
        enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
      }
    cout << "\tNuclear repulsion energy = " << enuc << endl;
    // build basis
    libint2::BasisSet obs("aug-cc-pVDZ",atoms);
    libint2::BasisSet cabs("aug-cc-pVDZ-optri",atoms);
    // Get information of basis set
    int max_l=obs.max_l();
    int max_nprim= obs.max_nprim();
    auto shell2bf=obs.shell2bf();
    auto bf1=shell2bf[0];
    //std::cout<<"max l is "<<max_l<<endl;
    //std::cout<<"max nprim is"<<max_nprim<<endl;
    int n_shell=obs.size();
    std::cout<<"The basis set has "<<n_shell<<" shells"<<endl;
    for (int i=0;i<n_shell;++i){
        int n_basis=obs[i].size();
        std::cout<<"shell "<< i<<" has "<<n_basis<<" basis"<<endl;
    }
    // initial s engine
    libint2::Engine s_engine(libint2::Operator::overlap,
                     obs.max_nprim(),
                     obs.max_l()
                     ); 
    // calculate only one shell at a time
    libint2::Shell first=obs[3]; // p shell 
    libint2::Shell second=obs[4]; // d shell
    // get the address of result beforehand
    const auto& buf = s_engine.results();
    s_engine.compute(first ,second);
    // std::cout<<"S compute Done"<<endl;
    // need to get the results. 
    // let us read it first to get a sence of buf
    int n1=first.size();
    int n2=second.size();
    // std::cout <<"First shell have "<<n1<<" contr basis"<<endl;
    // std::cout <<"Second shell have "<<n2<<" contr basis"<<endl;
    // for general we can also get the same result in shell to basis function?
    //std::cout<<"shell2bf type is vector ->"<<typeid(shell2bf).name()<<endl;
    n1=shell2bf[3];
    n2=shell2bf[3];
    // this is in total 
    //std::cout <<"First shell have basis index in total "<<n1<<" contr basis"<<endl;
    //std::cout <<"Second shell first basis index in total  "<<n2<<" contr basis"<<endl;
    //
    auto ints_shellset=buf[0]; // location of computed integral
    if (ints_shellset==nullptr)
        std::cout<<"This is being screened out"; // nullprt retured if the entaire shell-set was screened out
    else{
        // try transfer this into eigen matrix
        Eigen::Map<const Matrix> buf_mat(buf[0],n1,n2);
        //std::cout<<buf_mat<<std::endl;
    }
    // calculate all shell in a for loop ;no symmetry yet.
    // Need initialize a big eigen matrix to store result
    int n_basis=obs.nbf();
    //std::cout<<"Number of basis is "<<n_basis<<endl;
    Matrix S = Matrix::Zero(n_basis, n_basis);
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            first=obs[i]; 
            second=obs[j]; 
            s_engine.compute(first ,second);
            // transfer into eigen buf
            n1=first.size();
            n2=second.size();
           // std::cout<<"N1 " <<n1<<" N2"<<n2<<std::endl;
            Eigen::Map<const Matrix> buf_mat(buf[0],n1,n2);
            // check block index using shell2bf
            int bind1=shell2bf[i];
            int bind2=shell2bf[j];
           // std::cout<<"Buffer is \n"<<buf_mat<<std::endl;
            S.block(bind1,bind2,n1,n2)=buf_mat;
        }
    }
    //save S matrix
    auto S_f=S_integral(obs,obs);
    auto S_f1=S_integral1(obs,obs);
    std::cout<<"test functionS "<<(S_f-S).norm()<<std::endl;
    std::cout<<"test functionS1 "<<(S_f1-S).norm()<<std::endl;
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    ofstream file("S.csv");
	if (file.is_open())
	{
		file << S.format(CSVFormat);
		file.close();
	}

    // test S correct.
    // compute T correct
    // step 1 build engine
    libint2::Engine t_engine(libint2::Operator::kinetic,  // will compute overlap ints
                obs.max_nprim(),    // max # of primitives in shells this engine will accept
                obs.max_l()         // max angular momentum of shells this engine will accept
               );
    const auto& buf_t = t_engine.results();
    //step 2 build final data matrix
    Matrix T = Matrix::Zero(n_basis, n_basis);
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            first=obs[i]; 
            second=obs[j]; 
            t_engine.compute(first ,second);   // heare need change
            // transfer into eigen buf
            n1=first.size();
            n2=second.size();
            //std::cout<<"N1 " <<n1<<" N2"<<n2<<std::endl;
            Eigen::Map<const Matrix> buf_mat(buf_t[0],n1,n2); // change buffer
            // check block index using shell2bf
            int bind1=shell2bf[i];
            int bind2=shell2bf[j];
            //std::cout<<"Buffer is \n"<<buf_mat<<std::endl;
            T.block(bind1,bind2,n1,n2)=buf_mat; // here need change
        }
    }
    // test T fucntion
    Matrix T_f=T_integral1(obs,obs);
    std::cout<<"test functionT1 "<<(T_f-T).norm()<<std::endl;
    // save T data
    // const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    ofstream file2("T.csv");
	if (file2.is_open())
	{
		file2 << T.format(CSVFormat);
		file2.close();
	}
    Matrix T_obs_cabs=T_integral1(obs,cabs);
    Save_Eigen_Matrix("T_obs_cabs.csv",T_obs_cabs);
    // calculate V_nuc 
    // step1 iniitalize engine
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
            first=obs[i]; 
            second=obs[j]; 
            v_engine.compute(first ,second); // here need change 
            // transfer into eigen buf
            n1=first.size();
            n2=second.size();
            //std::cout<<"N1 " <<n1<<" N2"<<n2<<std::endl;
            Eigen::Map<const Matrix> buf_mat(buf_v[0],n1,n2); // here need change .
            // check block index using shell2bf
            int bind1=shell2bf[i];
            int bind2=shell2bf[j];
            //std::cout<<"Buffer is \n"<<buf_mat<<std::endl; // and here
            V.block(bind1,bind2,n1,n2)=buf_mat;
        }
    }
    // test v function
    Matrix V_f=V_integral1(obs,obs,atoms);
    std::cout<<"test functionV1 "<<(V_f-V).norm()<<std::endl;
    // save V data
    // const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    ofstream file3("V.csv");
	if (file3.is_open())
	{
		file3 << V.format(CSVFormat);
		file3.close();
	} 
    Matrix V_obs_cabs=V_integral1(obs,cabs,atoms);
    Save_Eigen_Matrix("V_obs_cabs.csv",V_obs_cabs);
    // eri integral
    // initialize engine
    libint2::Engine eri_engine(libint2::Operator::coulomb	,  // will compute overlap ints
        obs.max_nprim(),    // max # of primitives in shells this engine will accept
        obs.max_l() , // max angular momentum of shells this engine will accept
        0      
        );
    const auto& buf_eri = eri_engine.results(); 
    // using xtensor to store eri results
    XTMatrix eri_tensor =xt::zeros<real_t>({n_basis, n_basis,n_basis,n_basis});
    // now fill the tensor
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            for(int k=0;k<n_shell;k++){
                for (int l=0;l<n_shell;l++){                
                    first=obs[i]; 
                    second=obs[j]; 
                    auto third=obs[k];
                    auto fourth=obs[l];
                    eri_engine.compute(first ,second,third,fourth); // here need change 
                    auto ints_shellset=buf_eri[0]; // location of the computed integrals
                    if (ints_shellset==nullptr)
                        continue; // nullprt retured if the entaire shell-set was screened out
                    // transfer into eigen buf
                    n1=first.size();
                    n2=second.size();
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
    // test eri
    XTMatrix eri_f=eri_integral(obs,obs,obs,obs);
    xt::dump_npy("erif.npy",eri_f);
    // save eri to npy file
    xt::dump_npy("eri.npy",eri_tensor);
    // Now save gr_ijxy in atomic basis
    double gamma=1.5;
    libint2::Engine yukawa_engine(libint2::Operator::yukawa	,  // will compute overlap ints
        obs.max_nprim(),    // max # of primitives in shells this engine will accept
        obs.max_l()         // max angular momentum of shells this engine will accept
        );
    yukawa_engine.set_params(gamma);
    const auto& buf_yukawa = yukawa_engine.results(); 
    // using xtensor to store eri results
    XTMatrix yukawa_tensor =xt::zeros<real_t>({n_basis, n_basis,n_basis,n_basis});
    // now fill the tensor
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            for(int k=0;k<n_shell;k++){
                for (int l=0;l<n_shell;l++){                
                    first=obs[i]; 
                    second=obs[j]; 
                    auto third=obs[k];
                    auto fourth=obs[l];
                    yukawa_engine.compute(first ,second,third,fourth); // here need change 
                    auto ints_shellset=buf_yukawa[0]; // location of the computed integrals
                    if (ints_shellset==nullptr)
                        continue; // nullprt retured if the entaire shell-set was screened out
                    // transfer into eigen buf
                    n1=first.size();
                    n2=second.size();
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
                                yukawa_tensor(bind1+f1,bind2+f2,bind3+f3,bind4+f4)=ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4];
                                }
                            }

                        }
                    }
                    

                }
            }
        }
    }
    xt::dump_npy("yukawa.npy",yukawa_tensor);   
    // test yukawa 
    xt::dump_npy("yukawa_f.npy",yukawa_integral(obs,obs,obs,obs,gamma));
    

    // eri integral
    // initialize engine
    int hybrid_max_nprim=obs.max_nprim();
    int hybrid_max_l=obs.max_l();
    if (obs.max_nprim()<cabs.max_nprim()){
        hybrid_max_nprim=cabs.max_nprim();
    }
    if (obs.max_l()<cabs.max_l()){
        hybrid_max_l=cabs.max_l();
    }
    libint2::Engine hyb_eri_engine(libint2::Operator::coulomb	,  // will compute overlap ints
        hybrid_max_nprim,    // max # of primitives in shells this engine will accept
        hybrid_max_l        // max angular momentum of shells this engine will accept
        );
    const auto& buf_hyb_eri = hyb_eri_engine.results();
    int n_cabs=cabs.nbf(); 
    int n_cabs_shell=cabs.size();
    auto cabs_shell2bf=cabs.shell2bf();
    std::cout<<"Here ?"<<std::endl;
    std::cout<<n_cabs;
    // using xtensor to store eri results
    XTMatrix hyb_eri_tensor =xt::zeros<real_t>({n_basis, n_basis,n_basis,n_cabs});
    // now fill the tensor
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            for(int k=0;k<n_shell;k++){
                for (int l=0;l<n_cabs_shell;l++){                
                    first=obs[i]; 
                    second=obs[j]; 
                    auto third=obs[k];
                    auto fourth=cabs[l];
                    hyb_eri_engine.compute(first ,second,third,fourth); // here need change 
                    auto ints_shellset=buf_hyb_eri[0]; // location of the computed integrals
                    if (ints_shellset==nullptr)
                        {   std::cout<<"null"<<std::endl;
                            continue;} // nullprt retured if the entaire shell-set was screened out
                    // transfer into eigen buf
                    n1=first.size();
                    n2=second.size();
                    int n3=third.size();
                    int n4=fourth.size();
                     // here need change .
                    // check block index using shell2bf
                    int bind1=shell2bf[i];
                    int bind2=shell2bf[j];
                    int bind3=shell2bf[k];
                    int bind4=cabs_shell2bf[l];
                    // write to full tensor 
                    // this is not the most efficient way to write such a product,
                    // but is the only way possible with the current feature set
                    for (auto f1=0;f1!=n1;++f1){
                        for (auto f2=0; f2!=n2;++f2){
                            for(auto f3=0; f3!=n3;++f3){
                                for(auto f4=0;f4!=n4;++f4){
                                
                                //cout<<"  "<<bind1+f1<<" " <<bind2+f2<<" " <<bind3+f3 <<" "<<bind4+f4<<" "<<setprecision(15)<< ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4]<<endl;
                                        // print out results
                                hyb_eri_tensor(bind1+f1,bind2+f2,bind3+f3,bind4+f4)=ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4];
                                }
                            }

                        }
                    }
                    

                }
            }
        }
    }
    // save eri to npy file
    xt::dump_npy("hyb_eri_tensor.npy",hyb_eri_tensor);
    xt::dump_npy("hyb_eri_tensorf.npy",eri_integral(obs,obs,obs,cabs));
    xt::dump_npy("cooc_eri_tensorf.npy",eri_integral(cabs,obs,obs,cabs));
    // stg
    libint2::Engine hyb_std_engine(libint2::Operator::stg	,  // will compute overlap ints
        hybrid_max_nprim,    // max # of primitives in shells this engine will accept
        hybrid_max_l        // max angular momentum of shells this engine will accept
        );
    hyb_std_engine.set_params(gamma);
    const auto& buf_hyb_stg = hyb_std_engine.results();
    // using xtensor to store eri results
    XTMatrix hyb_stg_tensor =xt::zeros<real_t>({n_basis, n_basis,n_basis,n_cabs});
    // now fill the tensor
    std::cout<<"calcualting hyb_stg_tensor"<<std::endl;
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            for(int k=0;k<n_shell;k++){
                for (int l=0;l<n_cabs_shell;l++){                
                    first=obs[i]; 
                    second=obs[j]; 
                    auto third=obs[k];
                    auto fourth=cabs[l];
                    hyb_std_engine.compute(first ,second,third,fourth); // here need change 
                    auto ints_shellset=buf_hyb_stg[0]; // location of the computed integrals
                    if (ints_shellset==nullptr)
                        {   std::cout<<"null"<<std::endl;
                            continue;} // nullprt retured if the entaire shell-set was screened out
                    // transfer into eigen buf
                    n1=first.size();
                    n2=second.size();
                    int n3=third.size();
                    int n4=fourth.size();
                     // here need change .
                    // check block index using shell2bf
                    int bind1=shell2bf[i];
                    int bind2=shell2bf[j];
                    int bind3=shell2bf[k];
                    int bind4=cabs_shell2bf[l];
                    // write to full tensor 
                    // this is not the most efficient way to write such a product,
                    // but is the only way possible with the current feature set
                    for (auto f1=0;f1!=n1;++f1){
                        for (auto f2=0; f2!=n2;++f2){
                            for(auto f3=0; f3!=n3;++f3){
                                for(auto f4=0;f4!=n4;++f4){
                                
                                //cout<<"  "<<bind1+f1<<" " <<bind2+f2<<" " <<bind3+f3 <<" "<<bind4+f4<<" "<<setprecision(15)<< ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4]<<endl;
                                        // print out results
                                hyb_stg_tensor(bind1+f1,bind2+f2,bind3+f3,bind4+f4)=ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4];
                                }
                            }

                        }
                    }
                    

                }
            }
        }
    }
    // save eri to npy file
    xt::dump_npy("hyb_stg_tensor.npy",hyb_stg_tensor);
    xt::dump_npy("hyb_stg_tensorf.npy",stg_integral(obs,obs,obs,cabs,gamma));
    xt::dump_npy("hyb_square_stg_tensorf.npy",stg_integral(obs,obs,obs,cabs,2.0*gamma));



    libint2::Engine stg_engine(libint2::Operator::stg	,  // will compute overlap ints
        max_nprim,    // max # of primitives in shells this engine will accept
        max_l        // max angular momentum of shells this engine will accept
        );
    stg_engine.set_params(gamma);
    const auto& buf_stg = stg_engine.results();
    XTMatrix stg_tensor =xt::zeros<real_t>({n_basis, n_basis,n_basis,n_basis});
    // now fill the tensor
    std::cout<<"calcualting stg_tensor"<<std::endl;
    for (int i=0;i<n_shell;i++){
        for (int j=0;j<n_shell;j++){
            for(int k=0;k<n_shell;k++){
                for (int l=0;l<n_shell;l++){                
                    first=obs[i]; 
                    second=obs[j]; 
                    auto third=obs[k];
                    auto fourth=obs[l];
                    stg_engine.compute(first ,second,third,fourth); // here need change 
                    auto ints_shellset=buf_stg[0]; // location of the computed integrals
                    if (ints_shellset==nullptr)
                        {   std::cout<<"null"<<std::endl;
                            continue;} // nullprt retured if the entaire shell-set was screened out
                    // transfer into eigen buf
                    n1=first.size();
                    n2=second.size();
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
                                stg_tensor(bind1+f1,bind2+f2,bind3+f3,bind4+f4)=ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4];
                                }
                            }

                        }
                    }
                    

                }
            }
        }
    }
    // save eri to npy file
    xt::dump_npy("stg_tensor.npy",stg_tensor);
    xt::dump_npy("stg_tensorf.npy",stg_integral(obs,obs,obs,obs,gamma));
    xt::dump_npy("stg_square_tensorf.npy",stg_integral(obs,obs,obs,obs,2.0*gamma));
    return 0;
}
Matrix S_integral(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2
){
    const int n_basis1=obs1.nbf();
    const int n_basis2=obs2.nbf();
    const int max_nprim1=obs1.max_nprim();
    const int max_nprim2=obs2.max_nprim();
    const int max_l1=obs1.max_l();
    const int max_l2=obs2.max_l();
    const int n_shell1=obs1.size();
    const int n_shell2=obs2.size();
    auto shell2bf1=obs1.shell2bf();
    auto shell2bf2=obs2.shell2bf();
    // build S engine
    int max_nprim=(max_nprim1>max_nprim2)? max_nprim1:max_nprim2;
    int max_l=(max_l1>max_l2)? max_l1:max_l2;
    libint2::Engine s_engine(libint2::Operator::overlap,
                     max_nprim,
                     max_l
                     ); 
    const auto& buf=s_engine.results();
    // now fill the tensor
    std::cout<<"calcualting S_tensor"<<std::endl;
    Matrix S = Matrix::Zero(n_basis1, n_basis2);
    for (int i=0;i<n_shell1;i++){
        for (int j=0;j<n_shell2;j++){
            libint2::Shell first=obs1[i]; 
            libint2::Shell  second=obs2[j]; 
            s_engine.compute(first ,second);
            // transfer into eigen buf
            int n1=first.size();
            int n2=second.size();
           // std::cout<<"N1 " <<n1<<" N2"<<n2<<std::endl;
            Eigen::Map<const Matrix> buf_mat(buf[0],n1,n2);
            // check block index using shell2bf
            int bind1=shell2bf1[i];
            int bind2=shell2bf2[j];
           // std::cout<<"Buffer is \n"<<buf_mat<<std::endl;
            S.block(bind1,bind2,n1,n2)=buf_mat;
        }
    }
    return S;
}
Matrix S_integral1(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2
){
    const int n_basis1=obs1.nbf();
    const int n_basis2=obs2.nbf();
    const int max_nprim1=obs1.max_nprim();
    const int max_nprim2=obs2.max_nprim();
    const int max_l1=obs1.max_l();
    const int max_l2=obs2.max_l();
    const int n_shell1=obs1.size();
    const int n_shell2=obs2.size();
    auto shell2bf1=obs1.shell2bf();
    auto shell2bf2=obs2.shell2bf();
    int max_nprim=(max_nprim1>max_nprim2)? max_nprim1:max_nprim2;
    int max_l=(max_l1>max_l2)? max_l1:max_l2;
    libint2::Engine engine(libint2::Operator ::overlap,
                     max_nprim,
                     max_l
                     ); 

  return one_body(obs1,obs2, engine);
}
Matrix T_integral1(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2
){
    const int n_basis1=obs1.nbf();
    const int n_basis2=obs2.nbf();
    const int max_nprim1=obs1.max_nprim();
    const int max_nprim2=obs2.max_nprim();
    const int max_l1=obs1.max_l();
    const int max_l2=obs2.max_l();
    const int n_shell1=obs1.size();
    const int n_shell2=obs2.size();
    auto shell2bf1=obs1.shell2bf();
    auto shell2bf2=obs2.shell2bf();
    int max_nprim=(max_nprim1>max_nprim2)? max_nprim1:max_nprim2;
    int max_l=(max_l1>max_l2)? max_l1:max_l2;
    libint2::Engine engine(libint2::Operator ::kinetic,
                     max_nprim,
                     max_l
                     ); 

  return one_body(obs1,obs2, engine);
}
Matrix V_integral1(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
     const vector<libint2::Atom>atoms
){
    const int n_basis1=obs1.nbf();
    const int n_basis2=obs2.nbf();
    const int max_nprim1=obs1.max_nprim();
    const int max_nprim2=obs2.max_nprim();
    const int max_l1=obs1.max_l();
    const int max_l2=obs2.max_l();
    const int n_shell1=obs1.size();
    const int n_shell2=obs2.size();
    auto shell2bf1=obs1.shell2bf();
    auto shell2bf2=obs2.shell2bf();
    int max_nprim=(max_nprim1>max_nprim2)? max_nprim1:max_nprim2;
    int max_l=(max_l1>max_l2)? max_l1:max_l2;
    libint2::Engine engine(libint2::Operator ::nuclear,
                     max_nprim,
                     max_l
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
    engine.set_params(q);
  return one_body(obs1,obs2, engine);
}
Matrix one_body(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
    libint2::Engine engine
){
    const int n_basis1=obs1.nbf();
    const int n_basis2=obs2.nbf();
    const int max_nprim1=obs1.max_nprim();
    const int max_nprim2=obs2.max_nprim();
    const int max_l1=obs1.max_l();
    const int max_l2=obs2.max_l();
    const int n_shell1=obs1.size();
    const int n_shell2=obs2.size();
    auto shell2bf1=obs1.shell2bf();
    auto shell2bf2=obs2.shell2bf();


    const auto& buf=engine.results();
    // now fill the tensor
    Matrix Result = Matrix::Zero(n_basis1, n_basis2);
    for (int i=0;i<n_shell1;i++){
        for (int j=0;j<n_shell2;j++){
            libint2::Shell first=obs1[i]; 
            libint2::Shell  second=obs2[j]; 
            engine.compute(first ,second);
            // transfer into eigen buf
            int n1=first.size();
            int n2=second.size();
           // std::cout<<"N1 " <<n1<<" N2"<<n2<<std::endl;
            Eigen::Map<const Matrix> buf_mat(buf[0],n1,n2);
            // check block index using shell2bf
            int bind1=shell2bf1[i];
            int bind2=shell2bf2[j];
           // std::cout<<"Buffer is \n"<<buf_mat<<std::endl;
            Result.block(bind1,bind2,n1,n2)=buf_mat;
        }
    }
    return Result;
};
XTMatrix eri_integral(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
    libint2::BasisSet obs3,
    libint2::BasisSet obs4
){
   

    const int max_nprim1=obs1.max_nprim();
    const int max_nprim2=obs2.max_nprim();
    const int max_nprim3=obs3.max_nprim();
    const int max_nprim4=obs4.max_nprim();

    const int max_l1=obs1.max_l();
    const int max_l2=obs2.max_l();
    const int max_l3=obs3.max_l();
    const int max_l4=obs4.max_l();

    std::vector<int> max_ls={max_l1,max_l2,max_l3,max_l4};
    std::vector<int> max_nprims={max_nprim1,max_nprim2,max_nprim3,max_nprim4};
    int max_l=*std::max_element(max_ls.begin(),max_ls.end());
    int max_nprim=*std::max_element(max_nprims.begin(),max_nprims.end());

    libint2::Engine eri_engine( libint2::Operator::coulomb	,  // will compute overlap ints
    max_nprim,    // max # of primitives in shells this engine will accept
    max_l , // max angular momentum of shells this engine will accept
    0  
    );
    return two_body(obs1,obs2,obs3,obs4,eri_engine);
};
XTMatrix yukawa_integral(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
    libint2::BasisSet obs3,
    libint2::BasisSet obs4,
    double gamma
){
   

    const int max_nprim1=obs1.max_nprim();
    const int max_nprim2=obs2.max_nprim();
    const int max_nprim3=obs3.max_nprim();
    const int max_nprim4=obs4.max_nprim();
    const int max_l1=obs1.max_l();
    const int max_l2=obs2.max_l();
    const int max_l3=obs3.max_l();
    const int max_l4=obs4.max_l();
    std::vector<int> max_ls={max_l1,max_l2,max_l3,max_l4};
    std::vector<int> max_nprims={max_nprim1,max_nprim2,max_nprim3,max_nprim4};
    int max_l=*std::max_element(max_ls.begin(),max_ls.end());
    int max_nprim=*std::max_element(max_nprims.begin(),max_nprims.end());
    libint2::Engine yukawa_engine(libint2::Operator::yukawa	,  // will compute overlap ints
        max_nprim,    // max # of primitives in shells this engine will accept
        max_l         // max angular momentum of shells this engine will accept
        );
    yukawa_engine.set_params(gamma);
    return two_body(obs1,obs2,obs3,obs4,yukawa_engine);
};
XTMatrix stg_integral(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
    libint2::BasisSet obs3,
    libint2::BasisSet obs4,
    double gamma
){
   

    const int max_nprim1=obs1.max_nprim();
    const int max_nprim2=obs2.max_nprim();
    const int max_nprim3=obs3.max_nprim();
    const int max_nprim4=obs4.max_nprim();
    const int max_l1=obs1.max_l();
    const int max_l2=obs2.max_l();
    const int max_l3=obs3.max_l();
    const int max_l4=obs4.max_l();
    std::vector<int> max_ls={max_l1,max_l2,max_l3,max_l4};
    std::vector<int> max_nprims={max_nprim1,max_nprim2,max_nprim3,max_nprim4};
    int max_l=*std::max_element(max_ls.begin(),max_ls.end());
    int max_nprim=*std::max_element(max_nprims.begin(),max_nprims.end());
    libint2::Engine stg_engine(libint2::Operator::stg	,  // will compute overlap ints
        max_nprim,    // max # of primitives in shells this engine will accept
        max_l         // max angular momentum of shells this engine will accept
        );
    stg_engine.set_params(gamma);
    return two_body(obs1,obs2,obs3,obs4,stg_engine);
};
XTMatrix two_body(
    libint2::BasisSet obs1,
    libint2::BasisSet obs2,
    libint2::BasisSet obs3,
    libint2::BasisSet obs4,
    libint2::Engine engine
){
    
    const int n_basis1=obs1.nbf();
    const int n_basis2=obs2.nbf();
    const int n_basis3=obs3.nbf();
    const int n_basis4=obs4.nbf();

    const int n_shell1=obs1.size();
    const int n_shell2=obs2.size();
    const int n_shell3=obs3.size();
    const int n_shell4=obs4.size();

    auto shell2bf1=obs1.shell2bf();
    auto shell2bf2=obs2.shell2bf();
    auto shell2bf3=obs3.shell2bf();
    auto shell2bf4=obs4.shell2bf();

    XTMatrix result =xt::zeros<real_t>({n_basis1, n_basis2,n_basis3,n_basis4});

    const auto& buf=engine.results();
    // now fill the tensor
    for (int i=0;i<n_shell1;i++){
        for (int j=0;j<n_shell2;j++){
            for(int k=0;k<n_shell3;k++){
                for (int l=0;l<n_shell4;l++){                
                    libint2::Shell first=obs1[i]; 
                    libint2::Shell second=obs2[j]; 
                    libint2::Shell third=obs3[k];
                    libint2::Shell fourth=obs4[l];
                    engine.compute(first ,second,third,fourth); // here need change 
                    auto ints_shellset=buf[0]; // location of the computed integrals
                    if (ints_shellset==nullptr)
                        continue; // nullprt retured if the entaire shell-set was screened out
                    // transfer into eigen buf
                    int n1=first.size();
                    int n2=second.size();
                    int n3=third.size();
                    int n4=fourth.size();
                     // here need change .
                    // check block index using shell2bf
                    int bind1=shell2bf1[i];
                    int bind2=shell2bf2[j];
                    int bind3=shell2bf3[k];
                    int bind4=shell2bf4[l];
                    // write to full tensor 
                    // this is not the most efficient way to write such a product,
                    // but is the only way possible with the current feature set
                    for (auto f1=0;f1!=n1;++f1){
                        for (auto f2=0; f2!=n2;++f2){
                            for(auto f3=0; f3!=n3;++f3){
                                for(auto f4=0;f4!=n4;++f4){
                                
                                //cout<<"  "<<bind1+f1<<" " <<bind2+f2<<" " <<bind3+f3 <<" "<<bind4+f4<<" "<<setprecision(15)<< ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4]<<endl;
                                        // print out results
                                result(bind1+f1,bind2+f2,bind3+f3,bind4+f4)=ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4];
                                }
                            }

                        }
                    }
                    

                }
            }
        }
    }  
    return   result;
};
void Save_Eigen_Matrix(std::string fname,Matrix matrix){
    std::ofstream myfile;
    const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
    myfile.open (fname);
    myfile<<matrix.format(CSVFormat);
    myfile.close();
}