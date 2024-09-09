#include <string>
#include <vector>
#include <libint2.hpp>
#include <algorithm>
#include <iomanip>
#include<iostream>
#include <typeinfo>
#include <Eigen/Dense>
using namespace std;

using real_t = libint2::scalar_type;
typedef Eigen::Matrix<real_t,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>
        Matrix;
int main(int argc, char* argv[]){
    libint2::initialize();
    // reading input molecule
    string xyzfile ="Ne.xyz";
    ifstream inputfile(xyzfile);
    vector<libint2::Atom> atoms= libint2::read_dotxyz(inputfile);
    cout <<"Number of atoms is "<<atoms.size()<<endl;
    // build basis
    libint2::BasisSet obs("aug-cc-pVDZ",atoms);
    // Get information of basis set
    int max_l=obs.max_l();
    int max_nprim= obs.max_nprim();
    auto shell2bf=obs.shell2bf();
    auto bf1=shell2bf[0];
    std::cout<<"max l is "<<max_l<<endl;
    std::cout<<"max nprim is"<<max_nprim<<endl;
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
    std::cout<<"S compute Done"<<endl;
    // need to get the results. 
    // let us read it first to get a sence of buf
    int n1=first.size();
    int n2=second.size();
    std::cout <<"First shell have "<<n1<<" contr basis"<<endl;
    std::cout <<"Second shell have "<<n2<<" contr basis"<<endl;
    // for general we can also get the same result in shell to basis function?
    std::cout<<"shell2bf type is vector ->"<<typeid(shell2bf).name()<<endl;
    n1=shell2bf[3];
    n2=shell2bf[3];
    // this is in total 
    std::cout <<"First shell have basis index in total "<<n1<<" contr basis"<<endl;
    std::cout <<"Second shell first basis index in total  "<<n2<<" contr basis"<<endl;
    //
    auto ints_shellset=buf[0]; // location of computed integral
    if (ints_shellset==nullptr)
        std::cout<<"This is being screened out"; // nullprt retured if the entaire shell-set was screened out
    else{
        // print out the matrix.
        for (int i =0;i<n1;i++){
            for (int j=0;j<n2;j++){
                std::cout<<i<<" "<<j<<" "<<setprecision(15)<< ints_shellset[i*n2+j]<<endl;
            }
        }
    }
    // try transfer this into eigen matrix
    // Matrix result(n1, n2);
    // Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
    // result.block( n1, n2) = buf_mat;

    // libint2::finalize();
    /*
    Need revision to use eigen effectivly.
    */
    return 0;
}