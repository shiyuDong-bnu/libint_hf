#include <string>
#include <vector>
#include <libint2.hpp>
#include <algorithm>
#include <iomanip>
using namespace std;

int main(int argc, char* argv[]){
    libint2::initialize();
    //
    string xyzfile ="Ne.xyz";
    ifstream inputfile(xyzfile);
    vector<libint2::Atom> atoms= libint2::read_dotxyz(inputfile);
    libint2::BasisSet obs("aug-cc-pVDZ",atoms);
    //std::copy(begin(obs),end(obs),
   // [](const libint2::Shell & a ,const libint2::Shell & b){
    //    return a.contr[0].l < b.contr[0].l;
   // });
   //
    libint2::Engine s_engine(libint2::Operator::overlap,
                     obs.max_nprim(),
                     obs.max_l()
                     ); 
    double gamma=1.5;
    libint2::Engine std_engine(libint2::Operator::stg,obs.max_nprim(),obs.max_l());
    std_engine.set_params(gamma);

    // compute integrals

    auto shell2bf=obs.shell2bf(); // map shell index to basis funciton index
                                   // shell2bf[0] = index of the first bassi funciton in shell 0
                                   // shell2bf[1] = index of the first basis function in shell 1
                                   // ...
    const auto & buf_vec = std_engine.results();
    for (auto s1=0;s1!=obs.size();++s1){
        for (auto s2=0; s2!=obs.size();++s2){
            for (auto s3=0; s3!=obs.size();++s3){
                for (auto s4=0; s4!=obs.size();++s4){
                    //cout << "compute shell set {"<< s1 <<","<<s2<<","<<s3<<","<<s4<<"}...";
                    std_engine.compute(obs[s1],obs[s2],obs[s3],obs[4]);
                    cout<<"done"<<endl;
                    auto ints_shellset=buf_vec[0]; // location of the computed integrals
                    if (ints_shellset==nullptr)
                        continue; // nullprt retured if the entaire shell-set was screened out
                    auto bf1=shell2bf[s1];
                    auto n1=obs[s1].size();
                    auto bf2 =shell2bf[s2];
                    auto n2= obs[s2].size();
                    auto bf3 =shell2bf[s3];  // use ri 
                    auto n3= obs[s3].size(); // use ri
                    auto bf4 =shell2bf[s4]; // use ri
                    auto n4= obs[s4].size(); // use ri
                    for (auto f1=0;f1!=n1;++f1){
                        for (auto f2=0; f2!=n2;++f2){
                            for(auto f3=0; f3!=n3;++f3){
                                for(auto f4=0;f4!=n4;++f4){
                                cout<< "  "<<bf1+f1<<" " <<bf2+f2<<" " <<bf3+f3 <<" "<<bf4+f4<<" "<<setprecision(15)<< ints_shellset[f1*(n2*n3*n4)+f2*(n3*n4)+f3*(n4)+f4]<<endl;
                                        // print out results
                                }
                            }

                        }
                    }
                }
            }
		}
	}
    libint2::finalize();
    return 0;
}
