g++ hf.cc -g  -o hf.o  -I$HOME/opt/linalg/eigen3/include/eigen3 -I$HOME/opt/boost/include -I$HOME/opt/quantum_chem/libint2/include -fPIC \
                       -I$HOME/opt/linalg/xtensor/include -I$HOME/opt/linalg/xtl/include \
                       -L$HOME/opt/quantum_chem/libint2/lib -lint2 
