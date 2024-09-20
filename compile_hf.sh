g++ hf_pure.cc -g  -o hf_pure.o  -I$HOME/opt/linalg/eigen3/include/eigen3 -I$HOME/opt/boost/include -I$HOME/opt/quantum_chem/libint_t4/include -fPIC \
                       -I$HOME/opt/linalg/xtensor/include -I$HOME/opt/linalg/xtl/include \
                       -I$HOME/opt/linalg/xtensor-blas/include/ \
                       -L$HOME/opt/quantum_chem/libint_t4/lib -lint2 \
                       -L/home/sydong/opt/linalg/openblas/lib/ -lopenblas 
