import numpy as np
import pandas as pd

n_obs=23
n_occ=5
gamma=1.5
# load hartree_fock coeff
C_mo=pd.read_csv("C_mo.csv",header=None).to_numpy()
print(C_mo.shape)
# load cabs coeff
C_ao_cabs=pd.read_csv("C_ao_cabs.csv",header=None).to_numpy()
print(C_ao_cabs.shape)

## initialize B term
B=np.zeros((n_occ,n_occ,n_occ,n_occ))
## compute fock term
# load ao integral
ao_stg_square_tensor=np.load("stg_square_tensorf.npy")
## double commutator term
C_ao_occ=C_mo[:,:n_occ]
temp=np.einsum("abcd,aA,bB,cC,dD->ABCD",ao_stg_square_tensor,C_ao_occ,C_ao_occ,C_ao_occ,C_ao_occ)
B+=temp
## square term 
hyb_ao_stg_square_tensor=np.load("hyb_square_stg_tensorf.npy")
# concatenate
print(hyb_ao_stg_square_tensor.shape)
print(ao_stg_square_tensor.shape)
ao_stg_square_tensor=np.concatenate((ao_stg_square_tensor,hyb_ao_stg_square_tensor),axis=-1)
print(ao_stg_square_tensor.shape)
ao_stg_square_tensor/=(gamma**2)
mo_square_cabs_tensor=np.einsum("abcd,aA,bB,cC,dD->ABCD",ao_stg_square_tensor,
                           C_ao_occ,
                           C_ao_occ,
                           C_ao_occ,
                           C_ao_cabs,optimize=True)          # occ occ occ cabs
mo_square_obs_tensor=np.einsum("abcd,aA,bB,cC,dD->ABCD",ao_stg_square_tensor[:,:,:,:n_obs],
                           C_ao_occ,
                           C_ao_occ,
                           C_ao_occ,
                           C_mo,optimize=True)          # occ occ occ cabs
print(mo_square_cabs_tensor.shape,mo_square_obs_tensor.shape)
mo_square_tensor=np.concatenate((mo_square_obs_tensor,mo_square_cabs_tensor),axis=-1)
## F+K matrix
F=np.einsum("abcd,aA,bB->ABCD",ao_stg_square_tensor,C_ao_occ,C_ao_occ,optimize=True)
