r"""
X is
=(f^2(r_{12}))_{ij}^{kl} 
-f_{ij}^{mp^\prime}f_{mp^\prime}^{kl} 
-f_{ij}^{p^\prime m}f_{p^\prime m}^{kl} 
-f_{ij}^{pq}f_{pq}^{kl} 
"""
n_obs=23
n_occ=5
gamma=1.5
import numpy as np
import pandas as pd
square=np.load("stg_square_tensorf.npy")
square/=(gamma)**2
hyb_stg=np.load("hyb_stg_tensorf.npy")
hyb_stg/=gamma
stg_tensor=np.load("stg_tensorf.npy")
stg_tensor/=gamma
stg_ao_tensor=np.concatenate((stg_tensor,hyb_stg),axis=-1)
##
C_mo=pd.read_csv("C_mo.csv",header=None).to_numpy()
C_ao_cabs=pd.read_csv("C_ao_cabs.csv",header=None).to_numpy()
# change from chemist's notation to physicst's notation
square=np.einsum("ijkl->ikjl",square)
stg_tensor=np.einsum("ijkl->ikjl",stg_tensor)
stg_ao_tensor=np.einsum("ijkl->ikjl",stg_ao_tensor)
## need to change from ao to mo.
square_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",square,C_mo,C_mo,C_mo,C_mo,optimize=True)
stg_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",stg_tensor,C_mo,C_mo,C_mo,C_mo,optimize=True)
stg_mo_oooc=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",stg_ao_tensor,C_mo,C_mo,C_mo,C_ao_cabs,optimize=True)
## then contraction.
## first term
X=np.zeros((n_occ,n_occ,n_occ,n_occ))
X+=square_mo[:n_occ,:n_occ,:n_occ,:n_occ]
## second term
f_oooc=stg_mo_oooc[:n_occ,:n_occ,:n_occ,:]
f_ocoo=np.moveaxis(f_oooc,[0,3,1,2],[0,1,2,3])
temp=np.einsum("mpij,klmp->ijkl",f_ocoo,f_oooc)
X-=temp
## third term
f_cooo=np.moveaxis(f_oooc,[3,0,1,2],[0,1,2,3])
f_ooco=np.moveaxis(f_oooc,[0,1,3,2],[0,1,2,3])
temp=np.einsum("pmij,klpm->ijkl",f_cooo,f_ooco)
X-=temp
## final term
f_pqij=stg_mo[:,:,:n_occ,:n_occ]
f_klpq=stg_mo[:n_occ,:n_occ,:,:]
temp=np.einsum("pqij,klpq->ijkl",f_pqij,f_klpq)
X-=temp
np.save("X.npy",X)
