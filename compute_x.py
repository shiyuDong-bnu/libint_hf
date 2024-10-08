"""
X is 

"""
n_obs=23
n_occ=5
import numpy as np
square=np.load("stg_square_tensorf.npy")
hyb_stg=np.load("hyb_stg_tensorf.npy")
stg_tensor=np.load("stg_tensorf.npy")

X=np.zeros((n_occ,n_occ,n_occ,n_occ))
X+=square[:n_occ,:n_occ,:n_occ,:n_occ]

f_oooc=hyb_stg[:n_occ,:n_occ,:n_occ,:]
f_ocoo=np.moveaxis(f_oooc,[0,3,1,2],[0,1,2,3])
temp=np.einsum("mpij,klmp->ijkl",f_ocoo,f_oooc)
X-=2*temp
f_pqij=stg_tensor[:,:,:n_occ,:n_occ]
f_klpq=stg_tensor[:n_occ,:n_occ,:,:]
temp=np.einsum("pqij,klpq->ijkl",f_pqij,f_klpq)
X-=temp


