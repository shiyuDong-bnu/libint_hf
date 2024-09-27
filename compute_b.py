import numpy as np
import pandas as pd
import scipy

n_obs=23
n_occ=5
gamma=1.5
# load hartree_fock coeff
C_mo=pd.read_csv("C_mo.csv",header=None).to_numpy()
print(C_mo.shape)
# load cabs coeff
C_ao_cabs=pd.read_csv("C_ao_cabs.csv",header=None).to_numpy()
print(C_ao_cabs.shape)
T=pd.read_csv("T.csv",header=None).to_numpy() 
print(T.shape)
T_obs_cabs=pd.read_csv("T_obs_cabs.csv",header=None).to_numpy() 
print(T_obs_cabs.shape)
V=pd.read_csv("V.csv",header=None).to_numpy()
print(V.shape)
V_obs_cabs=pd.read_csv("V_obs_cabs.csv",header=None).to_numpy()
print(V_obs_cabs.shape)
Fock=pd.read_csv("Fock.csv",header=None).to_numpy()
print("Fock matrix shape is" ,Fock.shape)
S=pd.read_csv("S.csv",header=None).to_numpy()
print("S matrix shape is ",S.shape)
eri_tensor=np.load("erif.npy")
print("eri tensor shape is ",eri_tensor.shape)
hyb_eri_tensor=np.load("hyb_eri_tensorf.npy")
print("hyb eri tensor shape is ",hyb_eri_tensor.shape)
## initialize B term
B=np.zeros((n_occ,n_occ,n_occ,n_occ))
## compute fock term
# load ao integral
ao_stg_square_tensor=np.load("stg_square_tensorf.npy")
## change from chemist's notation to physicst's notation
ao_stg_square_tensor=np.einsum("abcd->acbd",ao_stg_square_tensor)
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
# chemist's notation ot physicts's notation
ao_stg_square_tensor=np.einsum("abcd->acbd",ao_stg_square_tensor)
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
                           C_mo,optimize=True)          # occ occ occ obs
print(mo_square_cabs_tensor.shape,mo_square_obs_tensor.shape)
mo_square_tensor=np.concatenate((mo_square_obs_tensor,mo_square_cabs_tensor),axis=-1)  #occ occ occ ri
print(mo_square_tensor.shape)
## F+K matrix
## formation of fock matrix ,need T+V+2J-K in hartree fock space.?
Density=np.einsum("aA,bA->ab",C_ao_occ,C_ao_occ)
J=np.einsum("abcd,cd->ab",eri_tensor,Density)
K=np.einsum("ikjl,kl->ij",eri_tensor,Density)
F_plus_K=T+V+2*J
J_obs_cabs=np.einsum("abcd,ab->cd",hyb_eri_tensor,Density)
F_plus_K_obs_cabs=T_obs_cabs+V_obs_cabs+2*J_obs_cabs
F_plus_K_obs_ri=np.concatenate((F_plus_K,F_plus_K_obs_cabs),axis=-1)
print("F_plus_K_obs_cabs shape is ",F_plus_K_obs_cabs.shape)
print("F_plus_K_obs_ri shape is ",F_plus_K_obs_ri.shape)
F_plus_K_occ_obs=np.einsum("ij,iI,jJ->IJ",F_plus_K,C_ao_occ,C_mo)
F_plus_K_occ_cabs=np.einsum("ij,iI,jJ->IJ",F_plus_K_obs_ri,C_ao_occ,C_ao_cabs)
F_plus_K_occ_ri=np.concatenate((F_plus_K_occ_obs,F_plus_K_occ_cabs),axis=-1)
print("F_plus_K_occ_cabs shape is ",F_plus_K_occ_cabs.shape)
## now get all term in mo basis , do contraction
print(F_plus_K_occ_ri.shape)
print(mo_square_tensor.shape)
## transpose tensor ,but keep einsum index the same of equations.
f_plus_k_times_square=np.einsum("ia,klaj->klij",F_plus_K_occ_ri,
                                    np.moveaxis(mo_square_tensor,(0,1,2,3),(0,1,3,2)))
f_plus_k_times_square+=np.einsum("ja,klia->klij",F_plus_K_occ_ri,
                                 mo_square_tensor)
square_times_f_plus_k=np.einsum("alij,ka->klij",np.moveaxis(mo_square_tensor,(0,1,2,3),(3,1,2,0))
                                ,F_plus_K_occ_ri)
square_times_f_plus_k+=np.einsum("kaij,la->klij",np.moveaxis(mo_square_tensor,(0,1,2,3),(0,3,2,1)),
                                 F_plus_K_occ_ri)
B+=(f_plus_k_times_square+square_times_f_plus_k)/2
## calculate  rkr terms
# k_obs_cabs=
# k_obs_obs=
# k_cabs_cabs=
# B+=rkr_term