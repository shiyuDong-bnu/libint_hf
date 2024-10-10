import numpy as np
import pandas as pd

"""
&AAA+ABA+BAA+AAB+BAB= \\
    & =EEE-ECE+CAC -CAE-EAC -DBA-ABD -DBD
"""
n_obs=23
n_occ=5
gamma=1.5
n_cabs=69
n_ri=n_obs+n_cabs

def load_tensors():
    # load hartree_fock coeff
    C_mo=pd.read_csv("./int_data/C_mo.csv",header=None).to_numpy()
    print(C_mo.shape)
    # load cabs coeff
    C_ao_cabs=pd.read_csv("./int_data/C_ao_cabs.csv",header=None).to_numpy()
    print(C_ao_cabs.shape)
    T=pd.read_csv("./int_data/T.csv",header=None).to_numpy() 
    print(T.shape)
    T_obs_cabs=pd.read_csv("./int_data/T_obs_cabs.csv",header=None).to_numpy() 
    print(T_obs_cabs.shape)
    T_cabs_cabs=pd.read_csv("./int_data/T_cabs_cabs.csv",header=None).to_numpy() 
    print(T_cabs_cabs.shape)
    V=pd.read_csv("./int_data/V.csv",header=None).to_numpy()
    print(V.shape)
    V_obs_cabs=pd.read_csv("./int_data/V_obs_cabs.csv",header=None).to_numpy()
    print(V_obs_cabs.shape)
    V_cabs_cabs=pd.read_csv("./int_data/V_cabs_cabs.csv",header=None).to_numpy()
    print(V_cabs_cabs.shape)
    Fock=pd.read_csv("./int_data/Fock.csv",header=None).to_numpy()
    print("Fock matrix shape is" ,Fock.shape)
    S=pd.read_csv("./int_data/S.csv",header=None).to_numpy()
    print("S matrix shape is ",S.shape)
    eri_tensor=np.load("./int_data/erif.npy")
    print("eri tensor shape is ",eri_tensor.shape)
    hyb_eri_tensor=np.load("./int_data/hyb_eri_tensorf.npy")
    print("hyb eri tensor shape is ",hyb_eri_tensor.shape)
    cooc_eri_tensor=np.load("./int_data/cooc_eri_tensorf.npy")
    oocc_eri_tensor=np.load("./int_data/oocc_eri_tensorf.npy")
    ao_stg_square_tensor=np.load("./int_data/stg_square_tensorf.npy")
    hyb_ao_stg_square_tensor=np.load("./int_data/hyb_square_stg_tensorf.npy")
    return (
        C_mo,
        C_ao_cabs,
        T,  
        T_obs_cabs,
        T_cabs_cabs,
        V, 
        V_obs_cabs,
        V_cabs_cabs,
        Fock,
        S,
        eri_tensor,
        hyb_eri_tensor,
        cooc_eri_tensor,
        oocc_eri_tensor,
        ao_stg_square_tensor,
        hyb_ao_stg_square_tensor ,
            
    )
(
        C_mo,
        C_ao_cabs,
        T,  
        T_obs_cabs,
        T_cabs_cabs,
        V, 
        V_obs_cabs,
        V_cabs_cabs,
        Fock,
        S,
        eri_tensor,
        hyb_eri_tensor,
        cooc_eri_tensor ,
        oocc_eri_tensor ,
        ao_stg_square_tensor,
        hyb_ao_stg_square_tensor ,
    )=load_tensors()
def generate_ri(obs_obs,obs_cabs,cabs_cabs):
    t_ri_ri=np.zeros((n_ri,n_ri))
    t_ri_ri[:n_obs,:n_obs]=obs_obs
    t_ri_ri[:n_obs,n_obs:]=obs_cabs
    t_ri_ri[n_obs:,:n_obs]=obs_cabs.T
    t_ri_ri[n_obs:,n_obs:]=cabs_cabs
    return t_ri_ri

## initialize B term
B=np.zeros((n_occ,n_occ,n_occ,n_occ))
## compute fock term
# load ao integral
## change from chemist's notation to physicst's notation
ao_stg_square_tensor=np.einsum("abcd->acbd",ao_stg_square_tensor)
## double commutator term
C_ao_occ=C_mo[:,:n_occ]
temp=np.einsum("abcd,aA,bB,cC,dD->ABCD",ao_stg_square_tensor,C_ao_occ,C_ao_occ,C_ao_occ,C_ao_occ)
B+=temp
## square term 

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

## update B
B+=(f_plus_k_times_square+square_times_f_plus_k)/2

## calculate  rkr terms

K_obs_cabs=np.einsum("abcd,bc->ad",hyb_eri_tensor,Density)
K_obs_obs=np.einsum("abcd,bc->ad",eri_tensor,Density)
K_cabs_cabs=np.einsum("abcd,bc->ad",cooc_eri_tensor,Density)
# slice to generate ri
K_ri_ri=np.zeros((n_ri,n_ri))
K_ri_ri[:n_obs,:n_obs]=K_obs_obs
K_ri_ri[:n_obs,n_obs:]=K_obs_cabs
K_ri_ri[n_obs:,:n_obs]=K_obs_cabs.T
K_ri_ri[n_obs:,n_obs:]=K_cabs_cabs
## change from ao to mo
K_ri_ri_mo=np.zeros((n_ri,n_ri))
K_ri_ri_mo[:n_obs,:n_obs]=np.einsum("ij,iI,jJ",K_obs_obs,C_mo,C_mo)
K_ri_ri_mo[n_obs:,n_obs:]=np.einsum("ij,iI,jJ",K_ri_ri,C_ao_cabs,C_ao_cabs)
K_ri_ri_mo[:n_obs,n_obs:]=np.einsum("ij,iI,jJ",K_ri_ri[:n_obs,:],C_mo,C_ao_cabs)
K_ri_ri_mo[n_obs:,:n_obs]=np.einsum("ij,iI,jJ",K_ri_ri[:n_obs,:],C_mo,C_ao_cabs).T
## generat r 
stg_tensor=np.load("./int_data/stg_tensorf.npy")/gamma
hyb_stg_tensor=np.load("./int_data/hyb_stg_tensorf.npy")/gamma
cooc_stg_tensor=np.load("./int_data/cooc_stg_tensorf.npy")/gamma
# 
roor_stg_tensor=np.zeros((n_ri,n_obs,n_obs,n_ri))
roor_stg_tensor[:n_obs,:n_obs,:n_obs,:n_obs,]=stg_tensor
roor_stg_tensor[n_obs:,:n_obs,:n_obs,n_obs:]=cooc_stg_tensor
roor_stg_tensor[:n_obs,:n_obs,:n_obs,n_obs:]=hyb_stg_tensor
roor_stg_tensor[n_obs:,:n_obs,:n_obs,:n_obs]=np.moveaxis(hyb_stg_tensor,[3,1,2,0],[0,1,2,3])
# chemist's notaiton to physcist's notation
roor_stg_tensor=np.einsum("ijkl->ikjl",roor_stg_tensor)
# use symmetry 
oorr_stg_tensor=np.moveaxis(roor_stg_tensor,[2,1,0,3],[0,1,2,3])
# from ao to mo
oocc_stg_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",oorr_stg_tensor,C_ao_occ,C_ao_occ,C_ao_cabs,C_ao_cabs,optimize=True)
oooo_stg_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",oorr_stg_tensor[:n_obs,:n_obs,:n_obs,:n_obs],
                                               C_ao_occ,C_ao_occ,C_mo,C_mo,optimize=True)
oooc_stg_mo=np.einsum("ijkl,iI,jJ,kK,lL->IJKL",oorr_stg_tensor[:n_obs,:n_obs,:n_obs,:],
                                                C_ao_occ,C_ao_occ,C_mo,C_ao_cabs,optimize=True)
oorr_stg_mo=np.zeros((n_occ,n_occ,n_ri,n_ri))
print("oooo_stg_mo shape is ",oooo_stg_mo.shape)
oorr_stg_mo[:,:,:n_obs,:n_obs]=oooo_stg_mo
oorr_stg_mo[:,:,n_obs:,n_obs:]=oocc_stg_mo
oorr_stg_mo[:,:,:n_obs,n_obs:]=oooc_stg_mo
oorr_stg_mo[:,:,n_obs:,:n_obs]=np.moveaxis(oooc_stg_mo,[0,1,3,2],[0,1,2,3])
rroo_stg_mo=np.moveaxis(oorr_stg_mo,[2,3,0,1],[0,1,2,3])
## contraction
temp=np.einsum("pqmn,rp,klrq->klmn",rroo_stg_mo,
                                    K_ri_ri_mo,
                                    oorr_stg_mo,
                                    optimize=True)
B-=temp
## ECE
## generate Fock_ri_ri_mo
## slice to generate
T_ri_ri=generate_ri(T,T_obs_cabs,T_cabs_cabs)
V_ri_ri=generate_ri(V,V_obs_cabs,V_cabs_cabs)
J_cabs_cabs=np.einsum("ijkl,ij->kl",oocc_eri_tensor,Density)
J_ri_ri=generate_ri(J,J_obs_cabs,J_cabs_cabs)
Fock_ri_ri=T_ri_ri+V_ri_ri+2*J_ri_ri-K_ri_ri
# change from ao to mo
Fock_cabs_cabs=np.einsum("ij,iI,jJ->IJ",Fock_ri_ri,C_ao_cabs,C_ao_cabs)
Fock_obs_obs=np.einsum("ij,iI,jJ->IJ",Fock_ri_ri[:n_obs,:n_obs],C_mo,C_mo)
Fock_obs_cabs=np.einsum("ij,iI,jJ->IJ",Fock_ri_ri[:n_obs,:],C_mo,C_ao_cabs)
Fock_ri_ri_mo=generate_ri(Fock_obs_obs,Fock_obs_cabs,Fock_cabs_cabs)
##
rooo_stg_mo=rroo_stg_mo[:,:n_occ,:,:]
ooro_stg_mo=np.moveaxis(rooo_stg_mo,[2,3,0,1],[0,1,2,3])
# do contraction
temp=np.einsum("pjmn,rp,klrj->klmn",rooo_stg_mo,Fock_ri_ri_mo,ooro_stg_mo,optimize=True)
B-=temp
## CAC
## genearte ocoo
oooc_stg_tensor=oorr_stg_mo[:n_occ,:n_occ,:n_occ,n_obs:]    #occ cabs occ occ
ocoo_stg_tensor=np.moveaxis(oooc_stg_tensor,[2,3,0,1],[0,1,2,3])
print(oooc_stg_tensor.shape)
Fock_occ_occ_mo=Fock_ri_ri_mo[:n_occ,:n_occ]
## contraction
temp=np.einsum("ibmn,ji,kljb->klmn",ocoo_stg_tensor,Fock_occ_occ_mo,oooc_stg_tensor,optimize=True)
B+=temp
## DBD
ooob_stg_tensor=oorr_stg_mo[:,:,:n_obs,n_occ:n_obs] 
oboo_stg_tensor=np.moveaxis(ooob_stg_tensor,[2,3,0,1],[0,1,2,3])
Fock_obs_obs_mo=Fock_ri_ri_mo[:n_obs,:n_obs]
# contraction
temp=np.einsum("pbmn,rp,klrb->klmn",oboo_stg_tensor,Fock_obs_obs_mo,ooob_stg_tensor,optimize=True)
B-=temp
##  CAE+EAC
oooc_stg_tensor=oorr_stg_tensor[:,:,:n_occ,n_obs:]
ocoo_stg_tensor=np.moveaxis(oooc_stg_tensor,[2,3,0,1],[0,1,2,3])
oorc_stg_tensor=oorr_stg_tensor[:,:,:,n_obs:]
Fock_ri_occ_mo=Fock_ri_ri_mo[:,:n_occ]
temp=np.einsum("ibmn,pi,klpb->klmn",ocoo_stg_tensor,Fock_ri_occ_mo,oorc_stg_tensor,optimize=True)
swap_temp=np.einsum("ibkl,pi,mnpb->klmn",ocoo_stg_tensor,Fock_ri_occ_mo,oorc_stg_tensor,optimize=True)
B-=2*(temp+swap_temp)
## DBA+ABD
oocv_stg_tensor=oorr_stg_tensor[:,:,n_obs:,n_occ:n_obs]
cvoo_stg_tensor=np.moveaxis(oocv_stg_tensor,[2,3,0,1],[0,1,2,3])
ooov_stg_tensor=oorr_stg_tensor[:,:,:n_obs,n_occ:n_obs]
Fock_obs_vir_mo=Fock_ri_ri_mo[:n_obs,n_occ:n_obs]
temp=np.einsum("abmn,qa,klqb->klmn",cvoo_stg_tensor,Fock_obs_vir_mo,ooov_stg_tensor)
swap_temp=np.einsum("abkl,qa,mnqb->klmn",cvoo_stg_tensor,Fock_obs_vir_mo,ooov_stg_tensor)
B-=2*(temp+swap_temp)
