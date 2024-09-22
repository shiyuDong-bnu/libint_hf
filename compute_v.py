import numpy as np
import pandas as pd

n_obs=23
n_occ=5
# load hartree_fock coeff
C_mo=pd.read_csv("C_mo.csv",header=None).to_numpy()
print(C_mo.shape)
# load cabs coeff
C_ao_cabs=pd.read_csv("C_ao_cabs.csv",header=None).to_numpy()
print(C_ao_cabs.shape)
# load integrals this are in chemist's notation.
eri=np.load("eri.npy")
eri_f=np.load("erif.npy")
np.testing.assert_allclose(eri,eri_f)
yukawa=np.load("yukawa.npy")
yukawa_f=np.load("yukawa_f.npy")
np.testing.assert_allclose(yukawa,yukawa_f)
heri=np.load("hyb_eri_tensor.npy")
heri_f=np.load("hyb_eri_tensorf.npy")
np.testing.assert_allclose(heri,heri_f)
stg=np.load("stg_tensor.npy")
stgf=np.load("stg_tensorf.npy")
np.testing.assert_allclose(stg,stgf)
hstg=np.load("hyb_stg_tensor.npy")
hstgf=np.load("hyb_stg_tensorf.npy")
np.testing.assert_allclose(hstg,hstgf)
# transform integral from chemist's notation to physicists notaion
gr=np.swapaxes(yukawa,1,2)
g=np.concatenate([eri,heri],3)
r=np.concatenate([stg,hstg],3)
g=np.swapaxes(g,1,2) 
r=np.swapaxes(r,1,2)
print(g.shape) # obs ,obs, obs,union
print(r.shape) # obs, obs, obs,union
# \begin{equation}
#     V_{ij}^{pq}=(gr)_{ij}^{pq}-g_{ij}^{kp^\prime}r_{kp^\prime}^{pq}-g_{ij}^{p^\prime k}r_{p^\prime k}^{pq}-g_{ij}^{rs}r_{rs}^{pq}
# \end{equation}
C_occ=C_mo[:,:n_occ]

V=np.zeros((n_obs,n_obs,n_occ,n_occ))
# term1
term1=np.einsum("abcd,aP,bQ,cI,dJ->PQIJ  ",gr,C_mo,C_mo,C_occ,C_occ,optimize=True)
# term2
g_temp=np.moveaxis(g,[0,1,2,3],[2,3,0,1])
g_kpprimij=np.einsum("aubc,aK,uM,bI,cJ->KMIJ",g_temp,C_occ,C_ao_cabs,C_occ,C_occ,optimize=True)
r_pqkpprime=np.einsum("abcu,aP,bQ,cK,uM->PQKM  ",r,C_mo,C_mo,C_occ,C_ao_cabs,optimize=True)
term2=np.einsum("KMIJ,PQKM ->PQIJ  ",g_kpprimij,r_pqkpprime,optimize=True)
# term3
g_temp=np.moveaxis(g,[0,1,2,3],[3,2,1,0])
r_temp=np.moveaxis(r,[0,1,2,3],[1,0,3,2])
g_pprimekij=np.einsum("uabc,uM,aK,bI,cJ->MKIJ",g_temp,C_ao_cabs,C_occ,C_occ,C_occ,optimize=True)
r_pqpprimek=np.einsum("abuc,aP,bQ,uM,cK->PQMK",r_temp,C_mo,C_mo,C_ao_cabs,C_occ,optimize=True)
term3=np.einsum("MKIJ,PQMK ->PQIJ  ",g_pprimekij,r_pqpprimek,optimize=True)
# term4
g_temp=g[:n_obs,:n_obs,:n_obs,:n_obs]
r_temp=r[:n_obs,:n_obs,:n_obs,:n_obs]
g_rsij=np.einsum(
    "abcd,aR,bS,cI,dJ->RSIJ  ",g_temp,C_mo,C_mo,C_occ,C_occ,optimize=True)
r_pqsrs=np.einsum("abcd,aP,bQ,cR,dS->PQRS  ",r_temp,C_mo,C_mo,C_mo,C_mo,optimize=True)
term4=np.einsum("RSIJ,PQRS->PQIJ  ",g_rsij,r_pqsrs,optimize=True)
V=term1-term2-term3-term4
np.save("V.npy",V)
