import numpy, scipy
import sys, os
# change the path if necessary
sys.path.append("/home/ningzhangcaltech/Github_Repo/libdmet2")

import pyscf
from pyscf.pbc import gto, scf, df

from pyscf.isdf.isdf_local_k import PBC_ISDF_Info_Quad_K
from pyscf.isdf              import isdf_tools_cell

import libdmet
from libdmet.system import lattice
from libdmet.dmet import rdmet
from libdmet.lo import make_lo

beta  = 1000.0 # numpy.inf
#kmesh = [2, 2, 2]
kmesh = [1, 2, 2]

boxlen = 3.57371000
prim_a = numpy.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen]])
atm = [
    ['C', (0.        , 0.        , 0.    )],
    ['C', (0.8934275 , 0.8934275 , 0.8934275)],
    ['C', (1.786855  , 1.786855  , 0.    )],
    ['C', (2.6802825 , 2.6802825 , 0.8934275)],
    ['C', (1.786855  , 0.        , 1.786855)],
    ['C', (2.6802825 , 0.8934275 , 2.6802825)],
    ['C', (0.        , 1.786855  , 1.786855)],
    ['C', (0.8934275 , 2.6802825 , 2.6802825)],
]
ke_cutoff = 70
basis     = 'gth-szv'
c = isdf_tools_cell.build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=ke_cutoff, basis=basis, pseudo="gth-pade", verbose=4)

lat  = lattice.Lattice(c, kmesh)
kpts = lat.kpts
nkpts = len(kpts)
exxdiv = None

# cderi = 'gdf_ints.h5'
# gdf = df.GDF(c, kpts)
# gdf.verbose = 0
# gdf._cderi_to_save = cderi
# gdf.build()
# kmf = scf.KRHF(c, kpts, exxdiv=exxdiv).density_fit()
# kmf.with_df = gdf
# kmf.with_df._cderi = cderi
# kmf.conv_tol = 1e-12
# kmf.verbose = 4
# kmf.kernel()

prim_partition = [[0,1],[2,3],[4,5],[6,7]]
pbc_isdf_info = PBC_ISDF_Info_Quad_K(c,
                                    kmesh=kmesh,  
                                    with_robust_fitting=True, 
                                    rela_cutoff_QRCP=2e-4, 
                                    direct=False
                                    )
pbc_isdf_info.verbose = 10
pbc_isdf_info.build_IP_local(c=70, m=5, group=prim_partition)
pbc_isdf_info.build_auxiliary_Coulomb()
print("effective c = ", float(pbc_isdf_info.naux) / pbc_isdf_info.nao) 

kmf2 = scf.KRHF(c, kpts, exxdiv=exxdiv)
kmf2.with_df = pbc_isdf_info
kmf2.conv_tol = 1e-12
kmf2.verbose = 4
kmf2.kernel()

kmf = kmf2

# exit(1)

# define local orbitals
coeff_ao_lo, idx_cor, idx_val, idx_vir = make_lo.get_iao(
    kmf, minao="scf", full_return=True
)

# define impurity information
lat.build(idx_val=idx_val, idx_virt=idx_vir)

# define solver
from libdmet.solver.fci_solver import FCI
from libdmet.solver.cc_solver import CCSolver as CCSD
solver = FCI(restricted=True, tol=1e-10, max_cycle=1000, verbose=5, beta=beta)
solver_argss = [["C_lo_eo"]]

# define DMET object
dmet_obj = rdmet.RDMET(
    lat, kmf, solver, coeff_ao_lo, 
    vcor=None, solver_argss=solver_argss,
)
dmet_obj.dump_flags()
dmet_obj.beta = beta
dmet_obj.fit_method = 'CG'
dmet_obj.fit_kwargs = {"test_grad": False}
dmet_obj.mu_glob = 0.2

def get_h2_emb(C_lo_eo, **kwargs):
    coeff_lo_eo_r = numpy.array(C_lo_eo) # in supercell basis
    spin, ncell, nlo, neo = coeff_lo_eo_r.shape
    
    print("get_h2_emb")
    print("spin, ncell, nlo, neo = ", spin, ncell, nlo, neo)
    
    coeff_lo_eo_full = coeff_lo_eo_r.reshape(spin, ncell * nlo, neo)

    coeff_ao_lo_k = numpy.array(dmet_obj.C_ao_lo)
    nao = coeff_ao_lo_k.shape[1]

    coeff_ao_lo_k = coeff_ao_lo_k.reshape(ncell, nao, nlo)
    coeff_ao_lo_r = lat.k2R(coeff_ao_lo_k)
    
    print("coeff_lo_eo_r.shape = ", coeff_lo_eo_r.shape)
    print("coeff_ao_lo_k.shape = ", coeff_ao_lo_k.shape)
    print("coeff_ao_lo_r.shape = ", coeff_ao_lo_r.shape)

    coeff_ao_lo_full = lat.expand(coeff_ao_lo_r)
    coeff_ao_eo = numpy.einsum("mp,spr->smr", coeff_ao_lo_full, coeff_lo_eo_full)
    
    print("coeff_ao_lo_full.shape = ", coeff_ao_lo_full.shape)
    print("coeff_ao_eo.shape = ", coeff_ao_eo.shape)
    print("coeff_ao_eo.dtype = ", coeff_ao_eo.dtype)

    assert coeff_ao_lo_r.shape == (ncell, nao, nlo)
    assert coeff_ao_lo_full.shape == (ncell * nao, nlo * ncell)
    assert coeff_ao_eo.shape == (spin, ncell * nao, neo)

    print("C_lo_eo.shape = ", C_lo_eo.shape)
    print("C_lo_eo = ", )
    # print(C_lo_eo)

    from libdmet.dmet import gdmet, rdmet, udmet
    if isinstance(dmet_obj, rdmet.RDMET) or isinstance(dmet_obj, udmet.UDMET): # RHF and UHF
        from libdmet.dmet.rdmet_helper import get_h2_emb
        h2_emb = get_h2_emb(dmet_obj, C_lo_eo, **kwargs)
        # in fact the working function is trans_h2_loc
        print("h2_emd.shape = ", h2_emb.shape)
        print("h2_emd.dtype = ", h2_emb.dtype)

    else: # not sure about the shape, you can try
        from libdmet.dmet.gdmet_helper import get_h2_emb
        h2_emb = get_h2_emb(dmet_obj, C_lo_eo, **kwargs)

    neo2 = neo * (neo + 1) // 2
    spin2 = spin * (spin + 1) // 2
    h2_emb = h2_emb.reshape(spin2, neo2, neo2)
    
    print(h2_emb[0][0][:100])
    print(h2_emb[0][0][300:310])
    print(h2_emb[0][0][525:537])

    assert 1 == 2
    
    return h2_emb

dmet_obj.get_h2_emb = get_h2_emb

dmet_obj.kernel()
