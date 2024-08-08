#!/usr/bin/env python

"""
CCO using SC cell.
"""

# from mpi4pyscf.tools import mpi
import time
import numpy as np
from pyscf.pbc import df, dft
from pyscf import lib
from pyscf.pbc.lib import chkfile
from libdmet.system import lattice
from libdmet.utils.iotools import read_poscar
import libdmet.utils.logger as log
from libdmet.mean_field import pbc_helper as pbc_hp

from   pyscf.isdf.isdf_jk         import _benchmark_time
import pyscf.isdf.isdf_local      as isdf_linear_scaling
import pyscf.isdf.isdf_local_k    as isdf_linear_scaling_k
# from   pyscf.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition

############################################################################################################

log.verbose = "DEBUG1"

start = time.time()

nelec_dop = 50

cell = read_poscar(fname="./CCO-2x2-frac.vasp")
cell.basis  = {'Cu1':'cc-pvdz.dat', 
               'Cu2':'cc-pvdz.dat', 
               'O1' :'cc-pvdz.dat',
               'Ca' :'cc-pvdz.dat'}
cell.pseudo = {'Cu1': 'gth-pbe-q19', 
               'Cu2': 'gth-pbe-q19', 
               'O1' : 'gth-pbe',
               'Ca' : 'gth-pbe'}

cell.spin = 0 
cell.verbose = 5
cell.max_memory = 100000
cell.precision = 1e-14
cell.build()

kmesh = [4, 4, 2]
nkpts = np.prod(kmesh)
nelec0 = int(cell.nelectron)
cell.nelectron = (nelec0 * nkpts - nelec_dop) / nkpts
doping = nelec_dop / (nkpts * 4)
cell.build()

#cell.space_group_symmetry = True
#cell.symmorphic = True
cell.build()

print ("doping:             ", doping)
print ("doping nelec:       ", nelec_dop)
print ("total electrons:    ", cell.tot_electrons(nkpts))
print ("nelec per cell:     ", cell.nelectron)
print ("total spin:         ", cell.spin)
print ("atom charges:       ", cell.atom_charges())
print ("atom charges total: ", cell.atom_charges().sum())

# exit(1)

Cu_3d_A = cell.search_ao_label("Cu1 3dx2-y2")
Cu_3d_B = cell.search_ao_label("Cu2 3dx2-y2")

O_3band = np.hstack((cell.search_ao_label("4 O1 2py"), cell.search_ao_label("5 O1 2px"), \
         cell.search_ao_label("6 O1 2py")  , cell.search_ao_label("7 O1 2px"), \
         cell.search_ao_label("8 O1 2px")  , cell.search_ao_label("9 O1 2py"), \
         cell.search_ao_label("10 O1 2px") , cell.search_ao_label("11 O1 2py")))

Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

kpts_symm = cell.make_kpts(kmesh, 
                           with_gamma_point=True, 
                           #wrap_around=True, 
                           #space_group_symmetry=True, 
                           #time_reversal_symmetry=True
                           )

print(kpts_symm)
# print(kpts)
# exit(1)

## gdf_fname = '../integral/gdf_ints_CCO_442.h5'
# gdf_fname = '/global/cfs/cdirs/m4010/zhcui/proj/cuprate_doping/pressure/CCO/gth-cc_new_2/ab_0.000_cmem/gdf_ints_CCO_442.h5'
# gdf = df.GDF(cell, kpts)
# gdf._cderi_to_save = gdf_fname
# gdf.use_mpi = True

# kmf = dft.KUKS(cell, kpts_symm).density_fit()
kmf = dft.KUKS(cell, kpts_symm)
kmf.xc = 'pbe0'
kmf.grids.level = 5
kmf.exxdiv = None
kmf.init_guess_breaksym = False

from pyscf.data.nist import HARTREE2EV
sigma = 0.2 / HARTREE2EV
kmf = pbc_hp.smearing_(kmf, sigma=sigma, method="fermi", tol=1e-12)
#kmf.with_df = gdf
#kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-7
# chk_fname = './CCO_UPBE0_442.chk'
chk_fname = '/scratch/global/ningzhang/CCO_UPBE0_442_2.chk'
kmf.chkfile = chk_fname # NOTE: change to your scratch
kmf.diis_space = 15
kmf.max_cycle = 150

from libdmet.basis import trans_1e
from libdmet.lo import lowdin
#kmf_no_symm = pbc_hp.kmf_symm_(kmf)
kmf_no_symm = kmf
C_ao_lo = lowdin.lowdin_k(kmf_no_symm, pre_orth_ao='SCF')
dm0 = kmf_no_symm.get_init_guess(key='atom')

dm0_lo = trans_1e.trans_rdm1_to_lo(dm0, C_ao_lo, kmf_no_symm.get_ovlp())
dm0_lo_R = Lat.k2R(dm0_lo)
Lat.mulliken_lo_R0(dm0_lo_R[:, 0])

dm0_lo_R[0, 0, Cu_3d_A, Cu_3d_A] *= 2.0
dm0_lo_R[0, 0, Cu_3d_B, Cu_3d_B]  = 0.0
dm0_lo_R[1, 0, Cu_3d_A, Cu_3d_A]  = 0.0
dm0_lo_R[1, 0, Cu_3d_B, Cu_3d_B] *= 2.0

dm0_lo = Lat.R2k(dm0_lo_R)
dm0 = trans_1e.trans_rdm1_to_ao(dm0_lo, C_ao_lo)

dm0_lo = trans_1e.trans_rdm1_to_lo(dm0, C_ao_lo, kmf_no_symm.get_ovlp())
dm0_lo_R = Lat.k2R(dm0_lo)

print ("after polarization")
Lat.mulliken_lo_R0(dm0_lo_R[:, 0])

# dm0 = dm0[:, kpts_symm.ibz2bz]

# exit(1) 

# kmf.kernel(dm0=dm0)

# data = chkfile.load("./CCO_UPBE0_442.chk", "scf")
# kmf.__dict__.update(data)

# kmf = pbc_hp.kmf_symm_(kmf)

#########################    ISDF #########################

DM_CACHED = dm0 

Ke_CUTOFF = [128,192,256,384]
C_ARRAY = [15,20,20,25,30]
RELA_QR = [3e-3,1e-3,5e-4,2e-4,1e-4]
partition = [[0],[1],[2], [3], [4], [5], [6], [7],
             [8],[9],[10],[11],[12],[13],[14],[15]]  

for ke_cutoff in Ke_CUTOFF:
    
    if ke_cutoff == 128:
        c_array      = C_ARRAY
        relaqr_array = RELA_QR
    else:
        c_array      = C_ARRAY[-2:]
        relaqr_array = RELA_QR[-2:]
    
    cell.ke_cutoff = ke_cutoff
    cell.build()
    
    mesh = cell.mesh
    mesh = [(mesh[0]+7)//8*8,(mesh[1]+7)//8*8,(mesh[2]+7)//8*8]
    mesh = np.asarray(mesh, dtype=np.int32)
    cell.build(mesh=mesh)
    
    kmf = dft.KUKS(cell, kpts_symm)
    kmf.xc = 'pbe0'
    kmf.grids.level = 5
    kmf.exxdiv = None
    kmf.init_guess_breaksym = False
    
    sigma = 0.2 / HARTREE2EV
    kmf = pbc_hp.smearing_(kmf, sigma=sigma, method="fermi", tol=1e-12)
    #kmf.with_df = gdf
    #kmf.with_df._cderi = gdf_fname
    kmf.conv_tol      = 1e-7
    kmf.conv_tol_grad = 1e-1                                     # NOTE: when ke_cutoff is not large enough, the grad cannot converges to sqrt(conv_tol)
    # chk_fname = './CCO_UPBE0_442.chk'
    chk_fname = '/scratch/global/ningzhang/CCO_UPBE0_442_2.chk'  # NOTE: change to your scratch
    kmf.chkfile = chk_fname
    kmf.diis_space = 15
    # kmf.max_cycle = 150
    kmf.max_cycle = 32
    
    for c,rela_qr in list(zip(c_array,relaqr_array)):
        
        print('--------------------------------------------')
        print('C = %d, QR=%f, kc_cutoff = %d' % (c, rela_qr, ke_cutoff))

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        pbc_isdf_info = isdf_linear_scaling_k.PBC_ISDF_Info_Quad_K(cell, 
                                                                   kmesh=kmesh,  
                                                                   with_robust_fitting=True, 
                                                                   rela_cutoff_QRCP=rela_qr, 
                                                                   direct=True, 
                                                                   limited_memory=True,
                                                                   build_K_bunchsize=128
                                                                   )
        pbc_isdf_info.verbose      = 5
        pbc_isdf_info.ke_cutoff_pp = 512
        pbc_isdf_info.build_IP_local(c=c, m=5, group=partition)
        # pbc_isdf_info.build_auxiliary_Coulomb(debug=True)
        print("effective c = ", float(pbc_isdf_info.naux) / pbc_isdf_info.nao) 
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
        print(_benchmark_time(t1, t2, 'build ISDF', pbc_isdf_info))

        t1 = (lib.logger.process_clock(), lib.logger.perf_counter())
        #mf = scf.RHF(cell)
        kmf.with_df   = pbc_isdf_info
        #mf.max_cycle = 100
        #mf.conv_tol  = 1e-8
        if DM_CACHED is not None:
            kmf.kernel(DM_CACHED)
        else:
            kmf.kernel()
        t2 = (lib.logger.process_clock(), lib.logger.perf_counter())
                        
        print(_benchmark_time(t1, t2, 'RHF_bench', kmf))
        DM_CACHED = kmf.make_rdm1()

######################### END ISDF #########################

Lat.analyze(kmf, pre_orth_ao='SCF')

log.info(kmf, 'S^2 = %s, 2S+1 = %s' % kmf.spin_square())
np.save("ovlp_442.npy", kmf.get_ovlp())
np.save("hcore_442.npy", kmf.get_hcore())

