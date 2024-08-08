#!/usr/bin/env python

"""
CCO using SC cell.
"""

use_mpi = True
if use_mpi:
    from mpi4pyscf.tools import mpi
import os, sys
import time
import numpy as np
from pyscf.pbc import gto, scf, df, dft, cc
from pyscf import lib
from pyscf.pbc.lib import chkfile

from libdmet.system import lattice
from libdmet.utils.misc import mdot, max_abs, search_idx1d
from libdmet.utils.iotools import read_poscar
import libdmet.utils.logger as log
from libdmet.basis import trans_1e

from libdmet.lo import iao, make_lo, lowdin
from libdmet.lo.lowdin import check_orthogonal, check_orthonormal, check_span_same_space
from libdmet.mean_field import pbc_helper as pbc_hp
from libdmet.mean_field import mfd, vcor
from libdmet.solver import scf_solver, cc_solver
from libdmet.dmet import rdmet, udmet, gdmet

np.set_printoptions(4, linewidth=1000, suppress=True)
log.verbose = "DEBUG1"
lib.logger.TIMER_LEVEL = 5

start = time.time()

cell = read_poscar(fname="./CCO-2x2-frac.vasp")
cell.basis = {'Cu1':'cc-pvdz.dat', 'Cu2': 'cc-pvdz.dat', 'O1': 'cc-pvdz.dat',
              'Ca':'cc-pvdz.dat'}
cell.pseudo = {'Cu1': 'gth-pbe-q19', 'Cu2': 'gth-pbe-q19', 'O1': 'gth-pbe',
               'Ca': 'gth-pbe'}


kmesh = [4, 4, 2]
cell.spin = 0
cell.verbose = 5
cell.max_memory = 900000
cell.precision = 1e-14
cell.build()

nelec_dop = 50
nkpts = np.prod(kmesh)
nelec0 = int(cell.nelectron)
doping = nelec_dop / (nkpts * 4)
cell.nelectron = (nelec0 * nkpts - nelec_dop) / nkpts
cell.build()

print ("doping:             ", doping)
print ("doping nelec:       ", nelec_dop)
print ("total electrons:    ", cell.tot_electrons(nkpts))
print ("nelec per cell:     ", cell.nelectron)
print ("total spin:         ", cell.spin)
print ("atom charges:       ", cell.atom_charges())
print ("atom charges total: ", cell.atom_charges().sum())

cell.space_group_symmetry = True
cell.symmorphic = True
cell.build()

kpts_symm = cell.make_kpts(kmesh, with_gamma_point=True, wrap_around=True, 
                           space_group_symmetry=True, time_reversal_symmetry=True)

Lat = lattice.Lattice(cell, kmesh)
kpts = Lat.kpts
nao = Lat.nao
nkpts = Lat.nkpts

gdf_fname = '/global/cfs/cdirs/m4010/zhcui/proj/cuprate_doping/pressure/CCO/gth-cc_new_2/ab_0.000_cmem/gdf_ints_CCO_442.h5'
gdf = df.GDF(cell, kpts)
gdf._cderi_to_save = gdf_fname
gdf.use_mpi = use_mpi

kmf = scf.KUKS(cell, kpts_symm, exxdiv=None).density_fit()
kmf.grids.level = 5
kmf.xc = "pbe0"
kmf.init_guess_breaksym = False

from pyscf.data.nist import HARTREE2EV
sigma = 0.2 / HARTREE2EV
beta = 1000.0
kmf = pbc_hp.smearing_(kmf, sigma=sigma, method="fermi", tol=1e-12)
kmf.with_df = gdf
kmf.with_df._cderi = gdf_fname
kmf.conv_tol = 1e-7
chk_fname = './CCO_UPBE0.chk'
kmf.chkfile = chk_fname
kmf.diis_space = 15
kmf.max_cycle = 1

# kmf_no_symm = pbc_hp.kmf_symm_(kmf)
# dm0 = kmf_no_symm.get_init_guess(key='atom')
# dm0 = dm0[kpts_symm.ibz2bz]
# print("dm0.shape", dm0.shape)

# kmf.kernel(dm0=dm0)
# kmf = pbc_hp.kmf_symm_(kmf)
# np.save("ovlp_mf.npy", kmf.get_ovlp())
# np.save("hcore_mf.npy", kmf.get_hcore())
# assert False

###########################################################################

data = chkfile.load("../CCO_UPBE0.chk", "scf")
kmf.__dict__.update(data)

kmf = pbc_hp.kmf_symm_(kmf)

kmf = Lat.symmetrize_kmf(kmf)
ovlp = Lat.symmetrize_lo(np.load("../ovlp.npy"))
hcore = Lat.symmetrize_lo(np.load("../hcore.npy"))
rdm1 = np.asarray(kmf.make_rdm1())
rdm1 = Lat.symmetrize_lo(rdm1)

print ("rdm1.shape", rdm1.shape) # (2, 32, 308, 308)

kmf.get_hcore = lambda *args: hcore
kmf.get_ovlp = lambda *args: ovlp

mo_coeff = np.asarray(kmf.mo_coeff)
mo_occ = np.asarray(kmf.mo_occ)

# ***********************
# Construct IAO
# ***********************
minao = 'ccpvdz-atom-iao.dat'
basis_core = "ccpvdz-atom-iao-core.dat"
basis_val = "ccpvdz-atom-iao-val.dat"

pmol = iao.reference_mol(cell, minao=minao)
pmol_core, pmol_val = iao.build_pmol_core_val(pmol, basis_core, basis_val)

C_ao_iao, idx_core, idx_val, idx_virt = \
        make_lo.get_iao(kmf, minao=minao,
                        pmol_core=pmol_core, pmol_val=pmol_val,
                        tol=1e-9, full_return=True)

ncore = len(idx_core)
nval = len(idx_val)
nvirt = len(idx_virt)
print ("ncore: ", ncore)
print ("nval: ", nval)
print ("nvirt: ", nvirt)
print ("C_ao_iao shape: ", C_ao_iao.shape)


# ***********************
# Prepare (Frozen Core)
# ***********************
(gkmf, E0, hcore_hf_add,
    ghcore_xcore, govlp, grdm1_core, grdm1_xcore,
    gvj_xcore, gvk_xcore, gvxc_ao, exc,
    C_ao_lo_xcore, idx_val_xcore, idx_virt_xcore, labels_xcore,
    nelec_xcore, mu) = \
            gdmet.prepare_integral(kmf, C_ao_iao, idx_core, idx_val, idx_virt, beta=beta)


# ***********************
# Prepare the indices of fitting orbitals
# ***********************
idx_spec_val = iao.get_idx_each(cell, labels=labels_xcore[idx_val_xcore], kind='atom')
Cu_idx_val = list(idx_spec_val["Cu1"]) + list(idx_spec_val["Cu2"])
O1_idx_val = idx_spec_val["O1"]
Cu_O1_idx_val = idx_val_xcore[:len(Cu_idx_val) + len(O1_idx_val)]
ion_idx_val = idx_val_xcore[len(Cu_idx_val) + len(O1_idx_val):]

idx_spec_virt = iao.get_idx_each(cell, labels=labels_xcore[idx_virt_xcore], kind='atom')
Cu_idx_virt = list(idx_spec_virt["Cu1"]) + list(idx_spec_virt["Cu2"])
O1_idx_virt = idx_spec_virt["O1"]
# Cu_O1_idx_virt = idx_virt_xcore[:len(Cu_idx_virt) + len(O1_idx_virt)]
ion_idx_virt = idx_virt_xcore[len(Cu_idx_virt) + len(O1_idx_virt):]

# freeze virtual orbitals
frozen = ["Cu1 4f", "Cu2 4f",
          "O1 3d", "O2 3d"]
Cu_O1_idx_virt = []
idx_spec = iao.get_idx_each(cell, labels=labels_xcore, kind='atom')
Cu_idx = list(idx_spec["Cu1"]) + list(idx_spec["Cu2"])
O1_idx = idx_spec["O1"]
Cu_O1_idx = np.append(Cu_idx, O1_idx)
for idx in Cu_O1_idx:
    if not idx in Cu_O1_idx_val:
        fr = np.any([lab in labels_xcore[idx] for lab in frozen])
        if fr:
            pass
        else:
            Cu_O1_idx_virt.append(idx)

# vcor only fit t_band
idx_all = iao.get_idx_each(cell, labels=labels_xcore, kind='all')
idx_x = iao.get_idx_each(cell, labels=labels_xcore, kind='atom nlm')
Cu_3band = list(idx_x["Cu1 3dx2-y2"]) + list(idx_x["Cu2 3dx2-y2"])

tband_idx = Cu_3band \
          + list(idx_all["4 O1 2py   "])     + list(idx_all["5 O1 2px   "])\
          + list(idx_all["6 O1 2py   "])     + list(idx_all["7 O1 2px   "])\
          + list(idx_all["8 O1 2px   "])     + list(idx_all["9 O1 2py   "])\
          + list(idx_all["10 O1 2px   "])     + list(idx_all["11 O1 2py   "])


# ***********************
# Separate Cu_O1 and Ca
# ***********************
NCORE = ncore
NLO = nval + nvirt
nlo = NLO

cell_lo = cell.copy()
cell_lo.nelectron = cell_lo.nelectron - NCORE * 2
cell_lo.nao_nr = lambda *args: NLO
cell_lo.build()
print ("original nelectron (LO)", cell.nelectron)

Lat = lattice.Lattice(cell_lo, kmesh)
Lat_ion = lattice.Lattice(cell_lo, kmesh)

# SY TODO: warning: imp_idx is beyond the first cell nlo
Lat.build(idx_val=Cu_O1_idx_val, idx_virt=Cu_O1_idx_virt, labels=labels_xcore)
Lat_ion.build(idx_val=ion_idx_val, idx_virt=ion_idx_virt, labels=labels_xcore)

# print("Lat.imp_idx", Lat.imp_idx)
# print("Lat_ion.imp_idx", Lat_ion.imp_idx)

print ("tband_idx", tband_idx)
imp_idx_fit = search_idx1d(tband_idx, Lat.imp_idx)
print("imp_idx_fit", imp_idx_fit)


# ***********************
# Vcor
# ***********************
# ZHC NOTE here we use a symmetrized vcor (C2h symmetry)

from pyscf import gto as mol_gto
mol = mol_gto.Mole()
mol.atom = \
"""
Cu1   0.250000000000000    0.250000000000000    0.500000000000000 
Cu1   0.750000000000000    0.750000000000000    0.500000000000000 
Cu2   0.250000000000000    0.750000000000000    0.500000000000000 
Cu2   0.750000000000000    0.250000000000000    0.500000000000000 
O1    0.250000000000000    0.500000000000000    0.500000000000000 
O1    0.500000000000000    0.750000000000000    0.500000000000000 
O1    0.750000000000000    0.500000000000000    0.500000000000000 
O1    0.500000000000000    0.250000000000000    0.500000000000000 
O2    0.000000000000000    0.250000000000000    0.500000000000000 
O2    0.250000000000000    1.000000000000000    0.500000000000000 
O2    1.000000000000000    0.750000000000000    0.500000000000000 
O2    0.750000000000000    0.000000000000000    0.500000000000000 
"""
mol.symmetry = True
mol.basis = {"Cu1": "def2-svp@1d", "Cu2": "def2-svp@1d", 
             "O1": "def2-svp@1p", "O2": "def2-svp@1p"}
mol.build()

Cu1_3d = mol.search_ao_label("Cu1 3dx2-y2")
Cu2_3d = mol.search_ao_label("Cu2 3dx2-y2")

px_idx = []
for i in [5, 7, 8, 10]:
    px_idx.append(mol.search_ao_label("%d O. 2px"%i)[0])

py_idx = []
for i in [4, 6, 9, 11]:
    py_idx.append(mol.search_ao_label("%d O. 2py"%i)[0])

symm_idx = np.sort(np.hstack((*Cu1_3d, *Cu2_3d, *px_idx, *py_idx)), kind='mergesort')
Ca = lattice.get_symm_orb(mol, symm_idx)

Cu_O1_vcor = vcor.UBVcorSymm(nao=nlo, C_symm=Ca, imp_idx=tband_idx, bogo_only=True)

# initial d-wave guess
vcor_mat = Cu_O1_vcor.get()
pairs_Cu_nn, dis_Cu_nn = Lat.get_bond_pairs(length_range=[3.5, 4.5],
                                            bond_type=[('Cu1', 'Cu2')], triu=False)
pairs_O1_nn, dis_O1_nn = Lat.get_bond_pairs(length_range=[3.5, 4.5],
                                            bond_type=[('O1', 'O1')], triu=False)
rand = 1e-3
sign = 1
coords = Lat.frac_coords()

for i, j in pairs_Cu_nn:
    coord_i = coords[i]
    coord_j = coords[j]
    dis = np.abs(coord_i - coord_j)
    idxi = tband_idx[i]
    idxj = tband_idx[j]

    if abs(dis[0] - 0.5) < 1e-5:
        vcor_mat[2][idxi, idxj] = rand * sign
    elif abs(dis[1] - 0.5) < 1e-5:
        vcor_mat[2][idxi, idxj] = -rand * sign
    else:
        raise ValueError

Cu_O1_vcor.assign(vcor_mat)
vcor_mat = None

# ***********************
# DMET
# ***********************
restricted = False
ghf = True
solver = cc_solver.CCSolver(restricted=restricted, 
                            ghf=ghf, 
                            restart=True, 
                            verbose=cell.verbose+1,
                            max_memory=cell.max_memory, 
                            beta=beta, 
                            tol=1e-6, tol_normt=1e-4, 
                            level_shift=0.01,
                            alpha=1.0, 
                            diis_space=10, 
                            krylov=True, 
                            approx_l=False, 
                            restart_ovlp_tol=0.2,
                            use_mpi=True, 
                            )
solver_argss = [["C_lo_eo", ["scf_max_cycle", 0]]] #

# SY NOTE: only calculate CuO
mydmet = gdmet.GDMET(Lat, gkmf, 
                     solver, 
                     C_ao_lo=C_ao_lo_xcore, 
                     vcor=Cu_O1_vcor, 
                     solver_argss=solver_argss, 
                     h0=E0, hcore_hf_add=hcore_hf_add,
                     govlp_ao=govlp, ghcore_ao=ghcore_xcore, grdm1_ao=grdm1_xcore,
                     gvj_ao=gvj_xcore, gvk_ao=gvk_xcore, gvxc_ao=gvxc_ao, 
                     nelec=nelec_xcore, 
                     mu=mu,
                     grdm1_core_ao=grdm1_core,
                     load_mu_history=True,
                     )

mydmet.dump_flags()
mydmet.type_bath = 'svd'
mydmet.beta = beta
mydmet.mu_solver_args["init_step"] = 0.1
mydmet.mu_solver_args["max_step"] = 1.0
mydmet.mu_solver_args["tol"] = 1e+5

# fit
mydmet.imp_idx = imp_idx_fit
mydmet.fit_method = "BFGS"
mydmet.ytol = 1e-7
mydmet.gtol = 1e-4
mydmet.fit_max_cycle = 300
mydmet.fit_kwargs = {"max_stepsize": 0.001}
mydmet.charge_self_consistency = False # True

mydmet.calc_e_tot = False
mydmet.tol_energy = 1e+3

mydmet.chkfile = "fdmet.h5"

# SY NOTE: for pbe0
mydmet.fit_mu_nelec_emb = True

mydmet.kernel()
mydmet._finalize()

print ("resulting rdm1, shape=", mydmet.rdm1_glob.shape)
print (mydmet.rdm1_glob[0])

