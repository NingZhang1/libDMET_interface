#!/usr/bin/env python

############ sys module ############

import copy
import numpy as np
import scipy
import ctypes, sys

############ pyscf module ############

from pyscf import lib
from pyscf.pbc.gto import Cell
from pyscf.pbc import tools
from pyscf.gto.mole import *
libpbc = lib.load_library('libpbc')

import pyscf.pbc.df.isdf.isdf_linear_scaling as ISDF

if __name__ == '__main__':
    
    C = 15
    from pyscf.lib.parameters import BOHR
    from pyscf.pbc.df.isdf.isdf_tools_cell import build_supercell, build_supercell_with_partition
    import pyscf.pbc.gto as pbcgto
    
    verbose = 10
        
    cell   = pbcgto.Cell()
    boxlen = 3.5668
    cell.a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen*6]])
    prim_a = np.array([[boxlen,0.0,0.0],[0.0,boxlen,0.0],[0.0,0.0,boxlen*6]])
    atm = [
        ['C', (0.     , 0.     , 0.    )],
        ['C', (0.8917 , 0.8917 , 0.8917)],
        ['C', (1.7834 , 1.7834 , 0.    )],
        ['C', (2.6751 , 2.6751 , 0.8917)],
        ['C', (1.7834 , 0.     , 1.7834)],
        ['C', (2.6751 , 0.8917 , 2.6751)],
        ['C', (0.     , 1.7834 , 1.7834)],
        ['C', (0.8917 , 2.6751 , 2.6751)],
    ] 
    KE_CUTOFF = 70
    # basis = 'unc-gth-cc-tzvp'
    # pseudo = "gth-hf"  
    basis = 'gth-dzvp'
    pseudo = "gth-pade"   
    prim_cell = build_supercell(atm, prim_a, Ls = [1,1,1], ke_cutoff=KE_CUTOFF, basis=basis, pseudo=pseudo)    
    prim_partition = [[0,1],[2,3],[4,5],[6,7]]
    
    prim_mesh = prim_cell.mesh
    Ls = [1, 1, 1]
    # Ls = [2, 2, 2]
    Ls = np.array(Ls, dtype=np.int32)
    mesh = [Ls[0] * prim_mesh[0], Ls[1] * prim_mesh[1], Ls[2] * prim_mesh[2]]
    mesh = np.array(mesh, dtype=np.int32)
    
    cell, group_partition = build_supercell_with_partition(atm, prim_a, mesh=mesh, 
                                                     Ls=Ls,
                                                     basis=basis, 
                                                     pseudo=pseudo,
                                                     partition=prim_partition, ke_cutoff=KE_CUTOFF, verbose=verbose)
    print("group_partition = ", group_partition)
    
    pbc_isdf_info = ISDF.PBC_ISDF_Info_Quad(cell, with_robust_fitting=True, aoR_cutoff=1e-8, direct=False, use_occ_RI_K=False, rela_cutoff_QRCP=3e-3)
    pbc_isdf_info.verbose = 10
    pbc_isdf_info.build_IP_local(c=C, m=5, group=group_partition)
    pbc_isdf_info.build_auxiliary_Coulomb()
    
    from pyscf.pbc import scf
    
    mf = scf.RHF(cell)
    pbc_isdf_info.direct_scf = mf.direct_scf
    mf.with_df = pbc_isdf_info
    mf.max_cycle = 25
    mf.conv_tol = 1e-7
    mf.kernel()
    
    from pyscf.pbc import scf
    
    mf = scf.RHF(cell)
    # pbc_isdf_info.direct_scf = mf.direct_scf
    # mf.with_df = pbc_isdf_info
    mf.max_cycle = 25
    mf.conv_tol = 1e-7
    mf.kernel()