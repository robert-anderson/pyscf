import unittest
import itertools
import numpy as np

from pyscf.lib import finger
from pyscf.pbc import gto as pbcgto
from pyscf.pbc import scf as pbcscf
from pyscf.pbc import df as pbc_df

import pyscf.cc
import pyscf.pbc.cc as pbcc
import make_test_cell
from pyscf.pbc.lib import kpts_helper
import pyscf.pbc.cc.kccsd as kccsd
import pyscf.pbc.cc.eom_kccsd_ghf as kccsd_ghf

cell = pbcgto.Cell()
cell.atom = '''
He 0.000000000000   0.000000000000   0.000000000000
He 1.685068664391   1.685068664391   1.685068664391
'''
cell.basis = [[0, (1., 1.)], [0, (.5, 1.)]]
cell.a = '''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000'''
cell.unit = 'B'
#cell.verbose = 7
#cell.output = '/dev/null'
cell.build()
nmp = [1,1,2]
thresh = 1e-8

def get_idx_r2(nkpts,nocc,nvir,ki,kj,i,j,a):
    o1 = nvir
    o2 = nocc*o1
    o3 = nocc*o2
    o4 = nkpts*o3
    return ki*o4 + ki*o3 + i*o2 + i*o1 + a

def get_ip_identity(nocc,nvir,nkpts,I):
    indices = []

    for i in range(nocc):
        indices.append(i)
    count = nocc

    for ki, kj, i, j, a in itertools.product(range(nkpts),range(nkpts),range(nocc),range(nocc),range(nvir)):
        if j < i:
            r1 = np.zeros(nocc,dtype=complex)
            r2 = np.zeros((nkpts,nkpts,nocc,nocc,nvir),dtype=complex)
            r2[ki,kj,i,j,a] = 1.0
            r2[kj,ki,j,i,a] = -1.0
            I[:,count] = kccsd_ghf.amplitudes_to_vector_ip(r1,r2)
            indices.append(count)
        count = count + 1

    return indices

def get_ea_identity(nocc,nvir,nkpts,I,kshift,kconserv):
    indices = []

    for a in range(nvir):
        indices.append(a)
    count = nvir

    for ki, ka, i, a, b in itertools.product(range(nkpts),range(nkpts),range(nocc),range(nvir),range(nvir)):
        if b < a:
            kb = kconserv[kshift,ka,ki]
            r1 = np.zeros(nvir,dtype=complex)
            r2 = np.zeros((nkpts,nkpts,nocc,nvir,nvir),dtype=complex)
            r2[ki,ka,i,a,b] = 1.0
            r2[ki,kb,i,b,a] = -1.0
            I[:,count] = kccsd_ghf.amplitudes_to_vector_ea(r1,r2)
            indices.append(count)
        count = count + 1

    return indices

class TestHe(unittest.TestCase):
    def _test_ip_diag(self,cc):
        eom = kccsd_ghf.EOMIP(cc)
        imds = eom.make_imds()
        nkpts, nocc, nvir = imds.t1.shape
        diag = kccsd_ghf.ipccsd_diag(eom,0,imds=imds)
        
        I = np.zeros((diag.shape[0],diag.shape[0]),dtype=complex)
        I[:nocc,:nocc] = np.identity(nocc,dtype=complex)
        indices = get_ip_identity(nocc,nvir,nkpts,I)
        H = np.zeros((I.shape[0],len(indices)),dtype=complex)
        for j,idx in enumerate(indices):
            H[:,j] = kccsd_ghf.ipccsd_matvec(eom,I[:,idx],0,imds=imds)

        diag_ref = np.zeros(len(indices),dtype=complex)
        diag_out = np.zeros(len(indices),dtype=complex)
        for j,idx in enumerate(indices):
            diag_ref[j] = H[idx,j]
            diag_out[j] = diag[idx]
        diff = np.linalg.norm(diag_ref - diag_out)
        self.assertTrue(abs(diff) < thresh,"Difference in IP diag: {}".format(diff))

    def _test_ea_diag(self,cc):
       eom = kccsd_ghf.EOMEA(cc)
       imds = eom.make_imds()
       nkpts, nocc, nvir = imds.t1.shape
       diag = kccsd_ghf.eaccsd_diag(eom,0,imds=imds)

       I = np.zeros((diag.shape[0],diag.shape[0]),dtype=complex)
       I[:nvir,:nvir] = np.identity(nvir,dtype=complex)
       indices = get_ea_identity(nocc,nvir,nkpts,I,0,imds.kconserv)
       H = np.zeros((I.shape[0],len(indices)),dtype=complex)
       for j,idx in enumerate(indices):
           H[:,j] = kccsd_ghf.eaccsd_matvec(eom,I[:,idx],0,imds=imds)

       diag_ref = np.zeros(len(indices),dtype=complex)
       diag_out = np.zeros(len(indices),dtype=complex)
       for j,idx in enumerate(indices):
           diag_ref[j] = H[idx,j]
           diag_out[j] = diag[idx]
       diff = np.linalg.norm(diag_ref - diag_out)
       self.assertTrue(abs(diff) < thresh,"Difference in EA diag: {}".format(diff))

    def test_he_112_diag(self):
        kpts = cell.make_kpts([1,1,2])
        kmf = pbcscf.KGHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        cc = kccsd.KGCCSD(kmf)
        Ecc = cc.kernel()[0]

        self._test_ip_diag(cc)
        self._test_ea_diag(cc)

    def test_he_212_diag(self):
        kpts = cell.make_kpts([2,1,2])
        kmf = pbcscf.KGHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        cc = kccsd.KGCCSD(kmf)
        Ecc = cc.kernel()[0]

        self._test_ip_diag(cc)
        self._test_ea_diag(cc)

    def test_he_131_diag(self):
        kpts = cell.make_kpts([1,3,1])
        kmf = pbcscf.KGHF(cell, kpts, exxdiv=None)
        Escf = kmf.scf()
        cc = kccsd.KGCCSD(kmf)
        Ecc = cc.kernel()[0]

        self._test_ip_diag(cc)
        self._test_ea_diag(cc)

    def test_supercell_vs_kpt(self):
        # Running HF and CCSD with 1x1x2 Monkhorst-Pack k-point mesh
        kmf = pbcscf.KGHF(cell, kpts=cell.make_kpts(nmp), exxdiv=None)
        kmf.kernel()
        mycc = pbcc.KGCCSD(kmf)
        mycc.conv_tol = 1e-12
        mycc.conv_tol_normt = 1e-10
        ecc2, t1, t2 = mycc.kernel()
        ecc_ref = -0.01044680113334205
        print(ecc2)
        self.assertAlmostEqual(abs(ecc_ref - ecc2), 0, 10)


if __name__ == '__main__':
    unittest.main()
