#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

# See also JCP, 90, 1752

import time
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf import ao2mo
from pyscf.cc import ccsd
from pyscf.cc import ccsd_rdm
from pyscf import grad

def IX_intermediates(cc, t1, t2, l1, l2, eris=None, d1=None, d2=None):
    if eris is None:
# Note eris are in Chemist's notation
        eris = _ERIS(cc, cc._scf.mo_coeff)
    if d1 is None:
        d1 = gamma1_intermediates(cc, t1, t2, l1, l2)
    if d2 is None:
# Note gamma2 are in Chemist's notation
        d2 = gamma2_intermediates(cc, t1, t2, l1, l2)
    doo, dov, dvo, dvv = d1
    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    nocc, nvir = dov.shape

# Note Ioo is not hermitian
    Ioo  =(numpy.einsum('jakb,iakb->ij', dovov, eris.ovov)
         + numpy.einsum('kbja,iakb->ij', dovov, eris.ovov))
    Ioo +=(numpy.einsum('jabk,iakb->ij', dovvo, eris.ovov)
         + numpy.einsum('kbaj,iakb->ij', dovvo, eris.ovov)
         + numpy.einsum('jkab,ikab->ij', doovv, eris.oovv)
         + numpy.einsum('kjba,ikab->ij', doovv, eris.oovv))
    Ioo +=(numpy.einsum('jmlk,imlk->ij', doooo, eris.oooo) * 2
         + numpy.einsum('mjkl,imlk->ij', doooo, eris.oooo) * 2)
    Ioo +=(numpy.einsum('jlka,ilka->ij', dooov, eris.ooov)
         + numpy.einsum('klja,klia->ij', dooov, eris.ooov))
    Ioo += numpy.einsum('abjc,abic->ij', dvvov, eris.vvov)
    Ioo += numpy.einsum('ljka,lika->ij', dooov, eris.ooov)
    Ioo *= -1

# Note Ivv is not hermitian
    Ivv  =(numpy.einsum('ibjc,iajc->ab', dovov, eris.ovov)
         + numpy.einsum('jcib,iajc->ab', dovov, eris.ovov))
    Ivv +=(numpy.einsum('jcbi,iajc->ab', dovvo, eris.ovov)
         + numpy.einsum('ibcj,iajc->ab', dovvo, eris.ovov)
         + numpy.einsum('jibc,jiac->ab', doovv, eris.oovv)
         + numpy.einsum('ijcb,jiac->ab', doovv, eris.oovv))
    Ivv +=(numpy.einsum('bced,aced->ab', dvvvv, eris.vvvv) * 2
         + numpy.einsum('cbde,aced->ab', dvvvv, eris.vvvv) * 2)
    Ivv +=(numpy.einsum('dbic,daic->ab', dvvov, eris.vvov)
         + numpy.einsum('dcib,dcia->ab', dvvov, eris.vvov))
    Ivv += numpy.einsum('bcid,acid->ab', dvvov, eris.vvov)
    Ivv += numpy.einsum('jikb,jika->ab', dooov, eris.ooov)
    Ivv *= -1

    Ivo  =(numpy.einsum('kajb,kijb->ai', dovov, eris.ooov)
         + numpy.einsum('kbja,jikb->ai', dovov, eris.ooov))
    Ivo +=(numpy.einsum('acbd,icbd->ai', dvvvv, eris.ovvv) * 2
         + numpy.einsum('cadb,icbd->ai', dvvvv, eris.ovvv) * 2)
    Ivo +=(numpy.einsum('jbak,jbik->ai', dovvo, eris.ovoo)
         + numpy.einsum('kabj,jbik->ai', dovvo, eris.ovoo)
         + numpy.einsum('jkab,jkib->ai', doovv, eris.ooov)
         + numpy.einsum('kjba,jkib->ai', doovv, eris.ooov))
    Ivo +=(numpy.einsum('dajc,dijc->ai', dvvov, eris.voov)
         + numpy.einsum('dcja,dcji->ai', dvvov, eris.vvoo))
    Ivo += numpy.einsum('abjc,ibjc->ai', dvvov, eris.ovov)
    Ivo += numpy.einsum('jlka,jlki->ai', dooov, eris.oooo)
    Ivo *= -1

    if not (cc.frozen is None or cc.frozen is 0):
        mo_e_o = eris.fock.diagonal()[:nocc]
        mo_e_v = eris.fock.diagonal()[nocc:]
        OA, VA, OF, VF = index_frozen_active(cc)
        doo = doo.copy()
        dvv = dvv.copy()
        doo[OF[:,None],OA] = Ioo[OF[:,None],OA] / lib.direct_sum('i-j->ij', mo_e_o[OF], mo_e_o[OA])
        doo[OA[:,None],OF] = Ioo[OA[:,None],OF] / lib.direct_sum('i-j->ij', mo_e_o[OA], mo_e_o[OF])
        dvv[VF[:,None],VA] = Ivv[VF[:,None],VA] / lib.direct_sum('a-b->ab', mo_e_v[VF], mo_e_v[VA])
        dvv[VA[:,None],VF] = Ivv[VA[:,None],VF] / lib.direct_sum('a-b->ab', mo_e_v[VA], mo_e_v[VF])

    Xvo  =(numpy.einsum('kj,kjai->ai', doo+doo.T, eris.oovo) * 2
         - numpy.einsum('kj,ajki->ai', doo+doo.T, eris.vooo))
    Xvo +=(numpy.einsum('cb,cbai->ai', dvv+dvv.T, eris.vvvo) * 2
         - numpy.einsum('cb,ibca->ai', dvv+dvv.T, eris.ovvv))
    Xvo +=(numpy.einsum('icjb,acjb->ai', dovov, eris.vvov)
         + numpy.einsum('jcib,abjc->ai', dovov, eris.vvov))
    Xvo +=(numpy.einsum('iklj,aklj->ai', doooo, eris.vooo) * 2
         + numpy.einsum('kijl,aklj->ai', doooo, eris.vooo) * 2)
    Xvo +=(numpy.einsum('ibcj,abcj->ai', dovvo, eris.vvvo)
         + numpy.einsum('jcbi,abcj->ai', dovvo, eris.vvvo)
         + numpy.einsum('ijcb,ajcb->ai', doovv, eris.vovv)
         + numpy.einsum('jibc,ajcb->ai', doovv, eris.vovv))
    Xvo +=(numpy.einsum('ijkb,ajkb->ai', dooov, eris.voov)
         + numpy.einsum('kjib,kjab->ai', dooov, eris.oovv))
    Xvo += numpy.einsum('dbic,dbac->ai', dvvov, eris.vvvv)
    Xvo += numpy.einsum('jikb,jakb->ai', dooov, eris.ovov)
    Xvo += Ivo
    return Ioo, Ivv, Ivo, Xvo


def response_dm1(cc, t1, t2, l1, l2, eris=None, IX=None):
    from pyscf.scf import cphf
    if eris is None:
# Note eris are in Chemist's notation
        eris = _ERIS(cc, cc.mo_coeff)
    if IX is None:
        IX = IX_intermediates(cc, t1, t2, l1, l2, eris)
    Ioo, Ivv, Ivo, Xvo = IX
    nvir, nocc = Ivo.shape
    nmo = nocc + nvir
    def fvind(x):
        x = x.reshape(Xvo.shape)
        if eris is None:
            mo_coeff = cc.mo_coeff
            dm = reduce(numpy.dot, (mo_coeff[:,nocc:], x, mo_coeff[:,:nocc].T))
            v = reduce(numpy.dot, (mo_coeff[:,nocc:].T, cc._scf.get_veff(mol, dm),
                                   mo_coeff[:,:nocc]))
        else:
            v  = numpy.einsum('iajb,bj->ai', eris.ovov, x) * 4
            v -= numpy.einsum('abji,bj->ai', eris.vvoo, x)
            v -= numpy.einsum('ibja,bj->ai', eris.ovov, x)
        return v
    mo_energy = eris.fock.diagonal()
    mo_occ = numpy.zeros_like(mo_energy)
    mo_occ[:nocc] = 2
    dvo = cphf.solve(fvind, mo_energy, mo_occ, Xvo, max_cycle=30)[0]
    dm1 = numpy.zeros((nmo,nmo))
    dm1[nocc:,:nocc] = dvo
    dm1[:nocc,nocc:] = dvo.T
    return dm1

def kernel(cc, t1, t2, l1, l2, eris=None):
    if eris is None:
        eris = _ERIS(cc, cc.mo_coeff)
    mol = cc.mol
    mo_coeff = cc.mo_coeff
    mo_energy = cc._scf.mo_energy
    nao, nmo = mo_coeff.shape
    nocc = numpy.count_nonzero(cc.mo_occ > 0)
    nvir = nmo - nocc
    mo_e_o = mo_energy[:nocc]
    mo_e_v = mo_energy[nocc:]
    with_frozen = not (cc.frozen is None or cc.frozen is 0)

    d1 = gamma1_intermediates(cc, t1, t2, l1, l2)
    d2 = gamma2_intermediates(cc, t1, t2, l1, l2)
    IX = IX_intermediates(cc, t1, t2, l1, l2, eris, d1, d2)
    doo, dov, dvo, dvv = d1
    Ioo, Ivv, Ivo, Xvo = IX

    if with_frozen:
        OA, VA, OF, VF = index_frozen_active(cc)
        doo[OF[:,None],OA] = Ioo[OF[:,None],OA] / lib.direct_sum('i-j->ij', mo_e_o[OF], mo_e_o[OA])
        doo[OA[:,None],OF] = Ioo[OA[:,None],OF] / lib.direct_sum('i-j->ij', mo_e_o[OA], mo_e_o[OF])
        dvv[VF[:,None],VA] = Ivv[VF[:,None],VA] / lib.direct_sum('a-b->ab', mo_e_v[VF], mo_e_v[VA])
        dvv[VA[:,None],VF] = Ivv[VA[:,None],VF] / lib.direct_sum('a-b->ab', mo_e_v[VA], mo_e_v[VF])
    dm1 = response_dm1(cc, t1, t2, l1, l2, eris, IX)
    dm1[:nocc,:nocc] = doo + doo.T
    dm1[nocc:,nocc:] = dvv + dvv.T

    im1 = numpy.zeros_like(dm1)
    im1[:nocc,:nocc] = Ioo
    im1[nocc:,nocc:] = Ivv
    im1[nocc:,:nocc] = Ivo
    im1[:nocc,nocc:] = Ivo.T

    h1 =-(mol.intor('cint1e_ipkin_sph', comp=3)
         +mol.intor('cint1e_ipnuc_sph', comp=3))
    s1 =-mol.intor('cint1e_ipovlp_sph', comp=3)
    zeta = lib.direct_sum('i-j->ij', mo_energy, mo_energy)
    eri1 = mol.intor('int2e_ip1', comp=3).reshape(3,nao,nao,nao,nao)
    eri1 = numpy.einsum('xipkl,pj->xijkl', eri1, mo_coeff)
    eri1 = numpy.einsum('xijpl,pk->xijkl', eri1, mo_coeff)
    eri1 = numpy.einsum('xijkp,pl->xijkl', eri1, mo_coeff)

    dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
    dvvov = dovvv.transpose(2,3,0,1)
    dvvvv = ao2mo.restore(1, dvvvv, nvir).reshape((nvir,)*4)
    g0 = ao2mo.restore(1, ao2mo.full(mol, mo_coeff), nmo)

    de = numpy.empty((mol.natm,3))
    for k,(sh0, sh1, p0, p1) in enumerate(mol.offset_nr_by_atom()):
        mol.set_rinv_origin(mol.atom_coord(k))
        vrinv = -mol.atom_charge(k) * mol.intor('cint1e_iprinv_sph', comp=3)

# 2e AO integrals dot 2pdm
        de2 = numpy.zeros(3)
        for i in range(3):
            g1 = numpy.einsum('pjkl,pi->ijkl', eri1[i,p0:p1], mo_coeff[p0:p1])
            g1 = g1 + g1.transpose(1,0,2,3)
            g1 = g1 + g1.transpose(2,3,0,1)
            g1 *= -1
            hx =(numpy.einsum('pq,pi,qj->ij', h1[i,p0:p1], mo_coeff[p0:p1], mo_coeff)
               + reduce(numpy.dot, (mo_coeff.T, vrinv[i], mo_coeff)))
            hx = hx + hx.T
            sx = numpy.einsum('pq,pi,qj->ij', s1[i,p0:p1], mo_coeff[p0:p1], mo_coeff)
            sx = sx + sx.T
            fij =(hx[:nocc,:nocc]
                  - numpy.einsum('ij,j->ij', sx[:nocc,:nocc], mo_e_o) * .5
                  - numpy.einsum('ij,i->ij', sx[:nocc,:nocc], mo_e_o) * .5
                  - numpy.einsum('kl,ijlk->ij', sx[:nocc,:nocc],
                                 g0[:nocc,:nocc,:nocc,:nocc]) * 2
                  + numpy.einsum('kl,iklj->ij', sx[:nocc,:nocc],
                                 g0[:nocc,:nocc,:nocc,:nocc])
                  + numpy.einsum('ijkk->ij', g1[:nocc,:nocc,:nocc,:nocc]) * 2
                  - numpy.einsum('ikkj->ij', g1[:nocc,:nocc,:nocc,:nocc]))

            fab =(hx[nocc:,nocc:]
                  - numpy.einsum('ij,j->ij', sx[nocc:,nocc:], mo_e_v) * .5
                  - numpy.einsum('ij,i->ij', sx[nocc:,nocc:], mo_e_v) * .5
                  - numpy.einsum('kl,ijlk->ij', sx[:nocc,:nocc],
                                 g0[nocc:,nocc:,:nocc,:nocc]) * 2
                  + numpy.einsum('kl,iklj->ij', sx[:nocc,:nocc],
                                 g0[nocc:,:nocc,:nocc,nocc:])
                  + numpy.einsum('ijkk->ij', g1[nocc:,nocc:,:nocc,:nocc]) * 2
                  - numpy.einsum('ikkj->ij', g1[nocc:,:nocc,:nocc,nocc:]))

            fai =(hx[nocc:,:nocc]
                  - numpy.einsum('ai,i->ai', sx[nocc:,:nocc], mo_e_o)
                  - numpy.einsum('kl,ijlk->ij', sx[:nocc,:nocc],
                                 g0[nocc:,:nocc,:nocc,:nocc]) * 2
                  + numpy.einsum('kl,iklj->ij', sx[:nocc,:nocc],
                                 g0[nocc:,:nocc,:nocc,:nocc])
                  + numpy.einsum('ijkk->ij', g1[nocc:,:nocc,:nocc,:nocc]) * 2
                  - numpy.einsum('ikkj->ij', g1[nocc:,:nocc,:nocc,:nocc]))

            f1 = numpy.zeros((nmo,nmo))
            f1[:nocc,:nocc] = fij
            f1[nocc:,nocc:] = fab
            f1[nocc:,:nocc] = fai
            f1[:nocc,nocc:] = fai.T
            de2[i] += numpy.einsum('ij,ij', f1, dm1)
            de2[i] += numpy.einsum('ij,ij', sx, im1)

            de2[i] += numpy.einsum('iajb,iajb', dovov, g1[:nocc,nocc:,:nocc,nocc:]) * 2
            de2[i] += numpy.einsum('acbd,acbd', dvvvv, g1[nocc:,nocc:,nocc:,nocc:]) * 2
            de2[i] += numpy.einsum('kilj,kilj', doooo, g1[:nocc,:nocc,:nocc,:nocc]) * 2
            de2[i] += numpy.einsum('jbai,jbai', dovvo, g1[:nocc,nocc:,nocc:,:nocc]) * 2
            de2[i] += numpy.einsum('jiab,jiab', doovv, g1[:nocc,:nocc,nocc:,nocc:]) * 2
            de2[i] += numpy.einsum('abic,abic', dvvov, g1[nocc:,nocc:,:nocc,nocc:]) * 2
            de2[i] += numpy.einsum('jika,jika', dooov, g1[:nocc,:nocc,:nocc,nocc:]) * 2
        de[k] = de2

    return de


class _ERIS:
    def __init__(self, cc, mo_coeff):
        nocc = numpy.count_nonzero(cc.mo_occ > 0)
        eri0 = ao2mo.full(cc._scf._eri, mo_coeff)
        eri0 = ao2mo.restore(1, eri0, mo_coeff.shape[1])
        eri0 = eri0.reshape((mo_coeff.shape[1],)*4)
        self.oooo = eri0[:nocc,:nocc,:nocc,:nocc].copy()
        self.ooov = eri0[:nocc,:nocc,:nocc,nocc:].copy()
        self.ovoo = eri0[:nocc,nocc:,:nocc,:nocc].copy()
        self.oovo = eri0[:nocc,:nocc,nocc:,:nocc].copy()
        self.oovv = eri0[:nocc,:nocc,nocc:,nocc:].copy()
        self.ovov = eri0[:nocc,nocc:,:nocc,nocc:].copy()
        self.ovvo = eri0[:nocc,nocc:,nocc:,:nocc].copy()
        self.ovvv = eri0[:nocc,nocc:,nocc:,nocc:].copy()
        self.vvvv = eri0[nocc:,nocc:,nocc:,nocc:].copy()
        self.vvvo = eri0[nocc:,nocc:,nocc:,:nocc].copy()
        self.vovv = eri0[nocc:,:nocc,nocc:,nocc:].copy()
        self.vvov = eri0[nocc:,nocc:,:nocc,nocc:].copy()
        self.vvoo = eri0[nocc:,nocc:,:nocc,:nocc].copy()
        self.voov = eri0[nocc:,:nocc,:nocc,nocc:].copy()
        self.vooo = eri0[nocc:,:nocc,:nocc,:nocc].copy()
        self.mo_coeff = mo_coeff
        self.fock = numpy.diag(cc._scf.mo_energy)

def index_frozen_active(cc):
    nocc = numpy.count_nonzero(cc.mo_occ > 0)
    moidx = ccsd.get_moidx(cc)
    OA = numpy.where( moidx[:nocc])[0] # occupied active orbitals
    OF = numpy.where(~moidx[:nocc])[0] # occupied frozen orbitals
    VA = numpy.where( moidx[nocc:])[0] # virtual active orbitals
    VF = numpy.where(~moidx[nocc:])[0] # virtual frozen orbitals
    return OA, VA, OF, VF

def gamma1_intermediates(cc, t1, t2, l1, l2):
    d1 = ccsd_rdm.gamma1_intermediates(cc, t1, t2, l1, l2)
    if cc.frozen is None or cc.frozen is 0:
        return d1
    nocc = numpy.count_nonzero(cc.mo_occ>0)
    nvir = cc.mo_occ.size - nocc
    OA, VA, OF, VF = index_frozen_active(cc)
    doo = numpy.zeros((nocc,nocc))
    dov = numpy.zeros((nocc,nvir))
    dvo = numpy.zeros((nvir,nocc))
    dvv = numpy.zeros((nvir,nvir))
    doo[OA[:,None],OA] = d1[0]
    dov[OA[:,None],VA] = d1[1]
    dvo[VA[:,None],OA] = d1[2]
    dvv[VA[:,None],VA] = d1[3]
    return doo, dov, dvo, dvv

def gamma2_intermediates(cc, t1, t2, l1, l2):
    d2 = ccsd_rdm.gamma2_intermediates(cc, t1, t2, l1, l2)
    nocc, nvir = t1.shape
    if cc.frozen is None or cc.frozen is 0:
        dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov = d2
        dvvov = dovvv.transpose(2,3,0,1)
        dvvvv = ao2mo.restore(1, d2[1], nvir)
        return dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov
    nocc0 = numpy.count_nonzero(cc.mo_occ>0)
    nvir0 = cc.mo_occ.size - nocc0
    OA, VA, OF, VF = index_frozen_active(cc)
    dovov = numpy.zeros((nocc0,nvir0,nocc0,nvir0))
    dvvvv = numpy.zeros((nvir0,nvir0,nvir0,nvir0))
    doooo = numpy.zeros((nocc0,nocc0,nocc0,nocc0))
    doovv = numpy.zeros((nocc0,nocc0,nvir0,nvir0))
    dovvo = numpy.zeros((nocc0,nvir0,nvir0,nocc0))
    dovvv = numpy.zeros((nocc0,nvir0,nvir0,nvir0))
    dooov = numpy.zeros((nocc0,nocc0,nocc0,nvir0))
    dovov[OA[:,None,None,None],VA[:,None,None],OA[:,None],VA] = d2[0]
    dvvvv[VA[:,None,None,None],VA[:,None,None],VA[:,None],VA] = ao2mo.restore(1, d2[1], nvir)
    doooo[OA[:,None,None,None],OA[:,None,None],OA[:,None],OA] = d2[2]
    doovv[OA[:,None,None,None],OA[:,None,None],VA[:,None],VA] = d2[3]
    dovvo[OA[:,None,None,None],VA[:,None,None],VA[:,None],OA] = d2[4]
    dovvv[OA[:,None,None,None],VA[:,None,None],VA[:,None],VA] = d2[6]
    dooov[OA[:,None,None,None],OA[:,None,None],OA[:,None],VA] = d2[7]
    dvvov = dovvv.transpose(2,3,0,1)
    return dovov, dvvvv, doooo, doovv, dovvo, dvvov, dovvv, dooov


if __name__ == '__main__':
    from pyscf import gto
    from pyscf import scf
    from pyscf.cc import ccsd
    from pyscf import grad

    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol).run()
    mycc = ccsd.CCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2)
    ghf = grad.RHF(mf).grad()
    print('gcc')
    print(ghf+g1)
    print(lib.finger(g1) - -0.042511000925747583)
#[[ 0   0                1.00950969e-02]
# [ 0   2.28063353e-02  -5.04754844e-03]
# [ 0  -2.28063353e-02  -5.04754844e-03]]

    print('-----------------------------------')
    mol = gto.M(
        verbose = 0,
        atom = [
            ["O" , (0. , 0.     , 0.)],
            [1   , (0. ,-0.757  , 0.587)],
            [1   , (0. , 0.757  , 0.587)]],
        basis = '631g'
    )
    mf = scf.RHF(mol).run()
    mycc = ccsd.CCSD(mf)
    mycc.frozen = [0,1,10,11,12]
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2)
    ghf = grad.RHF(mf).grad()
    print('gcc')
    print(ghf+g1)
    print(lib.finger(g1) - 0.10048468674687236)
#[[ -7.81105940e-17   3.81840540e-15   1.20415540e-02]
# [  1.73095055e-16  -7.94568837e-02  -6.02077699e-03]
# [ -9.49844615e-17   7.94568837e-02  -6.02077699e-03]]

    r = 1.76
    mol = gto.M(
        verbose = 0,
        atom = '''H 0 0 0; H 0 0 %f''' % r,
        basis = '631g',
        unit = 'bohr')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf0 = mf.scf()
    ghf = grad.RHF(mf).grad()
    mycc = ccsd.CCSD(mf)
    ecc, t1, t2 = mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    g1 = kernel(mycc, t1, t2, l1, l2)
    ghf = grad.RHF(mf).grad()
    print('ghf')
    print(ghf)
    print('gcc')
    print(g1) # 0.015643667024
    print('tot')
    print(ghf+g1) # -0.0708003526454

    mol = gto.M(
        verbose = 0,
        atom = '''H 0 0 0; H 0 0 %f''' % (r-.001),
        basis = '631g',
        unit = 'bohr')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf0 = mf.scf()
    mycc = ccsd.CCSD(mf)
    ecc0 = mycc.kernel()[0]

    mol = gto.M(
        verbose = 0,
        atom = '''H 0 0 0; H 0 0 %f''' % (r+.001),
        basis = '631g',
        unit = 'bohr')
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-14
    ehf1 = mf.scf()
    mycc = ccsd.CCSD(mf)
    ecc1 = mycc.kernel()[0]
    print((ehf1-ehf0)*500 - ghf[1,2])
    print('decc', (ecc1-ecc0)*500 - g1[1,2])
    print('decc', (ehf1+ecc1-ehf0-ecc0)*500 - (ghf[1,2]+g1[1,2]))
