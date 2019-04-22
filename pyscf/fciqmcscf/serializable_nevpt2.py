import pyscf_interface
import pyscf
import pickle, os
import numpy as np
import shutil
from pyscf import mcscf, gto, scf, mrpt, ao2mo, fciqmcscf
einsum = mrpt.nevpt2.einsum

def output_fcidump(casci, mol, dirname=None):
    tmp = casci.fcisolver
    casci.fcisolver = fciqmcscf.FCIQMCCI(mol)
    casci.fcisolver.only_ints = 1
    try: casci.kernel()
    except TypeError: pass
    if dirname is not None: shutil.move('FCIDUMP', '{}/FCIDUMP'.format(dirname))
    casci.fcisolver = tmp

class SerializableNevpt2:
    mol_kwargs = None
    hf_canon_mo = None
    hf_mo_energy = None
    casci_canon_mo = None
    casci_mo_energy = None
    def __init__(self, fciqmc_dir=None, mol_kwargs=None, fname=None, norb=None, nelecas=None):
        if isinstance(mol_kwargs, dict):
            self.mol_kwargs = mol_kwargs
            mol = self.mol()
            hf = scf.RHF(mol)
            hf.conv_tol = 1e-10
            hf.kernel()
            self.hf_canon_mo = hf.mo_coeff
            self.hf_mo_energy = hf.mo_energy
            casci = mcscf.CASCI(hf, norb, nelecas)
            self.norb, self.nelecas = norb, nelecas
            if fciqmc_dir is not None:
                print '### Initial invokation for subsequent FCIQMC RDM sampling'
                output_fcidump(casci, mol, fciqmc_dir)
            else:
                print '### Exact CASCI+NEVPT2 invokation'
                output_fcidump(casci, mol)
                casci.kernel()
                self.casci_canon_mo = casci.mo_coeff
                self.casci_mo_energy = casci.mo_energy
                nevpt2 = mrpt.NEVPT(casci)
                nevpt2.kernel()
        elif isinstance(fname, str) and os.path.exists(fname):
            self.load(fname)
            mol = self.mol()
            hf = scf.RHF(mol)
            hf.kernel()
            #casci = mcscf.CASCI(mol, self.norb, self.nelecas)
            casci = mcscf.CASCI(hf, self.norb, self.nelecas)
            casci.mo_coeff = self.casci_canon_mo if self.casci_canon_mo is not None else self.hf_canon_mo
            casci.mo_energy = self.casci_mo_energy if self.casci_canon_mo is not None else self.hf_mo_energy

            with open('/scratch/scratch/mmm0043/work/nevpt2/production/N2/1.0977A/6o6e/casci.pkl', 'rb') as f:
                tmp = pickle.load(f)

            if fciqmc_dir is not None:
                print '### Stochastic NEVPT2 invokation with FCIQMC RDMs'
                casci.fcisolver = fciqmcscf.FCIQMCCI(mol)
                casci.fcisolver.dirname = fciqmc_dir
            else:
                print '### Exact NEVPT2 invokation from previous CASCI'
            nevpt2 = mrpt.NEVPT(casci)
            nevpt2.canonicalized = self.casci_canon_mo is not None
            nevpt2.kernel()
        else:
            raise Exception('Neither a mol dict nor a valid path was provided')
        self.save(fname)

    def mol(self):
        return gto.M(verbose=4, **self.mol_kwargs)

    def save(self, fname):
        if fname is None: return
        with open('tmp.pkl', 'wb') as f: pickle.dump(self.__dict__, f)
        shutil.move('tmp.pkl', fname)

    def load(self, fname):
        if fname is None: return
        with open(fname, 'rb') as f: self.__dict__.update(pickle.load(f))
        if self.casci_canon_mo is not None:
            print '### Loading complete, working with CASCI canonical orbitals'
        else:
            print '### Loading complete, working with RHF canonical orbitals'


class NEVPT:
    ls = [ 'r',    'i',   'ijrs',  'ijr',  'rsi',  'rs',   'ij',  'ir']
    ks = ['(-1)', '(+1)', '(0)',  '(+1)', '(-1)', '(-2)', '(+2)', '(0)']
    def __init__(self, mol_kwargs):
        mol = gto.M(mol_kwargs)
        self.ci = None
        self.dms = None
        self.saved_ls = []
        mol = gto

    def save(self, fname):
        with open('tmp.pkl', 'wb') as f: pickle.dump(self, f)
        shutil.move('tmp.pkl', fname)

    def load(self, fname):
        with open(fname, 'rb') as f: self.__dict__.update(pickle.load(f))

    def set_dms(self, dms):
        self.dms = dms
        self.dm1, self.dm2, self.dm3 = tuple(dms[i] for i in '123')

    def set_mos(self, mo_coeff, mo_energy, ncore, ncas, nelecas, eris):
        self.ncore, self.ncas, self.nelecas = ncore, ncas, nelecas
        self.nocc = ncore+ncas
        self.mo_coeff = mo_coeff
        self.mo_core = mo_coeff[:,:self.ncore]
        self.mo_cas = mo_coeff[:,self.ncore:self.nocc]
        self.mo_virt = mo_coeff[:,self.nocc:]
        self.mo_energy = mo_energy
        self.nvirt = self.mo_virt.shape[1]
        self.eris = eris
        self.h1e = self.eris['h1eff'][self.ncore:self.nocc, self.ncore:self.nocc]
        self.h2e = self.eris['ppaa'][self.ncore:self.nocc, self.ncore:self.nocc].transpose(0,2,1,3)

    def contract(self):
        tot_ener = 0.0
        for isubspace in range(len(self.ls)):
            l = self.ls[isubspace]
            k = self.ks[isubspace]
            print '\t{}/{}'.format(isubspace+1, len(self.ls))
            print '\tS{} subspace...'.format(l)
            try: norm, ener, diff = getattr(self, 'contract_S{}'.format(l))()
            except AttributeError: continue

            if l in self.saved_ls:
                mrpt.nevpt2.save_contractions(name, norm, ener, diff)
            if diff is not None:
                norm, ener = mrpt.nevpt2._norm_to_energy(norm, ener, diff)
            tot_ener+=ener
            print "S{l:4s}  {k:4s}, E = {r:.14f}".format(l=l, k=k, r=ener)
            print "S{l:4s}  {k:4s}, N = {r:.14f}".format(l=l, k=k, r=norm)
        print "Nevpt2 Energy = {r:.14f}".format(r=tot_ener)
        return tot_ener

    def scf(self, mol):
        mf = scf.RHF(mol)
        mf.kernel()
        return mf

    def casci(self, scf_obj):
        assert scf_obj.mo_coeff is not None, 'SCF must be run first'
        

    def make_eris(self, scf_obj):
        self.eris = pyscf.ao2mo.incore.general(scf_obj._eri, (orbs,)*4, compact=False)

        h_core = scf_obj.get_hcore(fciqmcci.mol)
        h = reduce(numpy.dot, (orbs.T, h_core, orbs))

        pyscf.tools.fcidump.from_integrals(fciqmcci.integralFile, h, 
                pyscf.ao2mo.restore(8,eri,nmo), nmo, nelec, fciqmcci.mol.energy_nuc(),
                fciqmcci.mol.spin, orbsym, tol=tol)

    def contract_Sr(self):
        h2e_v = self.eris['ppaa'][self.nocc:,self.ncore:self.nocc].transpose(0,2,1,3)
        h1e_v = self.eris['h1eff'][self.nocc:,self.ncore:self.nocc] - einsum('mbbn->mn',h2e_v)

        a16 = mrpt.nevpt2.make_a16(self.h1e, self.h2e, self.dms, self.ci, self.ncas, self.nelecas)
        a17 = mrpt.nevpt2.make_a17(self.h1e, self.h2e, self.dm2, self.dm3)
        a19 = mrpt.nevpt2.make_a19(self.h1e, self.h2e, self.dm1, self.dm2)

        ener = einsum('ipqr,pqrabc,iabc->i', h2e_v, a16, h2e_v)\
            +  einsum('ipqr,pqra,ia->i', h2e_v, a17, h1e_v)*2.0\
            +  einsum('ip,pa,ia->i', h1e_v, a19, h1e_v)

        norm = einsum('ipqr,rpqbac,iabc->i', h2e_v, self.dm3, h2e_v)\
            +  einsum('ipqr,rpqa,ia->i', h2e_v, self.dm2, h1e_v)*2.0\
            +  einsum('ip,pa,ia->i', h1e_v, self.dm1, h1e_v)

        diff = self.mo_energy[self.ncore+self.ncas:]
        return norm, ener, diff

    def contract_Si(self):
        h2e_v = self.eris['ppaa'][self.ncore:self.nocc,:self.ncore].transpose(0,2,1,3)
        h1e_v = self.eris['h1eff'][self.ncore:self.nocc,:self.ncore]

        a22 = mrpt.nevpt2.make_a22(self.h1e, self.h2e, self.dms, self.ci, self.ncas, self.nelecas)
        a23 = mrpt.nevpt2.make_a23(self.h1e, self.h2e, self.dm1, self.dm2, self.dm3)
        a25 = mrpt.nevpt2.make_a25(self.h1e, self.h2e, self.dm1, self.dm2)
        delta = np.eye(self.ncas)
        dm3_h = einsum('abef,cd->abcdef', self.dm2, delta)*2\
                - self.dm3.transpose(0,1,3,2,4,5)
        dm2_h = einsum('ab,cd->abcd', self.dm1, delta)*2\
                - self.dm2.transpose(0,1,3,2)
        dm1_h = 2*delta - self.dm1.transpose(1,0)

        ener = einsum('qpir,pqrabc,baic->i', h2e_v, a22, h2e_v)\
            +  einsum('qpir,pqra,ai->i', h2e_v, a23, h1e_v)*2.0\
            +  einsum('pi,pa,ai->i', h1e_v, a25, h1e_v)

        norm = einsum('qpir,rpqbac,baic->i', h2e_v, dm3_h, h2e_v)\
            +  einsum('qpir,rpqa,ai->i',h2e_v, dm2_h, h1e_v)*2.0\
            +  einsum('pi,pa,ai->i',h1e_v, dm1_h, h1e_v)

        diff = -mc.mo_energy[:mc.ncore]
        return norm, ener, diff

    def contract_Sijrs(self):
        feri = self.eris['cvcv']
        eia = self.mo_energy[:self.ncore, None] -self.mo_energy[None, self.nocc:]
        norm = 0
        ener = 0
        with ao2mo.load(feri) as cvcv:
            for i in range(self.ncore):
                djba = (eia.reshape(-1,1) + eia[i].reshape(1,-1)).ravel()
                gi = np.asarray(cvcv[i*self.nvirt:(i+1)*self.nvirt])
                gi = gi.reshape(self.nvirt, self.ncore, self.nvirt).transpose(1,2,0)
                t2i = (gi.ravel()/djba).reshape(self.ncore, self.nvirt, self.nvirt)
                # 2*ijab-ijba
                theta = gi*2 - gi.transpose(0,2,1)
                norm += einsum('jab,jab', gi, theta)
                ener += einsum('jab,jab', t2i, theta)
        return norm, ener, None

    def contract_Sijr(self):
        h2e_v = self.eris['pacv'][:self.ncore].transpose(3,1,2,0)
        hdm1 = mrpt.nevpt2.make_hdm1(self.dm1)

        a3 = mrpt.nevpt2.make_a3(self.h1e, self.h2e, self.dm1, self.dm2, hdm1)
        norm = 2.0*einsum('rpji,raji,pa->rji', h2e_v, h2e_v, hdm1)\
             - 1.0*einsum('rpji,raij,pa->rji', h2e_v, h2e_v, hdm1)
        ener = 2.0*einsum('rpji,raji,pa->rji', h2e_v, h2e_v, a3)\
             - 1.0*einsum('rpji,raij,pa->rji', h2e_v, h2e_v, a3)

        diff = self.mo_energy[self.ncore+self.ncas:,None,None] -\
               self.mo_energy[None,:self.ncore,None] -\
               self.mo_energy[None,None,:self.ncore]
        return norm, ener, diff

    def contract_Srsi(self):
        h2e_v = self.eris['pacv'][self.nocc:].transpose(3,0,2,1)

        k27 = mrpt.nevpt2.make_k27(self.h1e, self.h2e, self.dm1, self.dm2)
        norm = 2.0*einsum('rsip,rsia,pa->rsi', h2e_v, h2e_v, self.dm1)\
             - 1.0*einsum('rsip,sria,pa->rsi', h2e_v, h2e_v, self.dm1)
        ener = 2.0*einsum('rsip,rsia,pa->rsi', h2e_v, h2e_v, k27)\
             - 1.0*einsum('rsip,sria,pa->rsi', h2e_v, h2e_v, k27)
        diff = self.mo_energy[self.ncore+self.ncas:, None, None] +\
               self.mo_energy[None, self.ncore+self.ncas:, None] -\
               self.mo_energy[None, None, :self.ncore]
        return norm, ener, diff

    def contract_Srs(self):
        if not self.nvirt: return 0, 0, None
        h2e_v = self.eris['papa'][self.nocc:,:,self.nocc:].transpose(0,2,1,3)

        # a7 is very sensitive to the accuracy of HF orbital and CI wfn
        rm2, a7 = mrpt.nevpt2.make_a7(self.h1e, self.h2e, self.dm1, self.dm2, self.dm3)
        norm = 0.5*einsum('rsqp,rsba,pqba->rs', h2e_v, h2e_v, rm2)
        ener = 0.5*einsum('rsqp,rsba,pqab->rs', h2e_v, h2e_v, a7)
        diff = self.mo_energy[self.ncore+self.ncas:,None] +\
               self.mo_energy[None,self.ncore+self.ncas:]
        return norm, ener, diff

    def contract_Sij(self):
        if not self.nvirt: return 0, 0, None
        h2e_v = self.eris['papa'][:self.ncore,:,:self.ncore].transpose(1,3,0,2)

        hdm1 = mrpt.nevpt2.make_hdm1(self.dm1)
        hdm2 = mrpt.nevpt2.make_hdm2(self.dm1, self.dm2)
        hdm3 = mrpt.nevpt2.make_hdm3(self.dm1, self.dm2, self.dm3, hdm1, hdm2)

        # a9 is very sensitive to the accuracy of HF orbital and CI wfn
        a9 = mrpt.nevpt2.make_a9(self.h1e, self.h2e, hdm1, hdm2, hdm3)
        norm = 0.5*einsum('qpij,baij,pqab->ij', h2e_v, h2e_v, hdm2)
        ener = 0.5*einsum('qpij,baij,pqab->ij', h2e_v, h2e_v, a9)
        diff = -(self.mo_energy[:self.ncore, None] + self.mo_energy[None, :self.ncore])
        return norm, ener, diff

    def contract_Sir(self):
        h2e_v1 = self.eris['ppaa'][self.nocc:,:self.ncore].transpose(0,2,1,3)
        h2e_v2 = self.eris['papa'][self.nocc:,:,:self.ncore].transpose(0,3,1,2)
        h1e_v = self.eris['h1eff'][self.nocc:,:self.ncore]

        norm = einsum('rpiq,raib,qpab->ir', h2e_v1, h2e_v1, self.dm2)*2.0\
             - einsum('rpiq,rabi,qpab->ir', h2e_v1, h2e_v2, self.dm2)\
             - einsum('rpqi,raib,qpab->ir', h2e_v2, h2e_v1, self.dm2)\
             + einsum('raqi,rabi,qb->ir', h2e_v2, h2e_v2, self.dm1)*2.0\
             - einsum('rpqi,rabi,qbap->ir', h2e_v2, h2e_v2, self.dm2)\
             + einsum('rpqi,raai,qp->ir', h2e_v2, h2e_v2, self.dm1)\
             + einsum('rpiq,ri,qp->ir', h2e_v1, h1e_v, self.dm1)*4.0\
             - einsum('rpqi,ri,qp->ir', h2e_v2, h1e_v, self.dm1)*2.0\
             + einsum('ri,ri->ir', h1e_v, h1e_v)*2.0

        a12 = mrpt.nevpt2.make_a12(self.h1e, self.h2e, self.dm1, self.dm2, self.dm3)
        a13 = mrpt.nevpt2.make_a13(self.h1e, self.h2e, self.dm1, self.dm2, self.dm3)

        ener = einsum('rpiq,raib,pqab->ir', h2e_v1, h2e_v1, a12)*2.0\
             - einsum('rpiq,rabi,pqab->ir', h2e_v1, h2e_v2, a12)\
             - einsum('rpqi,raib,pqab->ir', h2e_v2, h2e_v1, a12)\
             + einsum('rpqi,rabi,pqab->ir', h2e_v2, h2e_v2, a13)
        diff = -mc.mo_energy[:mc.ncore,None] + mc.mo_energy[None,mc.ncore+mc.ncas:]
        return norm, ener, diff


'''
myhf = scf.RHF(mol)
myhf.kernel()
s = myhf.get_ovlp()

def unitarily_related(a, b):
    return np.allclose(np.dot(a,a.T), np.dot(b,b.T))

mc = mcscf.CASCI(myhf, 6, 6)
a = mc.mo_coeff.copy()
#mc.canonicalization = 0
#mc.natorb = 1
mc.kernel()
b = mc.mo_coeff

print np.allclose(a[:mc.ncore], b[:mc.ncore])

print unitarily_related(a[:mc.ncore], b[:mc.ncore])
1/0

obj = mrpt.NEVPT(mc)
obj.kernel()

serial_nevpt2 = NEVPT(None)
serial_nevpt2.set_dms(obj.dms)
serial_nevpt2.set_mos(obj.mo_coeff, obj.mo_energy, obj.ncore, obj.ncas, obj.nelecas, obj.eris)

print
print
print
print serial_nevpt2.contract()

if __name__=='__main__':
    pass

'''

