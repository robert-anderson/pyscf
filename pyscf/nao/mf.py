from __future__ import print_function, division
import numpy as np
from numpy import require, zeros
from pyscf.nao import nao, prod_basis
from pyscf.nao import conv_yzx2xyz_c
    
#
#
#
class mf(nao):

  def __init__(self, **kw):
    """ Constructor a mean field class (store result of a mean-field calc, deliver density matrix etc) """
    #print(__name__, 'before construct')
    nao.__init__(self, **kw)
    self.dtype = kw['dtype'] if 'dtype' in kw else np.float64
    self.pseudo = hasattr(self, 'sp2ion') 
    self.gen_pb = kw['gen_pb'] if 'gen_pb' in kw else True
    
    if 'mf' in kw:
      self.init_mo_from_pyscf(**kw)      
    elif 'label' in kw: # init KS orbitals with SIESTA
      self.init_mo_coeff_label(**kw)
      self.k2xyzw = self.xml_dict["k2xyzw"]
      self.xc_code = 'LDA,PZ' # just a guess...
    elif 'fireball' in kw: # init KS orbitals with Fireball
      self.init_mo_coeff_fireball(**kw)
      self.xc_code = 'GGA,PBE' # just a guess...
    elif 'gpaw' in kw:
      self.init_mo_coeff_label(**kw)
      self.k2xyzw = np.array([[0.0,0.0,0.0,1.0]])
      self.xc_code = 'LDA,PZ' # just a guess, but in case of GPAW there is a field...
    elif 'openmx' in kw:
      self.xc_code = 'GGA,PBE' # just a guess...
      pass
    else:
      raise RuntimeError('unknown constructor')
    if self.verbosity>0: print(__name__, ' pseudo ', self.pseudo)
    self.init_libnao()
    if self.gen_pb:
      self.pb = prod_basis(nao=self, **kw)
      if self.verbosity>0: print(__name__, ' dtype ', self.dtype, ' norbs ', self.norbs)
      #self.pb.init_prod_basis_pp_batch(nao=self, **kw)

  def make_rdm1(self, mo_coeff=None, mo_occ=None, **kw):
    # from pyscf.scf.hf import make_rdm1 -- different index order here
    if mo_occ is None: mo_occ = self.mo_occ[0,:,:]
    if mo_coeff is None: mo_coeff = self.mo_coeff[0,:,:,:,0]
    dm = np.zeros((1,self.nspin,self.norbs,self.norbs,1))
    for s in range(self.nspin):
      xocc = mo_coeff[s,mo_occ[s]>0,:]
      focc = mo_occ[s,mo_occ[s]>0]
      dm[0,s,:,:,0] = np.dot(xocc.T.conj() * focc, xocc)
    return dm

  def init_mo_from_pyscf(self, **kw):
    """ Initializing from a previous pySCF mean-field calc. """
    from pyscf.nao.m_fermi_energy import fermi_energy as comput_fermi_energy
    from pyscf.nao.m_color import color as tc
    self.telec = kw['telec'] if 'telec' in kw else 0.0000317 # 10K
    self.mf = mf = kw['mf']
    if type(mf.mo_energy) == tuple: 
      self.nspin = len(mf.mo_energy)
    elif type(mf.mo_energy) == np.ndarray:
      self.nspin = 1
      assert mf.mo_coeff.shape[0]==mf.mo_coeff.shape[1]
      assert len(mf.mo_coeff.shape)==2
    else:
      raise RuntimeError(tc.RED+'unknown type of mo_energy %s %s' % (type(mf.mo_energy), tc.ENDC))
    
    self.xc_code = mf.xc if hasattr(mf, 'xc') else 'HF'
    self.k2xyzw = np.array([[0.0,0.0,0.0,1.0]])
    nspin,n=self.nspin,self.norbs
    self.mo_coeff =  require(zeros((1,nspin,n,n,1), dtype=self.dtype), requirements='CW')
    self.mo_energy = require(zeros((1,nspin,n), dtype=self.dtype), requirements='CW')
    self.mo_occ =    require(zeros((1,nspin,n),dtype=self.dtype), requirements='CW')

    conv = conv_yzx2xyz_c(kw['gto'])
    if nspin==1 :
      self.mo_coeff[0,0,:,:,0] = conv.conv_yzx2xyz_1d(self.mf.mo_coeff, conv.m_xyz2m_yzx).T
      self.mo_energy[0,0,:] = self.mf.mo_energy
      self.mo_occ[0,0,:] = self.mf.mo_occ
    else:
      for s in range(self.nspin):
        self.mo_coeff[0,s,:,:,0] = conv.conv_yzx2xyz_1d(self.mf.mo_coeff[s], conv.m_xyz2m_yzx).T
        self.mo_energy[0,s,:] = self.mf.mo_energy[s]
      self.mo_occ[0,0:self.nspin,:] = self.mf.mo_occ
      
    self.nelec = kw['nelec'] if 'nelec' in kw else np.array([int(s2o.sum()) for s2o in self.mo_occ[0]])
    fermi = comput_fermi_energy(self.mo_energy, sum(self.nelec), self.telec)
    self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else fermi

  def init_mo_coeff_label(self, **kw):
    """ Constructor a mean-field class from the preceeding SIESTA calculation """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    self.mo_coeff = np.require(self.wfsx.x, dtype=self.dtype, requirements='CW')
    self.mo_energy = np.require(self.wfsx.ksn2e, dtype=self.dtype, requirements='CW')
    self.telec = kw['telec'] if 'telec' in kw else self.hsx.telec
    if self.nspin==1:
      self.nelec = kw['nelec'] if 'nelec' in kw else np.array([self.hsx.nelec])
    elif self.nspin==2:
      self.nelec = kw['nelec'] if 'nelec' in kw else np.array([int(self.hsx.nelec/2), int(self.hsx.nelec/2)])      
      print(__name__, 'not sure here: self.nelec', self.nelec)
    else:
      raise RuntimeError('0>nspin>2?')
      
    self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else self.fermi_energy
    ksn2fd = fermi_dirac_occupations(self.telec, self.mo_energy, self.fermi_energy)
    self.mo_occ = (3-self.nspin)*ksn2fd

  def init_mo_coeff_fireball(self, **kw):
    """ Constructor a mean-field class from the preceeding FIREBALL calculation """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    from pyscf.nao.m_fireball_get_eigen_dat import fireball_get_eigen_dat
    from pyscf.nao.m_fireball_hsx import fireball_hsx
    self.telec = kw['telec'] if 'telec' in kw else self.telec
    self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else self.fermi_energy
    self.mo_energy = np.require(fireball_get_eigen_dat(self.cd), dtype=self.dtype, requirements='CW')
    ksn2fd = fermi_dirac_occupations(self.telec, self.mo_energy, self.fermi_energy)
    self.mo_occ = (3-self.nspin)*ksn2fd
    if abs(self.nelectron-self.mo_occ.sum())>1e-6: raise RuntimeError("mo_occ wrong?" )
    print(__name__, ' self.nspecies ', self.nspecies)
    print(self.sp_mu2j)
    
    self.hsx = fireball_hsx(self, **kw)
    #print(self.telec)
    #print(self.mo_energy)
    #print(self.fermi_energy)
    #print(__name__, ' sum(self.mo_occ)', sum(self.mo_occ))
    #print(self.mo_occ)
    
        

  def diag_check(self, atol=1e-5, rtol=1e-4):
    from pyscf.nao.m_sv_diag import sv_diag 
    ksn2e = self.mo_energy
    ac = True
    for k,kvec in enumerate(self.k2xyzw):
      for spin in range(self.nspin):
        e,x = sv_diag(self, kvec=kvec[0:3], spin=spin)
        eref = ksn2e[k,spin,:]
        acks = np.allclose(eref,e,atol=atol,rtol=rtol)
        ac = ac and acks
        if(not acks):
          aerr = sum(abs(eref-e))/len(e)
          print("diag_check: "+bc.RED+str(k)+' '+str(spin)+' '+str(aerr)+bc.ENDC)
    return ac

  def get_occupations(self, telec=None, ksn2e=None, fermi_energy=None):
    """ Compute occupations of electron levels according to Fermi-Dirac distribution """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations
    Telec = self.hsx.telec if telec is None else telec
    ksn2E = self.wfsx.ksn2e if ksn2e is None else ksn2e
    Fermi = self.fermi_energy if fermi_energy is None else fermi_energy
    ksn2fd = fermi_dirac_occupations(Telec, ksn2E, Fermi)
    ksn2fd = (3.0-self.nspin)*ksn2fd
    return ksn2fd

  def init_libnao(self, wfsx=None):
    """ Initialization of data on libnao site """
    from pyscf.nao.m_libnao import libnao
    from pyscf.nao.m_sv_chain_data import sv_chain_data
    from ctypes import POINTER, c_double, c_int64, c_int32, byref

    if wfsx is None:
        data = sv_chain_data(self)
        # (nkpoints, nspin, norbs, norbs, nreim)
        #print(' data ', sum(data))
        size_x = np.array([1, self.nspin, self.norbs, self.norbs, 1], dtype=np.int32)
        libnao.init_sv_libnao_orbs.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int32))
        libnao.init_sv_libnao_orbs(data.ctypes.data_as(POINTER(c_double)), c_int64(len(data)), size_x.ctypes.data_as(POINTER(c_int32)))
        self.init_sv_libnao = True
    else:
        size_x = np.zeros(len(self.wfsx.x.shape), dtype=np.int32)
        for i, sh in enumerate(self.wfsx.x.shape): size_x[i] = sh

        data = sv_chain_data(self)
        libnao.init_sv_libnao_orbs.argtypes = (POINTER(c_double), POINTER(c_int64), POINTER(c_int32))
        libnao.init_sv_libnao_orbs(data.ctypes.data_as(POINTER(c_double)), c_int64(len(data)), size_x.ctypes.data_as(POINTER(c_int32)))
        self.init_sv_libnao = True

    libnao.init_aos_libnao.argtypes = (POINTER(c_int64), POINTER(c_int64))
    info = c_int64(-999)
    libnao.init_aos_libnao(c_int64(self.norbs), byref(info))
    if info.value!=0: raise RuntimeError("info!=0")
    return self

  def vxc_lil(self, **kw):   # Compute exchange-correlation potentials
    from pyscf.nao.m_vxc_lil import vxc_lil
    return vxc_lil(self, deriv=1, **kw)

  def dens_elec(self, coords, dm): # Compute electronic density for a given density matrix and on a given set of coordinates
    from pyscf.nao.m_dens_libnao import dens_libnao
    from pyscf.nao.m_init_dm_libnao import init_dm_libnao
    from pyscf.nao.m_init_dens_libnao import init_dens_libnao
    if not self.init_sv_libnao : raise RuntimeError('not self.init_sv_libnao')
    if init_dm_libnao(dm) is None : raise RuntimeError('init_dm_libnao(dm) is None')
    if init_dens_libnao()!=0 : raise RuntimeError('init_dens_libnao()!=0')
    return dens_libnao(coords, self.nspin)

  def exc(self, dm, xc_code, **kw):   # Compute exchange-correlation energies
    from pyscf.nao.m_exc import exc
    return exc(self, dm, xc_code, **kw)

  def get_init_guess(self, mol=None, key=None):
    """ Compute an initial guess for the density matrix. """
    from pyscf.scf.hf import init_guess_by_minao
    if hasattr(self, 'mol'):
      dm = init_guess_by_minao(self.mol)
    else:
      dm = self.make_rdm1()  # the loaded ks orbitals will be used
      if dm.shape[0:2]==(1,1) and dm.shape[4]==1 : dm = dm.reshape((self.norbs,self.norbs))
    return dm

  def get_hamiltonian(self): # Returns the stored matrix elements of current hamiltonian 
    return self.hsx.spin2h4_csr
  
  def dos(self, comegas, **kw):
    """ Ordinary Density of States (from the current mean-field eigenvalues) """
    from pyscf.nao.scf_dos import scf_dos
    return scf_dos(self, comegas, **kw)

  def pdos(self, comegas, **kw):
    """ Partial Density of States (resolved in angular momentum of atomic orbitals) """
    from pyscf.nao.pdos import pdos
    return pdos(self, comegas, **kw)

  def lsoa_dos(self, comegas, **kw):
    """ Partial Density of States (contributions from a given list of atoms) """
    from pyscf.nao.pdos import lsoa_dos
    return lsoa_dos(self, comegas, **kw)

  def gdos(self, comegas, **kw):
    """ Some molecular orbital population analysis """
    from pyscf.nao.pdos import gdos
    return gdos(self, comegas, **kw)

  def read_wfsx(self, fname, **kw):
    """ An occasional reading of the SIESTA's .WFSX file """
    from pyscf.nao.m_siesta_wfsx import siesta_wfsx_c
    from pyscf.nao.m_siesta2blanko_denvec import _siesta2blanko_denvec
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations

    self.wfsx = siesta_wfsx_c(fname=fname, **kw)
    
    assert self.nkpoints == self.wfsx.nkpoints
    assert self.norbs == self.wfsx.norbs 
    assert self.nspin == self.wfsx.nspin
    orb2m = self.get_orb2m()
    for k in range(self.nkpoints):
      for s in range(self.nspin):
        for n in range(self.norbs):
          _siesta2blanko_denvec(orb2m, self.wfsx.x[k,s,n,:,:])

    self.mo_coeff = np.require(self.wfsx.x, dtype=self.dtype, requirements='CW')
    self.mo_energy = np.require(self.wfsx.ksn2e, dtype=self.dtype, requirements='CW')
    self.telec = kw['telec'] if 'telec' in kw else self.hsx.telec
    self.nelec = kw['nelec'] if 'nelec' in kw else self.hsx.nelec
    self.fermi_energy = kw['fermi_energy'] if 'fermi_energy' in kw else self.fermi_energy
    ksn2fd = fermi_dirac_occupations(self.telec, self.mo_energy, self.fermi_energy)
    self.mo_occ = (3-self.nspin)*ksn2fd
    return self


  def plot_contour(self, w=0.0):
    """
      Plot contour with poles of Green's function in the self-energy 
      SelfEnergy(w) = G(w+w')W(w')
      with respect to w' = Re(w')+Im(w')
      Poles of G(w+w') are located: w+w'-(E_n-Fermi)+i*eps sign(E_n-Fermi)==0 ==> 
      w'= (E_n-Fermi) - w -i eps sign(E_n-Fermi)
    """
    try :
      import matplotlib.pyplot as plt
      from matplotlib.patches import Arc, Arrow 
    except:
      print('no matplotlib?')
      return

    fig,ax = plt.subplots()
    fe = self.fermi_energy
    ee = self.mo_energy
    iee = 0.5-np.array(ee>fe)
    eew = ee-fe-w
    ax.plot(eew, iee, 'r.', ms=10.0)
    pp = list()
    pp.append(Arc((0,0),4,4,angle=0, linewidth=2, theta1=0, theta2=90, zorder=2, color='b'))
    pp.append(Arc((0,0),4,4,angle=0, linewidth=2, theta1=180, theta2=270, zorder=2, color='b'))
    pp.append(Arrow(0,2,0,-4,width=0.2, color='b', hatch='o'))
    pp.append(Arrow(-2,0,4,0,width=0.2, color='b', hatch='o'))
    for p in pp: ax.add_patch(p)
    ax.set_aspect('equal')
    ax.grid(True, which='both')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    plt.ylim(-3.0,3.0)
    plt.show()

#
# Example of reading pySCF mean-field calculation.
#
if __name__=="__main__":
  from pyscf import gto, scf as scf_gto
  from pyscf.nao import nao, mf
  """ Interpreting small Gaussian calculation """
  mol = gto.M(atom='O 0 0 0; H 0 0 1; H 0 1 0; Be 1 0 0', basis='ccpvdz') # coordinates in Angstrom!
  dft = scf_gto.RKS(mol)
  dft.kernel()
  
  sv = mf(mf=dft, gto=dft.mol, rcut_tol=1e-9, nr=512, rmin=1e-6)
  
  print(sv.ao_log.sp2norbs)
  print(sv.ao_log.sp2nmult)
  print(sv.ao_log.sp2rcut)
  print(sv.ao_log.sp_mu2rcut)
  print(sv.ao_log.nr)
  print(sv.ao_log.rr[0:4], sv.ao_log.rr[-1:-5:-1])
  print(sv.ao_log.psi_log[0].shape, sv.ao_log.psi_log_rl[0].shape)
  print(dir(sv.pb))
  print(sv.pb.norbs)
  print(sv.pb.npdp)
  print(sv.pb.c2s[-1])
