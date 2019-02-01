from __future__ import print_function, division
import numpy as np
from numpy import array, argmax, einsum, require, zeros, dot
from timeit import default_timer as timer
from pyscf.nao import tddft_iter
from pyscf.nao import gw
from scipy.linalg import blas
from pyscf.nao.m_pack2den import pack2den_u, pack2den_l

class bse_iter(gw):

  def __init__(self, **kw):
    """ 
      Iterative BSE a la PK, DF, OC JCTC 
      additionally to the fields from tddft_iter_c, we add the dipole matrix elements dab[ixyz][a,b]
      which is constructed as list of numpy arrays 
       $ d_i = \int f^a(r) r_i f^b(r) dr $
    """
    from pyscf.nao.m_fermi_dirac import fermi_dirac_occupations

    xc_code_kw = kw['xc_code'] if 'xc_code' in kw else None
    gw.__init__(self, **kw)
    #print(__name__, ' dtype ', self.dtype)

    self.l0_ncalls = 0
    self.dip_ab = [d.toarray() for d in self.dipole_coo()]
    self.norbs2 = self.norbs**2
    kernel_den = pack2den_l(self.kernel)
    n = self.norbs
    v_dab = self.v_dab
    cc_da = self.cc_da
    self.x = self.mo_coeff[0,:,:,:,0]
    self.kernel_4p = (((v_dab.T*(cc_da*kernel_den))*cc_da.T)*v_dab).reshape([n*n,n*n])
    #print(type(self.kernel_4p), self.kernel_4p.shape, 'this is just a reference kernel, must be removed later for sure')

    self.xc_code = self.xc_code if xc_code_kw is None else xc_code_kw 
    xc = self.xc_code.split(',')[0]
    if self.verbosity>0: 
      print(__name__, '     xc_code_mf ', self.xc_code_mf)
      print(__name__, ' xc_code_kernel ', self.xc_code_kernel)
      print(__name__, '    xc_code_scf ', self.xc_code_scf)
      print(__name__, '        xc_code ', self.xc_code)
      
    if xc=='CIS' or xc=='HF' or xc=='GW':
      self.kernel_4p -= 0.5*einsum('abcd->bcad', self.kernel_4p.reshape([n,n,n,n])).reshape([n*n,n*n])
    elif xc=='RPA' or xc=='LDA' or xc=='GGA':
      pass
    else :
      print(' ?? xc_code ', self.xc_code, xc)
      raise RuntimeError('??? xc_code ???')

    if xc=='GW':
      w_c_4p = (((v_dab.T*(cc_da*  self.si_c([0.0])[0].real ))*cc_da.T)*v_dab).reshape([n*n,n*n])
      self.kernel_4p -= 0.5*einsum('abcd->bcad', w_c_4p.reshape([n,n,n,n])).reshape([n*n,n*n])


    if hasattr(self, 'mo_energy_gw') and xc=='GW':  # Need to redefine eigenvectors previously set in the class tddft_iter
      self.ksn2e = require(zeros((1,self.nspin,self.norbs)), dtype=self.dtype, requirements='CW')
      self.ksn2e[0,0,:] = self.mo_energy_gw

      ksn2fd = fermi_dirac_occupations(self.telec, self.ksn2e, self.fermi_energy)
      if all(ksn2fd[0,0,:]>self.nfermi_tol):
        print(__name__, self.telec, nfermi_tol, ksn2fd[0,0,:])
        raise RuntimeError('telec is too high?')
      self.ksn2f = (3-self.nspin)*ksn2fd

      self.x = self.mo_coeff_gw[0,:,:,:,0]
      self.nfermi = array([argmax(ksn2fd[0,s,:]<self.nfermi_tol) for s in range(self.nspin)], dtype=int)
      self.vstart = array([argmax(1.0-ksn2fd[0,s,:]>=self.nfermi_tol) for s in range(self.nspin)], dtype=int)
      self.xocc = [self.mo_coeff_gw[0,s,:nfermi,:,0] for s,nfermi in enumerate(self.nfermi)]
      self.xvrt = [self.mo_coeff_gw[0,s,vstart:,:,0] for s,vstart in enumerate(self.vstart)]

  def apply_l0(self, sab, comega=1j*0.0):
    """ This applies the non-interacting four point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    assert sab.size==(self.norbs2), "%r,%r"%(sab.size,self.norbs2)

    sab = sab.reshape([self.norbs,self.norbs])
    self.l0_ncalls+=1

    nm2v = zeros((self.norbs,self.norbs), self.dtypeComplex)
    nm2v[self.vstart[0]:, :self.nfermi[0]] = blas.cgemm(1.0, dot(self.xvrt[0], sab), self.xocc[0].T)
    nm2v[:self.nfermi[0], self.vstart[0]:] = blas.cgemm(1.0, dot(self.xocc[0], sab), self.xvrt[0].T)
    
    for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:],self.ksn2f[0,0,:])):
      for m,[em,fm] in enumerate(zip(self.ksn2e[0,0,:],self.ksn2f[0,0,:])):
        nm2v[n,m] = nm2v[n,m] * (fn-fm) * ( 1.0 / (comega - (em - en)))

    nb2v = blas.cgemm(1.0, nm2v, self.x[0])
    ab2v = blas.cgemm(1.0, self.x[0].T, nb2v)
    return ab2v

  def apply_l0_exp(self, sab, comega=1j*0.0):
    """ This applies the non-interacting four point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    assert sab.size==(self.norbs2), "%r,%r"%(sab.size,self.norbs2)

    sab = sab.reshape([self.norbs,self.norbs])
    self.l0_ncalls+=1
    nb2v = dot(self.x[0], sab)
    nm2v = blas.cgemm(1.0, nb2v, self.x[0].T)
    print(nm2v.dtype)
    print(nm2v[self.vstart[0]:, :self.nfermi[0]])
    print(nm2v[:self.nfermi[0], self.vstart[0]:])
    
    nm2v = zeros((self.norbs,self.norbs), self.dtypeComplex)
    nb2v1 = dot(self.xocc[0], sab)
    nm2v1 = blas.cgemm(1.0, nb2v1, self.xvrt[0].T)

    nb2v2 = dot(self.xvrt[0], sab)
    nm2v2 = blas.cgemm(1.0, nb2v2, self.xocc[0].T)

    nm2v[self.vstart[0]:, :self.nfermi[0]] = nm2v2
    nm2v[:self.nfermi[0], self.vstart[0]:] = nm2v1
    
    print(nm2v.dtype, nm2v1.shape, nm2v1.dtype)
    print(nm2v[self.vstart[0]:, :self.nfermi[0]])
    print(nm2v[:self.nfermi[0], self.vstart[0]:])
    #raise RuntimeError('11')
    
    #nm2v2 = np.copy(nm2v1)
    #for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
      #for m,[em,fm] in enumerate(zip(self.ksn2e[0,0,self.vstart:],self.ksn2f[0,0,self.vstart:])):
        #nm2v1[n,m] = nm2v1[n,m] * (fn-fm) * ( 1.0 / (comega - (em - en)))
    
    #for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:self.nfermi],self.ksn2f[0,0,:self.nfermi])):
      #for m,[em,fm] in enumerate(zip(self.ksn2e[0,0,self.vstart:],self.ksn2f[0,0,self.vstart:])):
        #nm2v2[n,m] = nm2v2[n,m] * (fm-fn) * ( 1.0 / (comega - (en - em)))

    for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:],self.ksn2f[0,0,:])):
      for m,[em,fm] in enumerate(zip(self.ksn2e[0,0,:],self.ksn2f[0,0,:])):
        nm2v[n,m] = nm2v[n,m] * (fn-fm) * ( 1.0 / (comega - (em - en)))
    
    

    nb2v = blas.cgemm(1.0, nm2v, self.x[0])
    ab2v = blas.cgemm(1.0, self.x[0].T, nb2v)
    return ab2v

  def apply_l0_ref(self, sab, comega=1j*0.0):
    """ This applies the non-interacting four point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    assert sab.size==(self.norbs2), "%r,%r"%(sab.size,self.norbs2)

    sab = sab.reshape([self.norbs,self.norbs])
    self.l0_ncalls+=1
    nb2v = dot(self.x[0], sab)
    nm2v = blas.cgemm(1.0, nb2v, self.x[0].T)
    
    for n,[en,fn] in enumerate(zip(self.ksn2e[0,0,:],self.ksn2f[0,0,:])):
      for m,[em,fm] in enumerate(zip(self.ksn2e[0,0,:],self.ksn2f[0,0,:])):
        nm2v[n,m] = nm2v[n,m] * (fn-fm) * ( 1.0 / (comega - (em - en)))

    nb2v = blas.cgemm(1.0, nm2v, self.x[0])
    ab2v = blas.cgemm(1.0, self.x[0].T, nb2v)
    return ab2v


  def seff(self, sext, comega=1j*0.0):
    """ This computes an effective two point field (scalar non-local potential) given an external two point field.
        L = L0 (1 - K L0)^-1
        We want therefore an effective X_eff for a given X_ext
        X_eff = (1 - K L0)^-1 X_ext   or   we need to solve linear equation
        (1 - K L0) X_eff = X_ext  

        The operator (1 - K L0) is named self.sext2seff_matvec """
    
    from scipy.sparse.linalg import gmres, lgmres as gmres_alias, LinearOperator
    assert sext.size==(self.norbs2), "%r,%r"%(sext.size,self.norbs2)

    self.comega_current = comega
    op = LinearOperator((self.norbs2,self.norbs2), matvec=self.sext2seff_matvec, dtype=self.dtypeComplex)
    sext_shape = np.require(sext.reshape(self.norbs2), dtype=self.dtypeComplex, requirements='C')
    resgm,info = gmres_alias(op, sext_shape, tol=self.tddft_iter_tol)
    return (resgm.reshape([self.norbs,self.norbs]),info)

  def sext2seff_matvec(self, sab):
    """ This is operator which we effectively want to have inverted (1 - K L0) and find the action of it's 
    inverse by solving a linear equation with a GMRES method. See the method seff(...)"""
    self.matvec_ncalls+=1 
    
    l0 = self.apply_l0(sab, self.comega_current).reshape(self.norbs2)
    
    l0_reim = require(l0.real, dtype=self.dtype, requirements=["A", "O"])     # real part
    mv_real = dot(self.kernel_4p, l0_reim)
    
    l0_reim = require(l0.imag, dtype=self.dtype, requirements=["A", "O"])     # imaginary part
    mv_imag = dot(self.kernel_4p, l0_reim)

    return sab - (mv_real + 1.0j*mv_imag)

  def apply_l(self, sab, comega=1j*0.0):
    """ This applies the interacting four point Green's function to a suitable vector (e.g. dipole matrix elements)"""
    seff,info = self.seff(sab, comega)
    return self.apply_l0( seff, comega )

  def comp_polariz_nonin_ave(self, comegas):
    """ Non-interacting average polarizability """
    p = np.zeros(len(comegas), dtype=self.dtypeComplex)
    for ixyz in range(3):
      for iw,omega in enumerate(comegas):
        vab = self.apply_l0(self.dip_ab[ixyz], omega)
        p[iw] += (vab*self.dip_ab[ixyz]).sum()/3.0
    return p

  def comp_polariz_inter_ave(self, comegas):
    """ Compute a direction-averaged interacting polarizability  """
    p = np.zeros(len(comegas), dtype=self.dtypeComplex)
    for ixyz in range(3):
      for iw,omega in enumerate(comegas):
        if self.verbosity>1: print(__name__, ixyz, iw)
        vab = self.apply_l(self.dip_ab[ixyz], omega)
        p[iw] += (vab*self.dip_ab[ixyz]).sum()/3.0
    return p
