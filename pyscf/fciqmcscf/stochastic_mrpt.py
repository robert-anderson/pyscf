import numpy
import os

def read_neci_two_pdm_mrpt(filename, norb, directory='.'):
    f = open(os.path.join(directory, filename), 'r')
    norb_active = norb
    two_pdm_active = numpy.zeros( (norb_active, norb_active, norb_active, norb_active) )
    for line in f.readlines():
        linesp = line.split()

        if(int(linesp[0]) != -1):
            # Arrays from neci are '1' indexed
            # We reorder from D[i,j,k,l] = < i^+ j^+ l k >
            # to              D[i,j,k,l] = < i^+ k^+ l j > to match pyscf
            # Therefore, all we need to do is to swap the middle two indices.
            ind1 = int(linesp[0]) - 1
            ind2 = int(linesp[2]) - 1
            ind3 = int(linesp[1]) - 1
            ind4 = int(linesp[3]) - 1
            assert(int(ind1) < norb_active)
            assert(int(ind2) < norb_active)
            assert(int(ind3) < norb_active)
            assert(int(ind4) < norb_active)
            assert(ind1 >= 0)
            assert(ind2 >= 0)
            assert(ind3 >= 0)
            assert(ind4 >= 0)

            two_pdm_active[ind1, ind2, ind3, ind4] = float(linesp[4])

    f.close()
    return two_pdm_active

def read_neci_three_pdm_mrpt(filename, norb, directory='.'):
    f = open(os.path.join(directory, filename), 'r')
    norb_active = norb
    three_pdm_active = numpy.zeros( (norb_active,)*6)
    for line in f.readlines():
        linesp = line.split()

        if(int(linesp[0]) != -1):
            # Arrays from neci are '1' indexed
            # We reorder from D[i,j,k,l] = < i^+ j^+ l k >
            # to              D[i,j,k,l] = < i^+ k^+ l j > to match pyscf
            # Therefore, all we need to do is to swap the middle two indices.
            ind1 = int(linesp[0]) - 1
            ind2 = int(linesp[3]) - 1
            ind3 = int(linesp[1]) - 1
            ind4 = int(linesp[4]) - 1
            ind5 = int(linesp[2]) - 1
            ind6 = int(linesp[5]) - 1
            assert(int(ind1) < norb_active)
            assert(int(ind2) < norb_active)
            assert(int(ind3) < norb_active)
            assert(int(ind4) < norb_active)
            assert(int(ind5) < norb_active)
            assert(int(ind6) < norb_active)
            assert(ind1 >= 0)
            assert(ind2 >= 0)
            assert(ind3 >= 0)
            assert(ind4 >= 0)
            assert(ind5 >= 0)
            assert(ind6 >= 0)

            three_pdm_active[ind1, ind2, ind3, ind4, ind5, ind6] = float(linesp[6])

    f.close()
    return three_pdm_active

def read_neci_four_pdm_mrpt(filename, norb, directory='.'):
    f = open(os.path.join(directory, filename), 'r')
    norb_active = norb
    four_pdm_active = numpy.zeros( (norb_active,)*8 )
    for line in f.readlines():
        linesp = line.split()

        if(int(linesp[0]) != -1):
            # Arrays from neci are '1' indexed
            ind1 = int(linesp[0]) - 1
            ind2 = int(linesp[4]) - 1
            ind3 = int(linesp[1]) - 1
            ind4 = int(linesp[5]) - 1
            ind5 = int(linesp[2]) - 1
            ind6 = int(linesp[6]) - 1
            ind7 = int(linesp[3]) - 1
            ind8 = int(linesp[7]) - 1
            assert(int(ind1) < norb_active)
            assert(int(ind2) < norb_active)
            assert(int(ind3) < norb_active)
            assert(int(ind4) < norb_active)
            assert(int(ind5) < norb_active)
            assert(int(ind6) < norb_active)
            assert(int(ind7) < norb_active)
            assert(int(ind8) < norb_active)
            assert(ind1 >= 0)
            assert(ind2 >= 0)
            assert(ind3 >= 0)
            assert(ind4 >= 0)
            assert(ind5 >= 0)
            assert(ind6 >= 0)
            assert(ind7 >= 0)
            assert(ind8 >= 0)

            four_pdm_active[ind1, ind2, ind3, ind4, ind5, ind6, ind7, ind8] = float(linesp[8])

    f.close()
    return four_pdm_active

def one_from_two_pdm(two_pdm, nelec):
    # Last two indices refer to middle two second quantized operators in the 2RDM
    one_pdm = numpy.einsum('ikjj->ik', two_pdm)
    one_pdm /= (numpy.sum(nelec)-1)
    return one_pdm


def calc_lower_rank_part_of_intermediates(rdm1, rdm2, rdm3, h2e):
    '''
    	f3ac += numpy.einsum('pqra,kibjqcpr->ijkabc', h2e, dm4).transpose(2, 0, 4, 1, 3, 5)
        f3ca += numpy.einsum('rcpq,kibjaqrp->ijkabc', h2e, dm4).transpose(2, 0, 4, 1, 3, 5)
    '''
    f3ac = numpy.zeros(rdm3.shape)
    f3ca = numpy.zeros(rdm3.shape)
    norb = rdm1.shape[0]
    #rdm4 = numpy.zeros((norb,)*8)
    for iorb in range(norb):
	#rdm4[:,iorb,:,:,:,:,iorb,:] += rdm3.transpose(0,2,3,4,5,1)
	# i == p == iorb
	f3ac[iorb,:,:,:,:,:] += numpy.einsum('qra,kbjqcr->jkabc', h2e[iorb,:,:,:], rdm3.transpose(0,2,3,4,5,1))
	# i == r == iorb
        f3ca[iorb,:,:,:,:,:] += numpy.einsum('cpq,kbjaqp->jkabc', h2e[iorb,:,:,:], rdm3.transpose(0,2,3,4,5,1))

       	#rdm4[:,:,:,iorb,:,:,iorb,:] += rdm3.transpose(0,1,2,4,5,3)
	# j == p == iorb
    	f3ac[:,iorb,:,:,:,:] += numpy.einsum('qra,kibqcr->ikabc', h2e[iorb,:,:,:], rdm3.transpose(0,1,2,4,5,3))
	# j == r == iorb
        f3ca[:,iorb,:,:,:,:] += numpy.einsum('cpq,kibaqp->ikabc', h2e[iorb,:,:,:], rdm3.transpose(0,1,2,4,5,3))

	#rdm4[:,:,:,:,:,iorb,iorb,:] += rdm3
	# c == p == iorb
    	f3ac[:,:,:,:,:,iorb] += numpy.einsum('qra,kibjqr->ijkab', h2e[iorb,:,:,:], rdm3)
	# q == r == iorb
        f3ca += numpy.einsum('cp,kibjap->ijkabc', h2e[iorb,:,:,iorb], rdm3)
		
       	#rdm4[:,iorb,:,:,iorb,:,:,:] += rdm3.transpose(0,2,3,1,4,5)
	# i == q == iorb
    	f3ac[iorb,:,:,:,:,:] += numpy.einsum('pra,kbjcpr->jkabc', h2e[:,iorb,:,:], rdm3.transpose(0,2,3,1,4,5))
	# i == a == iorb
        f3ca[iorb,:,:,iorb,:,:] += numpy.einsum('rcpq,kbjqrp->jkbc', h2e, rdm3.transpose(0,2,3,1,4,5))
        
	#rdm4[:,:,:,iorb,iorb,:,:,:] += rdm3
	# j == q == iorb
    	f3ac[:,iorb,:,:,:,:] += numpy.einsum('pra,kibcpr->ikabc', h2e[:,iorb,:,:], rdm3)
	# j == a == iorb
        f3ca[:,iorb,:,iorb,:,:] += numpy.einsum('rcpq,kibqrp->ikbc', h2e, rdm3)

       	#rdm4[:,iorb,iorb,:,:,:,:,:] += rdm3
	# i == b == iorb
        f3ac[iorb,:,:,:,iorb,:] += numpy.einsum('pqra,kjqcpr->jkac', h2e, rdm3)
	# i == b == iorb
        f3ca[iorb,:,:,:,iorb,:] += numpy.einsum('rcpq,kjaqrp->jkac', h2e, rdm3)

        for jorb in range(norb):
            #rdm4[:,iorb,iorb,jorb,:,:,jorb,:] += rdm2.transpose(0,2,3,1)
            # i == b == iorb
            # j == p == jorb
            f3ac[iorb,jorb,:,:,iorb,:] += numpy.einsum('qra,kqcr->kac', h2e[jorb,:,:,:], rdm2.transpose(0,2,3,1))
            # i == b == iorb
            # j == r == jorb
            f3ca[iorb,jorb,:,:,iorb,:] += numpy.einsum('cpq,kaqp->kac', h2e[jorb,:,:,:], rdm2.transpose(0,2,3,1))

	    #rdm4[:,iorb,iorb,:,:,jorb,jorb,:] += rdm2
            # i == b == iorb
            # c == p == jorb
            f3ac[iorb,:,:,:,iorb,jorb] += numpy.einsum('qra,kjqr->jka', h2e[jorb,:,:,:], rdm2)
            # i == b == iorb
            # q == r == jorb
            f3ca[iorb,:,:,:,iorb,:] += numpy.einsum('cp,kjap->jkac', h2e[jorb,:,:,jorb], rdm2)
            
            #rdm4[:,iorb,:,:,iorb,jorb,jorb,:] += rdm2.transpose(0,2,3,1)
            # i == q == iorb
            # c == p == jorb
            f3ac[iorb,:,:,:,:,jorb] += numpy.einsum('ra,kbjr->jkab', h2e[jorb,iorb,:,:], rdm2.transpose(0,2,3,1))
            # i == a == iorb
            # q == r == jorb
            f3ca[iorb,:,:,iorb,:,:] += numpy.einsum('cp,kbjp->jkbc', h2e[jorb,:,:,jorb], rdm2.transpose(0,2,3,1))

	    #rdm4[:,iorb,:,jorb,iorb,:,jorb,:] += rdm2.transpose(0,2,1,3)
            # i == q == iorb
            # j == p == jorb
            f3ac[iorb,jorb,:,:,:,:] += numpy.einsum('ra,kbcr->kabc', h2e[jorb,iorb,:,:], rdm2.transpose(0,2,1,3))
            # i == a == iorb
            # j == r == jorb
            f3ca[iorb,jorb,:,iorb,:,:] += numpy.einsum('cpq,kbqp->kbc', h2e[jorb,:,:,:], rdm2.transpose(0,2,1,3))

	    #rdm4[:,iorb,:,jorb,jorb,:,iorb,:] += rdm2.transpose(0,2,3,1)
            # i == p == iorb
            # j == q == jorb
            f3ac[iorb,jorb,:,:,:,:] += numpy.einsum('ra,kbcr->kabc', h2e[iorb,jorb,:,:], rdm2.transpose(0,2,3,1))
            # i == r == iorb
            # j == a == jorb
            f3ca[iorb,jorb,:,jorb,:,:] += numpy.einsum('cpq,kbqp->kbc', h2e[iorb,:,:,:], rdm2.transpose(0,2,3,1))

	    #rdm4[:,:,:,jorb,jorb,iorb,iorb,:] += rdm2
            # c == p == iorb
            # j == q == jorb
            f3ac[:,jorb,:,:,:,iorb] += numpy.einsum('ra,kibr->ikab', h2e[iorb,jorb,:,:], rdm2)
            # q == r == iorb
            # j == a == jorb
            f3ca[:,jorb,:,jorb,:,:] += numpy.einsum('cp,kibp->ikbc', h2e[iorb,:,:,iorb], rdm2)

	    #rdm4[:,iorb,iorb,jorb,jorb,:,:,:] += rdm2
            # i == b == iorb
            # j == q == jorb
            f3ac[iorb,jorb,:,:,iorb,:] += numpy.einsum('pra,kcpr->kac', h2e[:,jorb,:,:], rdm2)
            # i == b == iorb
            # j == a == jorb
            f3ca[iorb,jorb,:,jorb,iorb,:] += numpy.einsum('rcpq,kqrp->kc', h2e, rdm2)

	    for korb in range(norb):
	        #rdm4[:,iorb,iorb,jorb,jorb,korb,korb,:] += rdm1
                # i == b == iorb
                # j == q == jorb
                # c == p == korb
                f3ac[iorb,jorb,:,:,iorb,korb] += numpy.einsum('ra,kr->ka', h2e[korb,jorb,:,:], rdm1)
                # i == b == iorb
                # j == a == jorb
                # q == r == korb
                f3ca[iorb,jorb,:,jorb,iorb,:] += numpy.einsum('cp,kp->kc', h2e[korb,:,:,korb], rdm1)
    return f3ac.transpose(2, 0, 4, 1, 3, 5), f3ca.transpose(2, 0, 4, 1, 3, 5)

def unreorder_rdm(rdm1, rdm2, inplace=False):
    nmo = rdm1.shape[0]
    if not inplace:
        rdm2 = rdm2.copy()
    for k in range(nmo):
        rdm2[:,k,k,:] += rdm1
    return rdm1, rdm2

def unreorder_dm123(rdm1, rdm2, rdm3, inplace=True):
    if not inplace:
        rdm3 = rdm3.copy()
    norb = rdm1.shape[0]
    for q in range(norb):
        rdm3[:,q,q,:,:,:] += rdm2
        rdm3[:,:,:,q,q,:] += rdm2
        rdm3[:,q,:,:,q,:] += rdm2.transpose(0,2,3,1)
        for s in range(norb):
            rdm3[:,q,q,s,s,:] += rdm1
    rdm1, rdm2 = unreorder_rdm(rdm1, rdm2, inplace)
    return rdm1, rdm2, rdm3

def read_rdms_fciqmc(ncas, nelecas):
    neci_dm2 = read_neci_two_pdm_mrpt('spinfree_TwoRDM.1', ncas)
    if nelecas.__class__==tuple:
        neci_dm1 = one_from_two_pdm(neci_dm2, sum(nelecas))
    else:
        neci_dm1 = one_from_two_pdm(neci_dm2, nelec)
    neci_dm3 = read_neci_three_pdm_mrpt('spinfree_ThreeRDM.1', ncas)
    return neci_dm1, neci_dm2, neci_dm3

def full_nevpt2_intermediates_fciqmc(no_dm1, no_dm2, no_dm3, ncas, h2e):
    neci_nevpt2_intermediate = read_neci_three_pdm_mrpt('spinfree_NEVPT2_AUX.1', ncas)
    f3ac, f3ca = calc_lower_rank_part_of_intermediates(no_dm1, no_dm2, no_dm3, h2e)
    f3ac+=neci_nevpt2_intermediate.transpose(2, 3, 4, 0, 1, 5)
    f3ca+=neci_nevpt2_intermediate.transpose(0, 4, 3, 2, 5, 1)
    return f3ac, f3ca













