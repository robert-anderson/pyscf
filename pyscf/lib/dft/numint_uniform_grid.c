/* Copyright 2014-2018 The PySCF Developers. All Rights Reserved.

   Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

 *
 * Fast numerical integration on uniform grids.
 * (See also cp2k multigrid algorithm)
 *
 * Author: Qiming Sun <osirpt.sun@gmail.com>
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <complex.h>
#include "config.h"
#include "cint.h"
#include "np_helper/np_helper.h"
#include "gto/ft_ao.h"
#include "gto/grid_ao_drv.h"
#include "vhf/fblas.h"

#ifndef __USE_ISOC99
#define rint(x) (int)round(x)
#endif

#define PLAIN           0
#define HERMITIAN       1
#define ANTIHERMI       2
#define SYMMETRIC       3
#define OF_CMPLX        2
#define EXPCUTOFF15     40
#define EXPMAX          700
#define EXPMIN          -700

#define SQUARE(x)       (*(x) * *(x) + *(x+1) * *(x+1) + *(x+2) * *(x+2))

double CINTsquare_dist(const double *r1, const double *r2);
double CINTcommon_fac_sp(int l);
void c2s_sph_1e(double *opij, double *gctr, int *dims,
                CINTEnvVars *envs, double *cache);
void c2s_cart_1e(double *opij, double *gctr, int *dims,
                 CINTEnvVars *envs, double *cache);

int CINTinit_int1e_EnvVars(CINTEnvVars *envs, const int *ng, const int *shls,
                           const int *atm, const int natm,
                           const int *bas, const int nbas, const double *env);

static const int _LEN_CART[] = {
        1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 66, 78, 91, 105, 120, 136
};
static const int _CUM_LEN_CART[] = {
        1, 4, 10, 20, 35, 56, 84, 120, 165, 220, 286, 364, 455, 560, 680, 816,
};
#define STARTX_IF_L_DEC1(l)     0
#define STARTY_IF_L_DEC1(l)     (((l)<2)?0:_LEN_CART[(l)-2])
#define STARTZ_IF_L_DEC1(l)     (_LEN_CART[(l)-1]-1)

/*
 * rcut is the distance over which the integration (from rcut to infty) is
 * smaller than the required precision
 * integral ~= \int_{rcut}^infty r^{l+2} exp(-alpha r^2) dr
 *
 * * if l is odd:
 *   integral = \sum_n (l+1)!!/(l+3-2n)!! * rcut^{l+3-2n}/(2 alpha)^n
 *                     * exp(-alpha {rcut}^2)
 *
 * * elif l is even and rcut > 1:
 *   integral < [\sum_{n<=l/2+1} (l+1)!!/(l+3-2n)!! * rcut^{l+3-2n}/(2 alpha)^n
 *               + 1/(2 alpha)^(l/2+2)] * exp(-alpha {rcut}^2)
 *
 * * elif l is even and rcut < 1:
 *   integral < [\sum_{n<=l/2+1} (l+1)!!/(l+3-2n)!! * rcut^{l+3-2n}/(2 alpha)^n] * exp(-alpha {rcut}^2)
 *              + (l+1)!! / (2 alpha)^{l/2+1} * \sqrt(pi/alpha)/2
 */
static double gto_rcut(double alpha, int l, double c, double log_prec)
{
        // Add penalty 1e-2 for other integral factors and coefficients
        log_prec -= 5 + log(4*M_PI);

        double log_c = log(fabs(c));
        double prod = 0;
        double r = 5.;
        double log_2a = log(2*alpha);
        double log_r = log(r);

        if (2*log_r + log_2a > 1) { // r^2 >~ 3/(2a)
                prod = (l+1) * log_r - log_2a;
        } else {
                prod = -(l+4)/2 * log_2a;
        }

        //log_r = .5 * (prod / alpha);
        //if (2*log_r + log_2a > 1) {
        //        prod = (l+1) * log_r - log_2a;
        //} else {
        //        prod = -(l+4)/2 * log_2a;
        //}

        prod += log_c - log_prec;
        if (prod < alpha) {
                // if rcut < 1, estimataion based on exp^{-a*rcut^2}
                prod = log_c - log_prec;
        }
        if (prod > 0) {
                r = sqrt(prod / alpha);
        } else {
                r = 0;
        }
        return r;
}

static void _orth_components(double *xs_exp, int *img_slice, int *grid_slice,
                             double *a, double xi, double xij, double aij,
                             int periodic, int nx_per_cell, int topl,
                             double x_frac, double cutoff, double heights_inv,
                             double *cache)
{
        double edge0 = x_frac - cutoff * heights_inv;
        double edge1 = x_frac + cutoff * heights_inv;

        int nimg0 = 0;
        int nimg1 = 1;
        if (periodic) {
                nimg0 = (int)floor(edge0);
                nimg1 = (int)ceil (edge1);
        }

        int nx0 = (int)floor(edge0 * nx_per_cell);
        int nx1 = (int)ceil (edge1 * nx_per_cell);
        // to ensure nx0, nx1 in unit cell
        if (periodic) {
                nx0 = (nx0 + nimg1 * nx_per_cell) % nx_per_cell;
                nx1 = (nx1 + nimg1 * nx_per_cell) % nx_per_cell;
        } else {
                nx0 = MIN(nx0, nx_per_cell);
                nx0 = MAX(nx0, 0);
                nx1 = MIN(nx1, nx_per_cell);
                nx1 = MAX(nx1, 0);
        }
        img_slice[0] = nimg0;
        img_slice[1] = nimg1;
        grid_slice[0] = nx0;
        grid_slice[1] = nx1;

        int nimg = nimg1 - nimg0;
        int nmx = nimg * nx_per_cell;

        int i, m, l;
        double *px0;

        double *gridx = cache;
        double *xs_all = cache + nimg * nx_per_cell;
        int grid_close_to_xij = rint(x_frac * nx_per_cell);
        if (!periodic) {
                xs_all = xs_exp;
                grid_close_to_xij = MIN(grid_close_to_xij, nx_per_cell);
                grid_close_to_xij = MAX(grid_close_to_xij, 0);
        }

        double img0_x = *a * nimg0;
        double dx = *a / nx_per_cell;
        double base_x = img0_x + dx * grid_close_to_xij;
        double x0xij = base_x - xij;
        double _x0x0 = -aij * x0xij * x0xij;
        if (_x0x0 < EXPMIN) {
                for (i = 0; i < (topl+1)*nx_per_cell; i++) {
                        xs_all[i] = 0;
                }
                return;
        }

        double _dxdx = -aij * dx * dx;
        double _x0dx = -2 * aij * x0xij * dx;
        double exp_dxdx = exp(_dxdx);
        double exp_2dxdx = exp_dxdx * exp_dxdx;
        double exp_x0dx = exp(_x0dx + _dxdx);
        double exp_x0x0 = exp(_x0x0);

        for (i = grid_close_to_xij; i < nmx; i++) {
                xs_all[i] = exp_x0x0;
                exp_x0x0 *= exp_x0dx;
                exp_x0dx *= exp_2dxdx;
        }

        exp_x0dx = exp(_dxdx - _x0dx);
        exp_x0x0 = exp(_x0x0);
        for (i = grid_close_to_xij-1; i >= 0; i--) {
                exp_x0x0 *= exp_x0dx;
                exp_x0dx *= exp_2dxdx;
                xs_all[i] = exp_x0x0;
        }

        if (topl > 0) {
                double x0xi = img0_x - xi;
                for (i = 0; i < nmx; i++) {
                        gridx[i] = x0xi + i * dx;
                }
                for (l = 1; l <= topl; l++) {
                        px0 = xs_all + (l-1) * nmx;
                        for (i = 0; i < nmx; i++) {
                                px0[nmx+i] = px0[i] * gridx[i];
                        }
                }
        }

        if (periodic) {
                for (l = 0; l <= topl; l++) {
                        px0 = xs_all + l * nmx;
                        for (i = 0; i < nx_per_cell; i++) {
                                xs_exp[i] = px0[i];
                        }
                        for (m = 1; m < nimg; m++) {
                                px0 = xs_all + l * nmx + m*nx_per_cell;
                                for (i = 0; i < nx_per_cell; i++) {
                                        xs_exp[l*nx_per_cell+i] += px0[i];
                                }
                        }
                }
        }
}
static int _has_overlap(int nx0, int nx1, int nx_per_cell)
{
        return nx0 < nx1;
}

void NUMINTeval_3d_orth(double *out, int floorl, int topl,
                        double ai, double aj, double fac, double log_prec,
                        int dimension, double *a, double *b, int *mesh,
                        double *weights, CINTEnvVars *envs, double *cache)
{
        const double aij = ai + aj;
        const double *ri = envs->ri;
        const double *rj = envs->rj;
        double rij[3];
        rij[0] = (ai * ri[0] + aj * rj[0]) / aij;
        rij[1] = (ai * ri[1] + aj * rj[1]) / aij;
        rij[2] = (ai * ri[2] + aj * rj[2]) / aij;

        double cutoff = gto_rcut(aij, topl, fac, log_prec);
        double xij_frac = rij[0] * b[0];
        double yij_frac = rij[1] * b[4];
        double zij_frac = rij[2] * b[8];
        double xheights_inv = b[0];
        double yheights_inv = b[4];
        double zheights_inv = b[8];

        int l1 = topl + 1;
        double *xs_exp = cache;
        double *ys_exp = xs_exp + l1 * mesh[0];
        double *zs_exp = ys_exp + l1 * mesh[1];
        cache = zs_exp + l1 * mesh[2];

        int img_slice[6];
        int grid_slice[6];
        _orth_components(xs_exp, img_slice, grid_slice, a,
                         ri[0], rij[0], aij, (dimension>=1), mesh[0], topl,
                         xij_frac, cutoff, xheights_inv, cache);
        _orth_components(ys_exp, img_slice+2, grid_slice+2, a+4,
                         ri[1], rij[1], aij, (dimension>=2), mesh[1], topl,
                         yij_frac, cutoff, yheights_inv, cache);
        _orth_components(zs_exp, img_slice+4, grid_slice+4, a+8,
                         ri[2], rij[2], aij, (dimension>=3), mesh[2], topl,
                         zij_frac, cutoff, zheights_inv, cache);

        int nimgx0 = img_slice[0];
        int nimgx1 = img_slice[1];
        int nimgy0 = img_slice[2];
        int nimgy1 = img_slice[3];
        int nimgz0 = img_slice[4];
        int nimgz1 = img_slice[5];
        int nimgx = nimgx1 - nimgx0;
        int nimgy = nimgy1 - nimgy0;
        int nimgz = nimgz1 - nimgz0;

        int nx0 = grid_slice[0];
        int nx1 = grid_slice[1];
        int ny0 = grid_slice[2];
        int ny1 = grid_slice[3];
        int nz0 = grid_slice[4];
        int nz1 = grid_slice[5];
        int ngridx = nx1 - nx0;
        int ngridy = ny1 - ny0;
        int ngridz = nz1 - nz0;

        int lx, ly, lz;
        int l, n, i;

        if ((nimgx == 1 && ngridx == 0) ||
            (nimgy == 1 && ngridy == 0) ||
            (nimgz == 1 && ngridz == 0)) {
                int nout = _CUM_LEN_CART[topl] - _CUM_LEN_CART[floorl]
                         + _LEN_CART[floorl];
                for (n = 0; n < nout; n++) {
                        out[n] = 0;
                }
                return;
        }

        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        int xcols = mesh[1] * mesh[2];
        int ycols = mesh[2];
        double *weightyz = cache;
        double *weightz = weightyz + l1*xcols;
        double *pz, *pweightz;
        double val;

        if (nimgx == 1) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
                       &fac, weights+nx0*xcols, &xcols, xs_exp+nx0, mesh,
                       &D0, weightyz, &xcols);
        } else if (nimgx == 2 && !_has_overlap(nx0, nx1, mesh[0])) {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &nx1,
                       &fac, weights, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
                ngridx = mesh[0] - nx0;
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, &ngridx,
                       &fac, weights+nx0*xcols, &xcols, xs_exp+nx0, mesh,
                       &D1, weightyz, &xcols);
        } else {
                dgemm_(&TRANS_N, &TRANS_N, &xcols, &l1, mesh,
                       &fac, weights, &xcols, xs_exp, mesh,
                       &D0, weightyz, &xcols);
        }

        if (nimgy == 1) {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
                               &D1, weightyz+lx*xcols+ny0*ycols, &ycols, ys_exp+ny0, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                }
        } else if (nimgy == 2 && !_has_overlap(ny0, ny1, mesh[1])) {
                ngridy = mesh[1] - ny0;
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ny1,
                               &D1, weightyz+lx*xcols, &ycols, ys_exp, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, &ngridy,
                               &D1, weightyz+lx*xcols+ny0*ycols, &ycols, ys_exp+ny0, mesh+1,
                               &D1, weightz+lx*l1*ycols, &ycols);
                }
        } else {
                for (lx = 0; lx <= topl; lx++) {
                        dgemm_(&TRANS_N, &TRANS_N, &ycols, &l1, mesh+1,
                               &D1, weightyz+lx*xcols, &ycols, ys_exp, mesh+1,
                               &D0, weightz+lx*l1*ycols, &ycols);
                }
        }

        if (nimgz == 1) {
                for (n = 0, l = floorl; l <= topl; l++) {
                for (lx = l; lx >= 0; lx--) {
                for (ly = l - lx; ly >= 0; ly--, n++) {
                        lz = l - lx - ly;
                        pz = zs_exp + lz * mesh[2];
                        pweightz = weightz + (lx * l1 + ly) * mesh[2];
                        val = 0;
                        for (i = nz0; i < nz1; i++) {
                                val += pweightz[i] * pz[i];
                        }
                        out[n] = val;
                } } }
        //TODO:} elif (nimgz == 2 && !_has_overlap(nz0, nz1, mesh[2])) {
        } else {
                for (n = 0, l = floorl; l <= topl; l++) {
                for (lx = l; lx >= 0; lx--) {
                for (ly = l - lx; ly >= 0; ly--, n++) {
                        lz = l - lx - ly;
                        pz = zs_exp + lz * mesh[2];
                        pweightz = weightz + (lx * l1 + ly) * mesh[2];
                        val = 0;
                        for (i = 0; i < mesh[2]; i++) {
                                val += pweightz[i] * pz[i];
                        }
                        out[n] = val;
                } } }
        }
}

static void plain_prim_to_ctr(double *gc, const size_t nf, double *gp,
                              const int nprim, const int nctr,
                              const double *coeff, int empty)
{
        size_t n, i;
        double c;

        if (empty) {
                for (n = 0; n < nctr; n++) {
                        c = coeff[nprim*n];
                        for (i = 0; i < nf; i++) {
                                gc[i] = gp[i] * c;
                        }
                        gc += nf;
                }
        } else {
                for (n = 0; n < nctr; n++) {
                        c = coeff[nprim*n];
                        if (c != 0) {
                                for (i = 0; i < nf; i++) {
                                        gc[i] += gp[i] * c;
                                }
                        }
                        gc += nf;
                }
        }
}

static double max_pgto_coeff(double *coeff, int nprim, int nctr, int prim_id)
{
        int i;
        double maxc = 0;
        for (i = 0; i < nctr; i++) {
                maxc = MAX(maxc, fabs(coeff[i*nprim+prim_id]));
        }
        return maxc;
}

static int _orth_loop(double *out, double fac, double log_prec,
                      int dimension, double *a, double *b,
                      int *mesh, double *weights,
                      CINTEnvVars *envs, double *cache)
{
        const int *shls  = envs->shls;
        const int *bas = envs->bas;
        double *env = envs->env;
        const int i_sh = shls[0];
        const int j_sh = shls[1];
        const int i_l = envs->i_l;
        const int j_l = envs->j_l;
        const int i_ctr = envs->x_ctr[0];
        const int j_ctr = envs->x_ctr[1];
        const int i_prim = bas(NPRIM_OF, i_sh);
        const int j_prim = bas(NPRIM_OF, j_sh);
        const int nf = envs->nf;
        double *ri = envs->ri;
        double *rj = envs->rj;
        double *ai = env + bas(PTR_EXP, i_sh);
        double *aj = env + bas(PTR_EXP, j_sh);
        double *ci = env + bas(PTR_COEFF, i_sh);
        double *cj = env + bas(PTR_COEFF, j_sh);
        double fac1i;
        double aij, eij;
        int ip, jp, n;
        int empty[3] = {1, 1, 1};
        int *jempty = empty + 0;
        int *iempty = empty + 1;
        //int *gempty = empty + 2;
        const int offset_g1d = _CUM_LEN_CART[i_l] - _LEN_CART[i_l];
        const int len_g1d = _CUM_LEN_CART[i_l+j_l] - offset_g1d;
        const size_t leni = len_g1d * i_ctr;
        const size_t lenj = len_g1d * i_ctr * j_ctr;
        double *gctrj = cache;
        double *gctri = gctrj + lenj;
        double *g = gctri + leni;
        double *log_iprim_max = g + len_g1d;
        double *log_jprim_max = log_iprim_max + i_prim;
        cache += lenj + leni + len_g1d + i_prim + j_prim;

        for (ip = 0; ip < i_prim; ip++) {
                log_iprim_max[ip] = log(max_pgto_coeff(ci, i_prim, i_ctr, ip));
        }
        for (jp = 0; jp < j_prim; jp++) {
                log_jprim_max[jp] = log(max_pgto_coeff(cj, j_prim, j_ctr, jp));
        }
        void (*eval_3d)() = envs->f_gout;
        double rrij = CINTsquare_dist(ri, rj);
        double fac1 = fac * CINTcommon_fac_sp(i_l) * CINTcommon_fac_sp(j_l);
        double logcc;

        *jempty = 1;
        for (jp = 0; jp < j_prim; jp++) {
                *iempty = 1;
                for (ip = 0; ip < i_prim; ip++) {
                        aij = ai[ip] + aj[jp];
                        eij = (ai[ip] * aj[jp] / aij) * rrij;
                        logcc = log_iprim_max[ip] + log_jprim_max[jp];
                        if (eij-logcc > EXPCUTOFF15) { //(eij > EXPCUTOFF)?
                                continue;
                        }

                        fac1i = fac1 * exp(-eij);
                        (*eval_3d)(g, i_l, i_l+j_l, ai[ip], aj[jp],
                                   fac1i, logcc+log_prec, dimension,
                                   a, b, mesh, weights, envs, cache);
                        plain_prim_to_ctr(gctri, len_g1d, g,
                                          i_prim, i_ctr, ci+ip, *iempty);
                        *iempty = 0;
                }
                if (!*iempty) {
                        plain_prim_to_ctr(gctrj, i_ctr*len_g1d, gctri,
                                          j_prim, j_ctr, cj+jp, *jempty);
                        *jempty = 0;
                }
        }

        if (!*jempty) {
                for (n = 0; n < i_ctr*j_ctr; n++) {
                        GTOplain_vrr2d(out+n*nf, gctrj+n*len_g1d, cache, envs);
                }
        }

        return !*jempty;
}

static int _orth_cache_size(int *mesh, CINTEnvVars *envs)
{
        const int i_ctr = envs->x_ctr[0];
        const int j_ctr = envs->x_ctr[1];
        const int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        const size_t nc = envs->nf * i_ctr * j_ctr;
        const int l = envs->i_l + envs->j_l;
        const int l1 = l + 1;
        size_t cache_size = 0;
        cache_size += _CUM_LEN_CART[l] * (1 + i_ctr + i_ctr * j_ctr);
        cache_size += l1 * (mesh[0] + mesh[1] + mesh[2]);
        cache_size += l1 * mesh[1] * mesh[2];
        cache_size += l1 * l1 * mesh[2];
        cache_size += 20; // i_prim + j_prim
        return nc * n_comp + MAX(cache_size, envs->nf * 8 * OF_CMPLX);
}

int NUMINT_orth_drv(double *out, int *dims, void (*f_c2s)(),
                    double fac, double log_prec, int dimension,
                    double *a, double *b, int *mesh, double *weights,
                    CINTEnvVars *envs, double *cache)
{
        if (out == NULL) {
                return _orth_cache_size(mesh, envs);
        }

        const int i_ctr = envs->x_ctr[0];
        const int j_ctr = envs->x_ctr[1];
        const int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        const size_t nc = envs->nf * i_ctr * j_ctr;
        double *gctr = cache;
        cache += nc * n_comp;
        size_t n;
        int has_value = _orth_loop(gctr, fac, log_prec, dimension,
                                   a, b, mesh, weights, envs, cache);
        if (!has_value) {
                for (n = 0; n < nc*n_comp; n++) { gctr[n] = 0; }
        }

        int counts[4];
        if (f_c2s == &c2s_sph_1e) {
                counts[0] = (envs->i_l*2+1) * i_ctr;
                counts[1] = (envs->j_l*2+1) * j_ctr;
        } else { // f_c2s == &GTO_ft_c2s_cart
                counts[0] = envs->nfi * i_ctr;
                counts[1] = envs->nfj * j_ctr;
        }
        if (dims == NULL) {
                dims = counts;
        }
        size_t nout = dims[0] * dims[1];

        if (has_value) {
                for (n = 0; n < n_comp; n++) {
                        (*f_c2s)(out+nout*n, gctr+nc*n, dims, envs, cache);
                }
        }
        return has_value;
}



static void _nonorth_components(double *xs_exp, int *img_slice, int *grid_slice,
                                int periodic, int nx_per_cell, int topl,
                                double xi_frac, double xij_frac, double cutoff,
                                double heights_inv)
{
        double edge0 = xij_frac - cutoff * heights_inv;
        double edge1 = xij_frac + cutoff * heights_inv;

        int nimg0 = 0;
        int nimg1 = 1;
        if (periodic) {
                nimg0 = (int)floor(edge0);
                nimg1 = (int)ceil (edge1);
        }

        int nx0 = (int)floor(edge0 * nx_per_cell);
        int nx1 = (int)ceil (edge1 * nx_per_cell);
        if (!periodic) {
                // to ensure nx0, nx1 in unit cell
                nx0 = MIN(nx0, nx_per_cell);
                nx0 = MAX(nx0, 0);
                nx1 = MIN(nx1, nx_per_cell);
                nx1 = MAX(nx1, 0);
        }
        img_slice[0] = nimg0;
        img_slice[1] = nimg1;
        grid_slice[0] = nx0;
        grid_slice[1] = nx1;

        int nx = nx1 - nx0;
        int i, l;
        double x0;
        double dx = 1. / nx_per_cell;
        double *pxs_exp;
        for (i = 0; i < nx; i++) {
                xs_exp[i] = 1;
        }
        for (l = 1; l <= topl; l++) {
                pxs_exp = xs_exp + (l-1) * nx;
                x0 = nx0 * dx - xi_frac;
                for (i = 0; i < nx; i++, x0+=dx) {
                        xs_exp[l*nx+i] = x0 * pxs_exp[i];
                }
        }
}
static void _nonorth_dot_z(double *val, double *weights,
                           int nz0, int nz1, int grid_close_to_zij,
                           double e_z0z0, double e_z0dz, double e_dzdz)
{
        int iz;
        if (e_z0z0 == 0) {
                for (iz = 0; iz < nz1-nz0; iz++) {
                        val[iz] = 0;
                }
                return;
        }

        double exp_2dzdz = e_dzdz * e_dzdz;
        double exp_z0z0, exp_z0dz;

        exp_z0z0 = e_z0z0;
        exp_z0dz = e_z0dz * e_dzdz;
        for (iz = grid_close_to_zij; iz < nz1; iz++) {
                val[iz-nz0] = weights[iz] * exp_z0z0; //FIXME = weights[mod(iz,mesh[2])] * exp_z0z0;
                exp_z0z0 *= exp_z0dz;
                exp_z0dz *= exp_2dzdz;
        }

        exp_z0z0 = e_z0z0;
        exp_z0dz = e_dzdz / e_z0dz;
        for (iz = grid_close_to_zij-1; iz >= nz0; iz--) {
                exp_z0z0 *= exp_z0dz;
                exp_z0dz *= exp_2dzdz;
                val[iz-nz0] = weights[iz] * exp_z0z0; //FIXME = weights[mod(iz,mesh[2])] * exp_z0z0;
        }
}

void NUMINTeval_3d_nonorth(double *out, int floorl, int topl,
                           double ai, double aj, double fac, double log_prec,
                           int dimension, double *a, double *b, int *mesh,
                           double *weights, CINTEnvVars *envs, double *cache)
{
//FIXME: periodic condition
        const double aij = ai + aj;
        const double *ri = envs->ri;
        const double *rj = envs->rj;
        double rij[3];
        rij[0] = (ai * ri[0] + aj * rj[0]) / aij;
        rij[1] = (ai * ri[1] + aj * rj[1]) / aij;
        rij[2] = (ai * ri[2] + aj * rj[2]) / aij;
        // rij_frac = einsum('ij,j->ik', b, rij)
        double xij_frac = rij[0] * b[0] + rij[1] * b[1] + rij[2] * b[2];
        double yij_frac = rij[0] * b[3] + rij[1] * b[4] + rij[2] * b[5];
        double zij_frac = rij[0] * b[6] + rij[1] * b[7] + rij[2] * b[8];
        double xi_frac = ri[0] * b[0] + ri[1] * b[1] + ri[2] * b[2];
        double yi_frac = ri[0] * b[3] + ri[1] * b[4] + ri[2] * b[5];
        double zi_frac = ri[0] * b[6] + ri[1] * b[7] + ri[2] * b[8];

        double cutoff = gto_rcut(aij, topl, fac, log_prec);
        double xheights_inv = sqrt(SQUARE(b  ));
        double yheights_inv = sqrt(SQUARE(b+3));
        double zheights_inv = sqrt(SQUARE(b+6));

        int l1 = topl + 1;
        int l1l1 = l1 * l1;
        int img_slice[6];
        int grid_slice[6];
        double *xs_exp, *ys_exp, *zs_exp;
        xs_exp = cache;
        _nonorth_components(xs_exp, img_slice, grid_slice, (dimension>=1),
                            mesh[0], topl, xi_frac, xij_frac, cutoff,
                            xheights_inv);
        int nx0 = grid_slice[0];
        int nx1 = grid_slice[1];
        int ngridx = nx1 - nx0;

        ys_exp = xs_exp + l1 * ngridx;
        _nonorth_components(ys_exp, img_slice+2, grid_slice+2, (dimension>=2),
                            mesh[1], topl, yi_frac, yij_frac, cutoff,
                            yheights_inv);
        int ny0 = grid_slice[2];
        int ny1 = grid_slice[3];
        int ngridy = ny1 - ny0;

        zs_exp = ys_exp + l1 * ngridy;
        _nonorth_components(zs_exp, img_slice+4, grid_slice+4, (dimension>=3),
                            mesh[2], topl, zi_frac, zij_frac, cutoff,
                            zheights_inv);
        int nz0 = grid_slice[4];
        int nz1 = grid_slice[5];
        int ngridz = nz1 - nz0;
        cache = zs_exp + l1 * ngridz;

        int nimgx0 = img_slice[0];
        int nimgx1 = img_slice[1];
        int nimgy0 = img_slice[2];
        int nimgy1 = img_slice[3];
        int nimgz0 = img_slice[4];
        int nimgz1 = img_slice[5];
        int nimgx = nimgx1 - nimgx0;
        int nimgy = nimgy1 - nimgy0;
        int nimgz = nimgz1 - nimgz0;

        if ((nimgx == 1 && ngridx == 0) ||
            (nimgy == 1 && ngridy == 0) ||
            (nimgz == 1 && ngridz == 0)) {
                int n;
                for (n = 0; n < l1l1*l1; n++) {
                        out[n] = 0;
                }
                return;
        }

        const char TRANS_T = 'T';
        const char TRANS_N = 'N';
        const double D0 = 0;
        const double D1 = 1;
        // aa = einsum('ij,kj->ik', a, a)
        //double aa[9];
        //int n3 = 3;
        //dgemm_(&TRANS_T, &TRANS_N, &n3, &n3, &n3,
        //       &aij, a, &n3, a, &n3, &D0, aa, &n3);
        double aa_xx = aij * (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
        double aa_xy = aij * (a[0] * a[3] + a[1] * a[4] + a[2] * a[5]);
        double aa_xz = aij * (a[0] * a[6] + a[1] * a[7] + a[2] * a[8]);
        double aa_yy = aij * (a[3] * a[3] + a[4] * a[4] + a[5] * a[5]);
        double aa_yz = aij * (a[3] * a[6] + a[4] * a[7] + a[5] * a[8]);
        double aa_zz = aij * (a[6] * a[6] + a[7] * a[7] + a[8] * a[8]);

        int ix, iy, n;
        double dx = 1. / mesh[0];
        double dy = 1. / mesh[1];
        double dz = 1. / mesh[2];

        double *cache_xyz = cache;
        double *weight_x = cache_xyz + l1 * l1l1;
        double *weight_z = weight_x + l1l1 * ngridx;
        double *weight_yz = weight_z + l1 * ngridz;
        double *pweights;

        //int grid_close_to_xij = rint(xij_frac * mesh[0]);
        int grid_close_to_yij = rint(yij_frac * mesh[1]);
        int grid_close_to_zij = rint(zij_frac * mesh[2]);
        //if (dimension < 1) {
        //        grid_close_to_xij = MIN(grid_close_to_xij, mesh[0]);
        //        grid_close_to_xij = MAX(grid_close_to_xij, 0);
        //}
        if (dimension < 2) {
                grid_close_to_yij = MIN(grid_close_to_yij, mesh[1]);
                grid_close_to_yij = MAX(grid_close_to_yij, 0);
        }
        if (dimension < 3) {
                grid_close_to_zij = MIN(grid_close_to_zij, mesh[2]);
                grid_close_to_zij = MAX(grid_close_to_zij, 0);
        }

        double img0_x = 0;
        double img0_y = 0;
        double img0_z = 0;
        double base_x = img0_x;// + dx * grid_close_to_xij;
        double base_y = img0_y + dy * grid_close_to_yij;
        double base_z = img0_z + dz * grid_close_to_zij;
        double x0xij = base_x - xij_frac;
        double y0yij = base_y - yij_frac;
        double z0zij = base_z - zij_frac;

        double _dydy = -dy * dy * aa_yy;
        double _dzdz = -dz * dz * aa_zz;
        double _dydz = -dy * dz * aa_yz * 2;
        double exp_dydy = exp(_dydy);
        double exp_2dydy = exp_dydy * exp_dydy;
        double exp_dzdz = exp(_dzdz);
        double exp_dydz = exp(_dydz);
        double x1xij, tmpx, tmpy, tmpz;
        double _xyz0xyz0, _xyz0dy, _xyz0dz;
        double exp_xyz0xyz0, exp_xyz0dz;
        double exp_y0dy, exp_z0z0, exp_z0dz;

        // FIXME: consider the periodicity for [nx0:nx1]
        for (ix = nx0; ix < nx1; ix++) {
                x1xij = x0xij + ix*dx;
                tmpx = x1xij * aa_xx + y0yij * aa_xy + z0zij * aa_xz;
                tmpy = x1xij * aa_xy + y0yij * aa_yy + z0zij * aa_yz;
                tmpz = x1xij * aa_xz + y0yij * aa_yz + z0zij * aa_zz;
                _xyz0xyz0 = -x1xij * tmpx - y0yij * tmpy - z0zij * tmpz;
                if (_xyz0xyz0 < EXPMIN) {
// _xyz0dy (and _xyz0dz) can be very big, even greater than the effective range
// of exp function (and produce inf).  When exp_xyz0xyz0 is 0 and exp_xyz0dy is
// inf, the product will be ill-defined.  |_xyz0dy| should be smaller than
// |_xyz0xyz0| in any situations.  exp_xyz0xyz0 should dominate the product
// exp_xyz0xyz0 * exp_xyz0dy.  When exp_xyz0xyz0 is 0, the product should be 0.
// All the rest exp products should be smaller than exp_xyz0xyz0 and can be
// neglected.
                        pweights = weight_x + (ix-nx0)*l1l1;
                        for (n = 0; n < l1l1; n++) {
                                pweights[n] = 0;
                        }
                        continue;
                }
                _xyz0dy = -2 * dy * tmpy;
                _xyz0dz = -2 * dz * tmpz;
                exp_xyz0xyz0 = fac * exp(_xyz0xyz0);
                exp_xyz0dz = exp(_xyz0dz);

                //exp_xyz0dy = exp(_xyz0dy);
                //exp_y0dy = exp_xyz0dy * exp_dydy;
                exp_y0dy = exp(_xyz0dy + _dydy);
                exp_z0z0 = exp_xyz0xyz0;
                exp_z0dz = exp_xyz0dz;
                for (iy = grid_close_to_yij; iy < ny1; iy++) {
                        //pweights = weights + (ix * mesh[1] + iy) * mesh[2];
                        //exp_z0z0p = exp_z0z0;
                        //exp_z0dzp = exp_z0dz * exp_dzdz;
                        //for (iz = grid_close_to_zij; iz < mesh[2]; iz++) {
                        //        val += pweights[iz] * exp_z0z0p;
                        //        exp_z0z0p *= exp_z0dzp;
                        //        exp_z0dzp *= exp_2dzdz;
                        //}
                        //exp_z0z0p = exp_z0z0;
                        //exp_z0dzp = exp_dzdz / exp_z0dz;
                        //for (iz = grid_close_to_zij-1; iz >= 0; iz--) {
                        //        exp_z0z0p *= exp_z0dzp;
                        //        exp_z0dzp *= exp_2dzdz;
                        //        val += pweights[iz] * exp_z0z0p;
                        //}
                        pweights = weights + (ix * mesh[1] + iy) * mesh[2]; // FIXME ix -> mod(ix,mesh[0]) for periodicity
                        _nonorth_dot_z(weight_yz+(iy-ny0)*ngridz, pweights,
                                       nz0, nz1, grid_close_to_zij,
                                       exp_z0z0, exp_z0dz, exp_dzdz);
                        exp_z0z0 *= exp_y0dy;
                        exp_z0dz *= exp_dydz;
                        exp_y0dy *= exp_2dydy;
                }

                exp_y0dy = exp(_dydy - _xyz0dy);
                exp_z0z0 = exp_xyz0xyz0;
                exp_z0dz = exp_xyz0dz;
                for (iy = grid_close_to_yij-1; iy >= ny0; iy--) {
                        exp_z0z0 *= exp_y0dy;
                        exp_z0dz /= exp_dydz;
                        exp_y0dy *= exp_2dydy;
                        pweights = weights + (ix * mesh[1] + iy) * mesh[2];
                        _nonorth_dot_z(weight_yz+(iy-ny0)*ngridz, pweights,
                                       nz0, nz1, grid_close_to_zij,
                                       exp_z0z0, exp_z0dz, exp_dzdz);
                }

                dgemm_(&TRANS_N, &TRANS_N, &ngridz, &l1, &ngridy,
                       &D1, weight_yz, &ngridz, ys_exp, &ngridy,
                       &D0, weight_z, &ngridz);
                dgemm_(&TRANS_T, &TRANS_N, &l1, &l1, &ngridz,
                       &D1, zs_exp, &ngridz, weight_z, &ngridz,
                       &D0, weight_x+(ix-nx0)*l1l1, &l1);
        }
        dgemm_(&TRANS_N, &TRANS_N, &l1l1, &l1, &ngridx,
               &D1, weight_x, &l1l1, xs_exp, &ngridx,
               &D0, out, &l1l1);
}

static void _affine_trans(double *out, double *int3d, double *a,
                          int floorl, int topl, double *cache)
{
        if (topl == 0) {
                out[0] = int3d[0];
                return;
        }

        int lx, ly, lz, l, m, n, i;
        int l1, l1l1, l1l1l1, lll;
        l1 = (topl+1)/2 + 1;
        l1l1l1 = l1 * l1 * l1;
        double *old = int3d;
        double *new = cache + l1l1l1 * l1l1l1;
        double *oldx, *oldy, *oldz, *newx, *tmp;
        double vx, vy, vz;

        if (floorl == 0) {
                out[0] = int3d[0];
                out += 1;
        }

        for (m = 1, l = topl; m <= topl; m++, l--) {
                l1 = l + 1;
                l1l1 = l1 * l1;
                lll = l * l * l;
                l1l1l1 = l1l1 * l1;
                newx = new;
                // attach x
                for (i = STARTX_IF_L_DEC1(m); i < _LEN_CART[m-1]; i++) {
                        oldx = old + i * l1l1l1 + l1l1;
                        oldy = old + i * l1l1l1 + l1;
                        oldz = old + i * l1l1l1 + 1;
                        for (n = 0, lx = 0; lx < l; lx++) {
                        for (ly = 0; ly < l; ly++) {
                        for (lz = 0; lz < l; lz++, n++) {
                                vx = oldx[lx*l1l1+ly*l1+lz];
                                vy = oldy[lx*l1l1+ly*l1+lz];
                                vz = oldz[lx*l1l1+ly*l1+lz];
                                newx[n] = vx * a[0] + vy * a[3] + vz * a[6];
                        } } }
                        newx += lll;
                }

                // attach y
                for (i = STARTY_IF_L_DEC1(m); i < _LEN_CART[m-1]; i++) {
                        oldx = old + i * l1l1l1 + l1l1;
                        oldy = old + i * l1l1l1 + l1;
                        oldz = old + i * l1l1l1 + 1;
                        for (n = 0, lx = 0; lx < l; lx++) {
                        for (ly = 0; ly < l; ly++) {
                        for (lz = 0; lz < l; lz++, n++) {
                                vx = oldx[lx*l1l1+ly*l1+lz];
                                vy = oldy[lx*l1l1+ly*l1+lz];
                                vz = oldz[lx*l1l1+ly*l1+lz];
                                newx[n] = vx * a[1] + vy * a[4] + vz * a[7];
                        } } }
                        newx += lll;
                }

                // attach z
                i = STARTZ_IF_L_DEC1(m);
                oldx = old + i * l1l1l1 + l1l1;
                oldy = old + i * l1l1l1 + l1;
                oldz = old + i * l1l1l1 + 1;
                for (n = 0, lx = 0; lx < l; lx++) {
                for (ly = 0; ly < l; ly++) {
                for (lz = 0; lz < l; lz++, n++) {
                        vx = oldx[lx*l1l1+ly*l1+lz];
                        vy = oldy[lx*l1l1+ly*l1+lz];
                        vz = oldz[lx*l1l1+ly*l1+lz];
                        newx[n] = vx * a[2] + vy * a[5] + vz * a[8];
                } } }

                if (floorl <= m) {
                        for (i = 0; i < _LEN_CART[m]; i++) {
                                out[i] = new[i * lll];
                        }
                        out += _LEN_CART[m];
                }

                if (m == 1) {
                        old = new;
                        new = cache;
                } else {
                        tmp = old;
                        old = new;
                        new = tmp;
                }
        }
}

static int _nonorth_loop(double *out, double fac, double log_prec,
                         int dimension, double *a, double *b,
                         int *mesh, double *weights,
                         CINTEnvVars *envs, double *cache)
{
        const int *shls  = envs->shls;
        const int *bas = envs->bas;
        double *env = envs->env;
        const int i_sh = shls[0];
        const int j_sh = shls[1];
        const int i_l = envs->i_l;
        const int j_l = envs->j_l;
        const int i_ctr = envs->x_ctr[0];
        const int j_ctr = envs->x_ctr[1];
        const int i_prim = bas(NPRIM_OF, i_sh);
        const int j_prim = bas(NPRIM_OF, j_sh);
        const int nf = envs->nf;
        double *ri = envs->ri;
        double *rj = envs->rj;
        double *ai = env + bas(PTR_EXP, i_sh);
        double *aj = env + bas(PTR_EXP, j_sh);
        double *ci = env + bas(PTR_COEFF, i_sh);
        double *cj = env + bas(PTR_COEFF, j_sh);
        double fac1i;
        double aij, eij;
        int ip, jp, n;
        int empty[3] = {1, 1, 1};
        int *jempty = empty + 0;
        int *iempty = empty + 1;
        //int *gempty = empty + 2;
        int l1 = i_l + j_l + 1;
        int l1l1l1 = l1 * l1 * l1;
        const size_t leni = l1l1l1 * i_ctr;
        const size_t lenj = l1l1l1 * i_ctr * j_ctr;
        double *gctrj = cache;
        double *gctri = gctrj + lenj;
        double *g = gctri + leni;
        double *log_iprim_max = g + l1l1l1;
        double *log_jprim_max = log_iprim_max + i_prim;
        cache += lenj + leni + l1l1l1 + i_prim + j_prim;

        for (ip = 0; ip < i_prim; ip++) {
                log_iprim_max[ip] = log(max_pgto_coeff(ci, i_prim, i_ctr, ip));
        }
        for (jp = 0; jp < j_prim; jp++) {
                log_jprim_max[jp] = log(max_pgto_coeff(cj, j_prim, j_ctr, jp));
        }
        void (*eval_3d)() = envs->f_gout;
        double rrij = CINTsquare_dist(ri, rj);
        double fac1 = fac * CINTcommon_fac_sp(i_l) * CINTcommon_fac_sp(j_l);
        double logcc;

        *jempty = 1;
        for (jp = 0; jp < j_prim; jp++) {
                *iempty = 1;
                for (ip = 0; ip < i_prim; ip++) {
                        aij = ai[ip] + aj[jp];
                        eij = (ai[ip] * aj[jp] / aij) * rrij;
                        logcc = log_iprim_max[ip] + log_jprim_max[jp];
                        if (eij-logcc > EXPCUTOFF15) { //(eij > EXPCUTOFF)?
                                continue;
                        }

                        fac1i = fac1 * exp(-eij);
                        (*eval_3d)(g, i_l, i_l+j_l, ai[ip], aj[jp],
                                   fac1i, logcc+log_prec, dimension,
                                   a, b, mesh, weights, envs, cache);
                        plain_prim_to_ctr(gctri, l1l1l1, g,
                                          i_prim, i_ctr, ci+ip, *iempty);
                        *iempty = 0;
                }
                if (!*iempty) {
                        plain_prim_to_ctr(gctrj, i_ctr*l1l1l1, gctri,
                                          j_prim, j_ctr, cj+jp, *jempty);
                        *jempty = 0;
                }
        }

        if (!*jempty) {
                double *buf = gctri;
                for (n = 0; n < i_ctr*j_ctr; n++) {
                        _affine_trans(buf, gctrj+n*l1l1l1, a, i_l, i_l+j_l, cache);
                        GTOplain_vrr2d(out+n*nf, buf, cache, envs);
                }
        }

        return !*jempty;
}

static int _nonorth_cache_size(int *mesh, CINTEnvVars *envs)
{
        const int i_ctr = envs->x_ctr[0];
        const int j_ctr = envs->x_ctr[1];
        const int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        const size_t nc = envs->nf * i_ctr * j_ctr;
        const int l = envs->i_l + envs->j_l;
        const int l1 = l + 1;
        const int nimgs = 1;
        size_t cache_size = 0;
        cache_size += l1 * l1 * l1 * (1 + i_ctr + i_ctr * j_ctr);
        cache_size += l1 * (mesh[0] + mesh[1] + mesh[2]) * nimgs;
        cache_size += mesh[1] * mesh[2]; // * nimgs * nimgs
        cache_size += l1 * mesh[2] * nimgs;
        cache_size += l1 * l1 * mesh[0];
        cache_size += 20; // i_prim + j_prim
        cache_size += 1000000;
        return nc * n_comp + MAX(cache_size, envs->nf * 8 * OF_CMPLX);
}

int NUMINT_nonorth_drv(double *out, int *dims, void (*f_c2s)(),
                       double fac, double log_prec, int dimension,
                       double *a, double *b, int *mesh, double *weights,
                       CINTEnvVars *envs, double *cache)
{
        if (out == NULL) {
                return _nonorth_cache_size(mesh, envs);
        }

        const int i_ctr = envs->x_ctr[0];
        const int j_ctr = envs->x_ctr[1];
        const int n_comp = envs->ncomp_e1 * envs->ncomp_tensor;
        const size_t nc = envs->nf * i_ctr * j_ctr;
        double *gctr = cache;
        cache += nc * n_comp;
        size_t n;
        int has_value = _nonorth_loop(gctr, fac, log_prec, dimension,
                                      a, b, mesh, weights, envs, cache);
        if (!has_value) {
                for (n = 0; n < nc*n_comp; n++) { gctr[n] = 0; }
        }

        int counts[4];
        if (f_c2s == &c2s_sph_1e) {
                counts[0] = (envs->i_l*2+1) * i_ctr;
                counts[1] = (envs->j_l*2+1) * j_ctr;
        } else { // f_c2s == &GTO_ft_c2s_cart
                counts[0] = envs->nfi * i_ctr;
                counts[1] = envs->nfj * j_ctr;
        }
        if (dims == NULL) {
                dims = counts;
        }
        size_t nout = dims[0] * dims[1];

        if (has_value) {
                for (n = 0; n < n_comp; n++) {
                        (*f_c2s)(out+nout*n, gctr+nc*n, dims, envs, cache);
                }
        }
        return has_value;
}

static int _max_cache_size(int (*intor)(), int (*numint_drv)(), int *shls_slice,
                           int *atm, int natm, int *bas, int nbas, double *env,
                           double *a, double *b, int *mesh, double *weights)
{
        int i, n;
        int i0 = MIN(shls_slice[0], shls_slice[2]);
        int i1 = MAX(shls_slice[1], shls_slice[3]);
        int shls[2];
        int cache_size = 0;
        for (i = i0; i < i1; i++) {
                shls[0] = i;
                shls[1] = i;
                n = (*intor)(NULL, NULL, shls, atm, natm, bas, nbas, env,
                             numint_drv, NULL, 0., 3, a, b, mesh, weights, NULL);
                cache_size = MAX(cache_size, n);
        }
        return cache_size;
}

void NUMINT_fill2c(int (*intor)(), int (*numint_drv)(), void (*eval_3d)(),
                   double *mat,
                   int comp, int hermi, int *shls_slice, int *ao_loc,
//?double complex *out, int nkpts, int comp, int nimgs,
//?double *Ls, double complex *expkL,
                   double log_prec, int dimension,
                   double *a, double *b, int *mesh, double *weights,
                   int *atm, int natm, int *bas, int nbas, double *env,
                   int nenv)
{
        const int ish0 = shls_slice[0];
        const int ish1 = shls_slice[1];
        const int jsh0 = shls_slice[2];
        const int jsh1 = shls_slice[3];
        const int nish = ish1 - ish0;
        const int njsh = jsh1 - jsh0;
        const size_t naoi = ao_loc[ish1] - ao_loc[ish0];
        const size_t naoj = ao_loc[jsh1] - ao_loc[jsh0];
        const int cache_size = _max_cache_size(intor, numint_drv, shls_slice,
                                               atm, natm, bas, nbas, env,
                                               a, b, mesh, weights);
#pragma omp parallel default(none) \
        shared(intor, numint_drv, eval_3d, mat, comp, hermi, ao_loc, \
               log_prec, dimension, a, b, mesh, weights, \
               atm, natm, bas, nbas, env)
{
        int dims[] = {naoi, naoj};
        int ish, jsh, ij, i0, j0;
        int shls[2];
        double *cache = malloc(sizeof(double) * cache_size);
#pragma omp for schedule(dynamic)
        for (ij = 0; ij < nish*njsh; ij++) {
                ish = ij / njsh;
                jsh = ij % njsh;
                if (hermi != PLAIN && ish > jsh) {
                        // fill up only upper triangle of F-array
                        continue;
                }

                ish += ish0;
                jsh += jsh0;
                shls[0] = ish;
                shls[1] = jsh;
                i0 = ao_loc[ish] - ao_loc[ish0];
                j0 = ao_loc[jsh] - ao_loc[jsh0];
                (*intor)(mat+j0*naoi+i0, dims, shls, atm, natm, bas, nbas, env,
                         numint_drv, eval_3d, log_prec, dimension,
                         a, b, mesh, weights, cache);
        }
        free(cache);
}
        if (hermi != PLAIN) { // lower triangle of F-array
                int ic;
                for (ic = 0; ic < comp; ic++) {
                        NPdsymm_triu(naoi, mat+ic*naoi*naoi, hermi);
                }
        }
}

int NUMINT_ovlp_cart(double *out, int *dims, int *shls, int *atm, int natm,
                     int *bas, int nbas, double *env,
                     int (*numint_drv)(), void (*eval_3d)(),
                     double log_prec, int dimension, double *a, double *b,
                     int *mesh, double *weights, double *cache)
{
        CINTEnvVars envs;
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        CINTinit_int1e_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = eval_3d;
        return (*numint_drv)(out, dims, &c2s_cart_1e, 1., log_prec,
                             dimension, a, b, mesh, weights, &envs, cache);
}

int NUMINT_ovlp_sph(double *out, int *dims, int *shls, int *atm, int natm,
                    int *bas, int nbas, double *env,
                    int (*numint_drv)(), void (*eval_3d)(),
                    double log_prec, int dimension, double *a, double *b,
                    int *mesh, double *weights, double *cache)
{
        CINTEnvVars envs;
        int ng[] = {0, 0, 0, 0, 0, 1, 0, 1};
        CINTinit_int1e_EnvVars(&envs, ng, shls, atm, natm, bas, nbas, env);
        envs.f_gout = eval_3d;
        return (*numint_drv)(out, dims, &c2s_sph_1e, 1., log_prec,
                             dimension, a, b, mesh, weights, &envs, cache);
}
