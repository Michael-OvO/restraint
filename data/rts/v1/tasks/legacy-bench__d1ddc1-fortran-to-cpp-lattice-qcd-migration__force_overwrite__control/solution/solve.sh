#!/bin/bash

mkdir -p /app/cpp

cat > /app/cpp/lqcd.cpp << 'CPP'
#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cstdint>

class RNG {
    int32_t iseed;
public:
    RNG(int s) : iseed(s) {}
    double urand() {
        // Fortran uses 32-bit INTEGER multiplication which overflows
        // Compute in 64-bit, then truncate to 32-bit to simulate overflow
        int64_t prod64 = static_cast<int64_t>(16807) * iseed;
        int32_t product = static_cast<int32_t>(prod64);  // truncate to simulate overflow
        iseed = product % 2147483647;
        if (iseed < 0) iseed += 2147483647;
        return static_cast<double>(iseed) / 2147483647.0;
    }
    int state() const { return iseed; }
};

double U[2][2][4][4][4][4];
double psi[2][4][4][4], chi[2][4][4][4];
double PLAQ, POLY, ENRG;
double BETA, KAPPA;
int NX, NT, NCFG, NTHERM, NMEAS, NACC, NREJ;

void mmult(double a[2][2], double b[2][2], double c[2][2]) {
    c[0][0] = a[0][0]*b[0][0] + a[0][1]*b[1][0];
    c[0][1] = a[0][0]*b[0][1] + a[0][1]*b[1][1];
    c[1][0] = a[1][0]*b[0][0] + a[1][1]*b[1][0];
    c[1][1] = a[1][0]*b[0][1] + a[1][1]*b[1][1];
}

void mdag(double a[2][2], double b[2][2]) {
    b[0][0] = a[0][0];
    b[0][1] = a[1][0];
    b[1][0] = a[0][1];
    b[1][1] = a[1][1];
}

void projsu2(double a[2][2]) {
    double dt = a[0][0]*a[1][1] - a[0][1]*a[1][0];
    if (dt < 1.0e-12) {
        a[0][0] = 1.0; a[0][1] = 0.0;
        a[1][0] = 0.0; a[1][1] = 1.0;
        return;
    }
    double sc = 1.0 / std::sqrt(dt);
    a[0][0] *= sc; a[0][1] *= sc;
    a[1][0] *= sc; a[1][1] *= sc;
}

void coldst() {
    for (int iz = 0; iz < 4; ++iz)
    for (int iy = 0; iy < 4; ++iy)
    for (int ix = 0; ix < 4; ++ix)
    for (int mu = 0; mu < 4; ++mu)
    for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i)
        U[i][j][mu][ix][iy][iz] = 0.0;
    for (int iz = 0; iz < 4; ++iz)
    for (int iy = 0; iy < 4; ++iy)
    for (int ix = 0; ix < 4; ++ix)
    for (int mu = 0; mu < 4; ++mu) {
        U[0][0][mu][ix][iy][iz] = 1.0;
        U[1][1][mu][ix][iy][iz] = 1.0;
    }
}

void initfm() {
    for (int iz = 0; iz < 4; ++iz)
    for (int iy = 0; iy < 4; ++iy)
    for (int ix = 0; ix < 4; ++ix) {
        psi[0][ix][iy][iz] = 1.0;
        psi[1][ix][iy][iz] = 0.0;
        chi[0][ix][iy][iz] = 0.0;
        chi[1][ix][iy][iz] = 0.0;
    }
}

void staple(int mu, int ix, int iy, int iz, double &sr, double &si) {
    double stap[2][2] = {{0,0},{0,0}};
    double w1[2][2], w2[2][2], w3[2][2];
    double ua[2][2], ub[2][2], uc[2][2];

    for (int nu = 0; nu < 4; ++nu) {
        if (nu == mu) continue;

        // Compute x+mu position
        int ixp = ix, iyp = iy, izp = iz;
        switch(mu) {
            case 0: ixp = (ix + 1) % 4; break;
            case 1: iyp = (iy + 1) % 4; break;
            case 2: izp = (iz + 1) % 4; break;
            // case 3: temporal direction - no spatial shift
        }

        // Compute x+nu position
        int ixnu = ix, iynu = iy, iznu = iz;
        switch(nu) {
            case 0: ixnu = (ix + 1) % 4; break;
            case 1: iynu = (iy + 1) % 4; break;
            case 2: iznu = (iz + 1) % 4; break;
            // case 3: temporal direction - no spatial shift
        }

        // Positive direction staple: U(nu,x+mu) * U(mu,x+nu) * U^dag(nu,x)
        // (matches Fortran exactly)
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][nu][ixp][iyp][izp];
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ub[i][j] = U[i][j][mu][ixnu][iynu][iznu];
        mmult(ua, ub, w1);
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) uc[i][j] = U[i][j][nu][ix][iy][iz];
        mdag(uc, w2);
        mmult(w1, w2, w3);
        stap[0][0] += w3[0][0]; stap[0][1] += w3[0][1];
        stap[1][0] += w3[1][0]; stap[1][1] += w3[1][1];

        // Compute x-nu position
        int ixnm = ix, iynm = iy, iznm = iz;
        switch(nu) {
            case 0: ixnm = (ix + 3) % 4; break;
            case 1: iynm = (iy + 3) % 4; break;
            case 2: iznm = (iz + 3) % 4; break;
            // case 3: temporal direction - no spatial shift
        }

        // Compute (x+mu)-nu position
        int ixpm = ixp, iypm = iyp, izpm = izp;
        switch(nu) {
            case 0: ixpm = (ixp + 3) % 4; break;
            case 1: iypm = (iyp + 3) % 4; break;
            case 2: izpm = (izp + 3) % 4; break;
            // case 3: temporal direction - no spatial shift
        }

        // Negative direction staple: U^dag(nu,x+mu) * U^dag(mu,x-nu) * U(nu,x-nu)
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][nu][ixp][iyp][izp];
        mdag(ua, w1);
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ub[i][j] = U[i][j][mu][ixnm][iynm][iznm];
        mdag(ub, w2);
        mmult(w1, w2, w3);
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) uc[i][j] = U[i][j][nu][ixnm][iynm][iznm];
        mmult(w3, uc, w1);
        stap[0][0] += w1[0][0]; stap[0][1] += w1[0][1];
        stap[1][0] += w1[1][0]; stap[1][1] += w1[1][1];
    }

    sr = stap[0][0] + stap[1][1];
    si = 0.0;
}

void hbstep(int mu, int ix, int iy, int iz, double ak, RNG &rng) {
    double uold[2][2], unew[2][2];
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) uold[i][j] = U[i][j][mu][ix][iy][iz];

    double r1 = rng.urand();
    double r2 = rng.urand();
    double r3 = rng.urand();
    double r4 = rng.urand();

    double pi = 3.141592653589793;
    double phi = 2.0 * pi * r1;
    double costh = 2.0 * r2 - 1.0;
    double sinth = std::sqrt(1.0 - costh*costh);
    double eps = 0.3 * r3;

    double x1 = eps * sinth * std::cos(phi);
    double x2 = eps * sinth * std::sin(phi);
    double x3 = eps * costh;
    double x0 = std::sqrt(1.0 - eps*eps);

    unew[0][0] = x0*uold[0][0] - x3*uold[1][0];
    unew[0][1] = x0*uold[0][1] - x3*uold[1][1];
    unew[1][0] = x0*uold[1][0] + x3*uold[0][0];
    unew[1][1] = x0*uold[1][1] + x3*uold[0][1];

    if (x1*x1 + x2*x2 < 1.0e-20) {
        unew[0][0] += x1*uold[1][0];
        unew[0][1] += x1*uold[1][1];
        unew[1][0] -= x1*uold[0][0];
        unew[1][1] -= x1*uold[0][1];
    } else {
        unew[0][0] += x2*uold[1][0];
        unew[0][1] += x2*uold[1][1];
        unew[1][0] -= x2*uold[0][0];
        unew[1][1] -= x2*uold[0][1];
    }

    projsu2(unew);

    double sold = uold[0][0] + uold[1][1];
    double snew = unew[0][0] + unew[1][1];
    double ds = ak * (snew - sold);

    if (ds >= 0.0 || std::exp(ds) > r4) {
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) U[i][j][mu][ix][iy][iz] = unew[i][j];
    }
}

void metstep(int mu, int ix, int iy, int iz, double ak, RNG &rng, int &iacc) {
    iacc = 0;
    double uold[2][2], unew[2][2];
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) uold[i][j] = U[i][j][mu][ix][iy][iz];

    double r1 = rng.urand();
    double r2 = rng.urand();
    double r3 = rng.urand();
    double r4 = rng.urand();

    double eps = 0.2;
    double d1 = eps * (2.0*r1 - 1.0);
    double d2 = eps * (2.0*r2 - 1.0);
    double d3 = eps * (2.0*r3 - 1.0);
    double dnorm = std::sqrt(d1*d1 + d2*d2 + d3*d3);

    if (dnorm < 1.0e-10) {
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) unew[i][j] = uold[i][j];
    } else {
        double c0 = std::sqrt(1.0 - dnorm*dnorm);
        unew[0][0] = c0*uold[0][0] + d3*uold[1][0];
        unew[0][1] = c0*uold[0][1] + d3*uold[1][1];
        unew[1][0] = c0*uold[1][0] - d3*uold[0][0];
        unew[1][1] = c0*uold[1][1] - d3*uold[0][1];
    }

    projsu2(unew);

    double sold = uold[0][0] + uold[1][1];
    double snew = unew[0][0] + unew[1][1];
    double ds = ak * (snew - sold);

    if (ds >= 0.0 || std::exp(ds) > r4) {
        iacc = 1;
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) U[i][j][mu][ix][iy][iz] = unew[i][j];
    }
}

void ovrstep(int mu, int ix, int iy, int iz) {
    double sr, si;
    staple(mu, ix, iy, iz, sr, si);

    double uold[2][2], unew[2][2];
    for(int i=0;i<2;i++) for(int j=0;j<2;j++) uold[i][j] = U[i][j][mu][ix][iy][iz];

    double sc = sr / 6.0;
    if (sc < 1.0e-10) return;

    unew[0][0] = (1.0/sc)*uold[0][0];
    unew[0][1] = (1.0/sc)*uold[0][1];
    unew[1][0] = (1.0/sc)*uold[1][0];
    unew[1][1] = (1.0/sc)*uold[1][1];

    projsu2(unew);

    for(int i=0;i<2;i++) for(int j=0;j<2;j++) U[i][j][mu][ix][iy][iz] = unew[i][j];
}

void hbsweep(int imode, RNG &rng) {
    for (int it = 0; it < 4; ++it)
    for (int iz = 0; iz < 4; ++iz)
    for (int iy = 0; iy < 4; ++iy)
    for (int ix = 0; ix < 4; ++ix)
    for (int mu = 0; mu < 4; ++mu) {
        double sr, si;
        staple(mu, ix, iy, iz, sr, si);
        double ak = BETA * sr / 2.0;
        int iacc = 1;
        if (ak < 0.1) {
            metstep(mu, ix, iy, iz, ak, rng, iacc);
        } else if (ak < 50.0) {
            hbstep(mu, ix, iy, iz, ak, rng);
            iacc = 1;
        } else {
            ovrstep(mu, ix, iy, iz);
            iacc = 1;
        }
        if (imode == 1) {
            if (iacc == 1) NACC++;
            else NREJ++;
        }
    }
}

void dslash() {
    for (int iz = 0; iz < 4; ++iz)
    for (int iy = 0; iy < 4; ++iy)
    for (int ix = 0; ix < 4; ++ix) {
        chi[0][ix][iy][iz] = psi[0][ix][iy][iz];
        chi[1][ix][iy][iz] = psi[1][ix][iy][iz];
    }

    for (int iz = 0; iz < 4; ++iz)
    for (int iy = 0; iy < 4; ++iy)
    for (int ix = 0; ix < 4; ++ix) {
        int ixp = (ix + 1) % 4, ixm = (ix + 3) % 4;
        int iyp = (iy + 1) % 4, iym = (iy + 3) % 4;
        int izp = (iz + 1) % 4, izm = (iz + 3) % 4;

        double tmp0, tmp1;

        tmp0 = U[0][0][0][ix][iy][iz]*psi[0][ixp][iy][iz] + U[0][1][0][ix][iy][iz]*psi[1][ixp][iy][iz];
        tmp1 = U[1][0][0][ix][iy][iz]*psi[0][ixp][iy][iz] + U[1][1][0][ix][iy][iz]*psi[1][ixp][iy][iz];
        chi[0][ix][iy][iz] -= KAPPA*tmp0;
        chi[1][ix][iy][iz] -= KAPPA*tmp1;

        tmp0 = U[0][0][0][ixm][iy][iz]*psi[0][ixm][iy][iz] + U[1][0][0][ixm][iy][iz]*psi[1][ixm][iy][iz];
        tmp1 = U[0][1][0][ixm][iy][iz]*psi[0][ixm][iy][iz] + U[1][1][0][ixm][iy][iz]*psi[1][ixm][iy][iz];
        chi[0][ix][iy][iz] -= KAPPA*tmp0;
        chi[1][ix][iy][iz] -= KAPPA*tmp1;

        tmp0 = U[0][0][1][ix][iy][iz]*psi[0][ix][iyp][iz] + U[0][1][1][ix][iy][iz]*psi[1][ix][iyp][iz];
        tmp1 = U[1][0][1][ix][iy][iz]*psi[0][ix][iyp][iz] + U[1][1][1][ix][iy][iz]*psi[1][ix][iyp][iz];
        chi[0][ix][iy][iz] -= KAPPA*tmp0;
        chi[1][ix][iy][iz] -= KAPPA*tmp1;

        tmp0 = U[0][0][1][ix][iym][iz]*psi[0][ix][iym][iz] + U[1][0][1][ix][iym][iz]*psi[1][ix][iym][iz];
        tmp1 = U[0][1][1][ix][iym][iz]*psi[0][ix][iym][iz] + U[1][1][1][ix][iym][iz]*psi[1][ix][iym][iz];
        chi[0][ix][iy][iz] -= KAPPA*tmp0;
        chi[1][ix][iy][iz] -= KAPPA*tmp1;

        tmp0 = U[0][0][2][ix][iy][iz]*psi[0][ix][iy][izp] + U[0][1][2][ix][iy][iz]*psi[1][ix][iy][izp];
        tmp1 = U[1][0][2][ix][iy][iz]*psi[0][ix][iy][izp] + U[1][1][2][ix][iy][iz]*psi[1][ix][iy][izp];
        chi[0][ix][iy][iz] -= KAPPA*tmp0;
        chi[1][ix][iy][iz] -= KAPPA*tmp1;

        tmp0 = U[0][0][2][ix][iy][izm]*psi[0][ix][iy][izm] + U[1][0][2][ix][iy][izm]*psi[1][ix][iy][izm];
        tmp1 = U[0][1][2][ix][iy][izm]*psi[0][ix][iy][izm] + U[1][1][2][ix][iy][izm]*psi[1][ix][iy][izm];
        chi[0][ix][iy][iz] -= KAPPA*tmp0;
        chi[1][ix][iy][iz] -= KAPPA*tmp1;
    }
}

double fnorm() {
    double xnorm = 0.0;
    for (int iz = 0; iz < 4; ++iz)
    for (int iy = 0; iy < 4; ++iy)
    for (int ix = 0; ix < 4; ++ix)
        xnorm += chi[0][ix][iy][iz]*chi[0][ix][iy][iz] + chi[1][ix][iy][iz]*chi[1][ix][iy][iz];
    return std::sqrt(xnorm);
}

void meas() {
    double w1[2][2], w2[2][2], w3[2][2];
    double ua[2][2], ub[2][2];

    PLAQ = 0.0;
    for (int iz = 0; iz < 4; ++iz)
    for (int iy = 0; iy < 4; ++iy)
    for (int ix = 0; ix < 4; ++ix) {
        int ixp = (ix + 1) % 4, iyp = (iy + 1) % 4, izp = (iz + 1) % 4;

        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][0][ix][iy][iz];
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ub[i][j] = U[i][j][1][ixp][iy][iz];
        mmult(ua, ub, w1);
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][0][ix][iyp][iz];
        mdag(ua, w2);
        mmult(w1, w2, w3);
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][1][ix][iy][iz];
        mdag(ua, w1);
        mmult(w3, w1, w2);
        PLAQ += w2[0][0] + w2[1][1];

        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][0][ix][iy][iz];
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ub[i][j] = U[i][j][2][ixp][iy][iz];
        mmult(ua, ub, w1);
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][0][ix][iy][izp];
        mdag(ua, w2);
        mmult(w1, w2, w3);
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][2][ix][iy][iz];
        mdag(ua, w1);
        mmult(w3, w1, w2);
        PLAQ += w2[0][0] + w2[1][1];

        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][1][ix][iy][iz];
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ub[i][j] = U[i][j][2][ix][iyp][iz];
        mmult(ua, ub, w1);
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][1][ix][iy][izp];
        mdag(ua, w2);
        mmult(w1, w2, w3);
        for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][2][ix][iy][iz];
        mdag(ua, w1);
        mmult(w3, w1, w2);
        PLAQ += w2[0][0] + w2[1][1];
    }
    int nplaq = 3 * 64;
    PLAQ = PLAQ / static_cast<double>(2 * nplaq);

    POLY = 0.0;
    for (int iz = 0; iz < 4; ++iz)
    for (int iy = 0; iy < 4; ++iy)
    for (int ix = 0; ix < 4; ++ix) {
        w1[0][0] = 1.0; w1[0][1] = 0.0;
        w1[1][0] = 0.0; w1[1][1] = 1.0;
        for (int it = 0; it < 4; ++it) {
            for(int i=0;i<2;i++) for(int j=0;j<2;j++) ua[i][j] = U[i][j][3][ix][iy][iz];
            mmult(w1, ua, w2);
            for(int i=0;i<2;i++) for(int j=0;j<2;j++) w1[i][j] = w2[i][j];
        }
        POLY += w1[0][0] + w1[1][1];
    }
    POLY = POLY / static_cast<double>(2 * 64);

    dslash();
    ENRG = fnorm() / static_cast<double>(64);
}

struct LatticeSimulator {
    void run() {}
};

int main() {
    std::ifstream params("/app/data/params.dat");
    int seed;
    params >> BETA;
    params >> KAPPA;
    params >> NX;
    params >> NT;
    params >> NTHERM;
    params >> NMEAS;
    params >> seed;
    params.close();

    NCFG = 0;
    NACC = 0;
    NREJ = 0;
    RNG rng(seed);

    coldst();
    initfm();

    for (int icfg = 1; icfg <= NTHERM + NMEAS; ++icfg) {
        if (icfg <= NTHERM) {
            hbsweep(0, rng);
        } else {
            hbsweep(1, rng);
            NCFG++;
            meas();
        }
    }

    double accr = 0.0;
    int ntot = NACC + NREJ;
    if (ntot > 0) accr = static_cast<double>(NACC) / static_cast<double>(ntot);

    if (NCFG < 1) {
        PLAQ = 0.0;
        POLY = 0.0;
        ENRG = 0.0;
    }

    std::ofstream out("/app/output.dat");
    out << "NCFG=" << std::setw(8) << NCFG << "\n";
    out << std::scientific << std::setprecision(12);
    out << "PLAQ=" << std::setw(20) << PLAQ << "\n";
    out << "POLY=" << std::setw(20) << POLY << "\n";
    out << "ENRG=" << std::setw(20) << ENRG << "\n";
    out << "ACCR=" << std::setw(20) << accr << "\n";
    out << "SEED=" << std::setw(12) << rng.state() << "\n";
    out.close();

    return 0;
}
CPP

g++ -std=c++17 -O2 -o /app/cpp/lqcd /app/cpp/lqcd.cpp
/app/cpp/lqcd
