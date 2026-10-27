      SUBROUTINE HBSWEEP(IMODE)
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER NX, NT, IMODE, MU
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /GLK/ U(2,2,4,4,4,4)
      COMMON /ACC/ NACC, NREJ

      DO 100 IT = 1, 4
      DO 100 IZ = 1, 4
      DO 100 IY = 1, 4
      DO 100 IX = 1, 4
      DO 100 MU = 1, 4

          CALL STAPLE(MU,IX,IY,IZ,SR,SI)
          AK = BETA * SR / 2.0D0

          IF (AK - 0.1D0) 10, 20, 20
   10     CALL METSTEP(MU,IX,IY,IZ,AK,IACC)
          GO TO 50
   20     IF (AK - 50.0D0) 30, 30, 40
   30     CALL HBSTEP(MU,IX,IY,IZ,AK)
          IACC = 1
          GO TO 50
   40     CALL OVRSTEP(MU,IX,IY,IZ)
          IACC = 1
   50     CONTINUE

          IF (IMODE.EQ.1) THEN
              IF (IACC.EQ.1) THEN
                  NACC = NACC + 1
              ELSE
                  NREJ = NREJ + 1
              ENDIF
          ENDIF

  100 CONTINUE

      RETURN
      END

      SUBROUTINE HBSTEP(MU,IX,IY,IZ,AK)
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER NX, NT, MU, IX, IY, IZ
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /GLK/ U(2,2,4,4,4,4)
      DIMENSION UNEW(2,2), UOLD(2,2)

      PI = 3.141592653589793D0

      UOLD(1,1) = U(1,1,MU,IX,IY,IZ)
      UOLD(1,2) = U(1,2,MU,IX,IY,IZ)
      UOLD(2,1) = U(2,1,MU,IX,IY,IZ)
      UOLD(2,2) = U(2,2,MU,IX,IY,IZ)

      CALL URAND(R1)
      CALL URAND(R2)
      CALL URAND(R3)
      CALL URAND(R4)

      PHI = 2.0D0 * PI * R1
      COSTH = 2.0D0 * R2 - 1.0D0
      SINTH = DSQRT(1.0D0 - COSTH*COSTH)
      EPS = 0.3D0 * R3

      X1 = EPS * SINTH * DCOS(PHI)
      X2 = EPS * SINTH * DSIN(PHI)
      X3 = EPS * COSTH
      X0 = DSQRT(1.0D0 - EPS*EPS)

      UNEW(1,1) = X0*UOLD(1,1) - X3*UOLD(2,1)
      UNEW(1,2) = X0*UOLD(1,2) - X3*UOLD(2,2)
      UNEW(2,1) = X0*UOLD(2,1) + X3*UOLD(1,1)
      UNEW(2,2) = X0*UOLD(2,2) + X3*UOLD(1,2)

      IF (X1*X1 + X2*X2 - 1.0D-20) 10, 10, 20
   10 UNEW(1,1) = UNEW(1,1) + X1*UOLD(2,1)
      UNEW(1,2) = UNEW(1,2) + X1*UOLD(2,2)
      UNEW(2,1) = UNEW(2,1) - X1*UOLD(1,1)
      UNEW(2,2) = UNEW(2,2) - X1*UOLD(1,2)
      GO TO 30
   20 UNEW(1,1) = UNEW(1,1) + X2*UOLD(2,1)
      UNEW(1,2) = UNEW(1,2) + X2*UOLD(2,2)
      UNEW(2,1) = UNEW(2,1) - X2*UOLD(1,1)
      UNEW(2,2) = UNEW(2,2) - X2*UOLD(1,2)
   30 CONTINUE

      CALL PROJSU2(UNEW)

      SOLD = UOLD(1,1) + UOLD(2,2)
      SNEW = UNEW(1,1) + UNEW(2,2)
      DS = AK * (SNEW - SOLD)

      IF (DS) 40, 50, 50
   40 IF (DEXP(DS) - R4) 60, 50, 50
   50 U(1,1,MU,IX,IY,IZ) = UNEW(1,1)
      U(1,2,MU,IX,IY,IZ) = UNEW(1,2)
      U(2,1,MU,IX,IY,IZ) = UNEW(2,1)
      U(2,2,MU,IX,IY,IZ) = UNEW(2,2)
   60 CONTINUE

      RETURN
      END

      SUBROUTINE METSTEP(MU,IX,IY,IZ,AK,IACC)
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER NX, NT, MU, IX, IY, IZ, IACC
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /GLK/ U(2,2,4,4,4,4)
      DIMENSION UOLD(2,2), UNEW(2,2)

      IACC = 0

      UOLD(1,1) = U(1,1,MU,IX,IY,IZ)
      UOLD(1,2) = U(1,2,MU,IX,IY,IZ)
      UOLD(2,1) = U(2,1,MU,IX,IY,IZ)
      UOLD(2,2) = U(2,2,MU,IX,IY,IZ)

      CALL URAND(R1)
      CALL URAND(R2)
      CALL URAND(R3)
      CALL URAND(R4)

      EPS = 0.2D0
      D1 = EPS * (2.0D0*R1 - 1.0D0)
      D2 = EPS * (2.0D0*R2 - 1.0D0)
      D3 = EPS * (2.0D0*R3 - 1.0D0)

      DNORM = DSQRT(D1*D1 + D2*D2 + D3*D3)
      IF (DNORM - 1.0D-10) 10, 10, 20
   10 UNEW(1,1) = UOLD(1,1)
      UNEW(1,2) = UOLD(1,2)
      UNEW(2,1) = UOLD(2,1)
      UNEW(2,2) = UOLD(2,2)
      GO TO 30
   20 CONTINUE
      C0 = DSQRT(1.0D0 - DNORM*DNORM)
      UNEW(1,1) = C0*UOLD(1,1) + D3*UOLD(2,1)
      UNEW(1,2) = C0*UOLD(1,2) + D3*UOLD(2,2)
      UNEW(2,1) = C0*UOLD(2,1) - D3*UOLD(1,1)
      UNEW(2,2) = C0*UOLD(2,2) - D3*UOLD(1,2)
   30 CONTINUE

      CALL PROJSU2(UNEW)

      SOLD = UOLD(1,1) + UOLD(2,2)
      SNEW = UNEW(1,1) + UNEW(2,2)
      DS = AK * (SNEW - SOLD)

      IF (DS) 40, 50, 50
   40 IF (DEXP(DS) - R4) 60, 50, 50
   50 IACC = 1
      U(1,1,MU,IX,IY,IZ) = UNEW(1,1)
      U(1,2,MU,IX,IY,IZ) = UNEW(1,2)
      U(2,1,MU,IX,IY,IZ) = UNEW(2,1)
      U(2,2,MU,IX,IY,IZ) = UNEW(2,2)
   60 CONTINUE

      RETURN
      END

      SUBROUTINE OVRSTEP(MU,IX,IY,IZ)
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER NX, NT, MU, IX, IY, IZ
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /GLK/ U(2,2,4,4,4,4)
      DIMENSION UOLD(2,2), UNEW(2,2), STAP(2,2)

      CALL STAPLE(MU,IX,IY,IZ,SR,SI)

      UOLD(1,1) = U(1,1,MU,IX,IY,IZ)
      UOLD(1,2) = U(1,2,MU,IX,IY,IZ)
      UOLD(2,1) = U(2,1,MU,IX,IY,IZ)
      UOLD(2,2) = U(2,2,MU,IX,IY,IZ)

      SC = SR / 6.0D0
      IF (SC - 1.0D-10) 10, 10, 20
   10 RETURN
   20 CONTINUE

      STAP(1,1) = 1.0D0 / SC
      STAP(1,2) = 0.0D0
      STAP(2,1) = 0.0D0
      STAP(2,2) = 1.0D0 / SC

      UNEW(1,1) = STAP(1,1)*UOLD(1,1) + STAP(1,2)*UOLD(2,1)
      UNEW(1,2) = STAP(1,1)*UOLD(1,2) + STAP(1,2)*UOLD(2,2)
      UNEW(2,1) = STAP(2,1)*UOLD(1,1) + STAP(2,2)*UOLD(2,1)
      UNEW(2,2) = STAP(2,1)*UOLD(1,2) + STAP(2,2)*UOLD(2,2)

      CALL PROJSU2(UNEW)

      U(1,1,MU,IX,IY,IZ) = UNEW(1,1)
      U(1,2,MU,IX,IY,IZ) = UNEW(1,2)
      U(2,1,MU,IX,IY,IZ) = UNEW(2,1)
      U(2,2,MU,IX,IY,IZ) = UNEW(2,2)

      RETURN
      END
