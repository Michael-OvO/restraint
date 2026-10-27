      SUBROUTINE INITFM
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER NX, NT
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /FLD/ PSI(2,4,4,4), CHI(2,4,4,4)

      DO 20 IZ = 1, 4
      DO 20 IY = 1, 4
      DO 20 IX = 1, 4
          PSI(1,IX,IY,IZ) = 1.0D0
          PSI(2,IX,IY,IZ) = 0.0D0
          CHI(1,IX,IY,IZ) = 0.0D0
          CHI(2,IX,IY,IZ) = 0.0D0
   20 CONTINUE

      RETURN
      END

      SUBROUTINE DSLASH(SRC,DST)
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER NX, NT
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /GLK/ U(2,2,4,4,4,4)
      DIMENSION SRC(2,4,4,4), DST(2,4,4,4)
      DIMENSION TMP(2)

      DO 10 IZ = 1, 4
      DO 10 IY = 1, 4
      DO 10 IX = 1, 4
          DST(1,IX,IY,IZ) = SRC(1,IX,IY,IZ)
          DST(2,IX,IY,IZ) = SRC(2,IX,IY,IZ)
   10 CONTINUE

      DO 100 IZ = 1, 4
      DO 100 IY = 1, 4
      DO 100 IX = 1, 4

          IXP = MOD(IX,4) + 1
          IXM = MOD(IX+2,4) + 1
          IYP = MOD(IY,4) + 1
          IYM = MOD(IY+2,4) + 1
          IZP = MOD(IZ,4) + 1
          IZM = MOD(IZ+2,4) + 1

          TMP(1) = U(1,1,1,IX,IY,IZ)*SRC(1,IXP,IY,IZ)
     &           + U(1,2,1,IX,IY,IZ)*SRC(2,IXP,IY,IZ)
          TMP(2) = U(2,1,1,IX,IY,IZ)*SRC(1,IXP,IY,IZ)
     &           + U(2,2,1,IX,IY,IZ)*SRC(2,IXP,IY,IZ)
          DST(1,IX,IY,IZ) = DST(1,IX,IY,IZ) - KAPPA*TMP(1)
          DST(2,IX,IY,IZ) = DST(2,IX,IY,IZ) - KAPPA*TMP(2)

          TMP(1) = U(1,1,1,IXM,IY,IZ)*SRC(1,IXM,IY,IZ)
     &           + U(2,1,1,IXM,IY,IZ)*SRC(2,IXM,IY,IZ)
          TMP(2) = U(1,2,1,IXM,IY,IZ)*SRC(1,IXM,IY,IZ)
     &           + U(2,2,1,IXM,IY,IZ)*SRC(2,IXM,IY,IZ)
          DST(1,IX,IY,IZ) = DST(1,IX,IY,IZ) - KAPPA*TMP(1)
          DST(2,IX,IY,IZ) = DST(2,IX,IY,IZ) - KAPPA*TMP(2)

          TMP(1) = U(1,1,2,IX,IY,IZ)*SRC(1,IX,IYP,IZ)
     &           + U(1,2,2,IX,IY,IZ)*SRC(2,IX,IYP,IZ)
          TMP(2) = U(2,1,2,IX,IY,IZ)*SRC(1,IX,IYP,IZ)
     &           + U(2,2,2,IX,IY,IZ)*SRC(2,IX,IYP,IZ)
          DST(1,IX,IY,IZ) = DST(1,IX,IY,IZ) - KAPPA*TMP(1)
          DST(2,IX,IY,IZ) = DST(2,IX,IY,IZ) - KAPPA*TMP(2)

          TMP(1) = U(1,1,2,IX,IYM,IZ)*SRC(1,IX,IYM,IZ)
     &           + U(2,1,2,IX,IYM,IZ)*SRC(2,IX,IYM,IZ)
          TMP(2) = U(1,2,2,IX,IYM,IZ)*SRC(1,IX,IYM,IZ)
     &           + U(2,2,2,IX,IYM,IZ)*SRC(2,IX,IYM,IZ)
          DST(1,IX,IY,IZ) = DST(1,IX,IY,IZ) - KAPPA*TMP(1)
          DST(2,IX,IY,IZ) = DST(2,IX,IY,IZ) - KAPPA*TMP(2)

          TMP(1) = U(1,1,3,IX,IY,IZ)*SRC(1,IX,IY,IZP)
     &           + U(1,2,3,IX,IY,IZ)*SRC(2,IX,IY,IZP)
          TMP(2) = U(2,1,3,IX,IY,IZ)*SRC(1,IX,IY,IZP)
     &           + U(2,2,3,IX,IY,IZ)*SRC(2,IX,IY,IZP)
          DST(1,IX,IY,IZ) = DST(1,IX,IY,IZ) - KAPPA*TMP(1)
          DST(2,IX,IY,IZ) = DST(2,IX,IY,IZ) - KAPPA*TMP(2)

          TMP(1) = U(1,1,3,IX,IY,IZM)*SRC(1,IX,IY,IZM)
     &           + U(2,1,3,IX,IY,IZM)*SRC(2,IX,IY,IZM)
          TMP(2) = U(1,2,3,IX,IY,IZM)*SRC(1,IX,IY,IZM)
     &           + U(2,2,3,IX,IY,IZM)*SRC(2,IX,IY,IZM)
          DST(1,IX,IY,IZ) = DST(1,IX,IY,IZ) - KAPPA*TMP(1)
          DST(2,IX,IY,IZ) = DST(2,IX,IY,IZ) - KAPPA*TMP(2)

  100 CONTINUE

      RETURN
      END

      SUBROUTINE FNORM(F,XNORM)
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      DIMENSION F(2,4,4,4)

      XNORM = 0.0D0
      DO 10 IZ = 1, 4
      DO 10 IY = 1, 4
      DO 10 IX = 1, 4
          XNORM = XNORM + F(1,IX,IY,IZ)**2 + F(2,IX,IY,IZ)**2
   10 CONTINUE
      XNORM = DSQRT(XNORM)

      RETURN
      END
