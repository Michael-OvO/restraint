      SUBROUTINE COLDST
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER NX, NT
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /GLK/ U(2,2,4,4,4,4)

      DO 20 IT = 1, 4
      DO 20 IZ = 1, 4
      DO 20 IY = 1, 4
      DO 20 IX = 1, 4
      DO 20 MU = 1, 4
      DO 20 J = 1, 2
      DO 20 I = 1, 2
          U(I,J,MU,IX,IY,IZ) = 0.0D0
   20 CONTINUE

      DO 30 IT = 1, 4
      DO 30 IZ = 1, 4
      DO 30 IY = 1, 4
      DO 30 IX = 1, 4
      DO 30 MU = 1, 4
          U(1,1,MU,IX,IY,IZ) = 1.0D0
          U(2,2,MU,IX,IY,IZ) = 1.0D0
   30 CONTINUE

      RETURN
      END

      SUBROUTINE STAPLE(MU,IX,IY,IZ,SR,SI)
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER NX, NT, MU, IX, IY, IZ
      INTEGER IXP, IYP, IZP, IXM, IYM, IZM, NU
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /GLK/ U(2,2,4,4,4,4)
      DIMENSION W1(2,2), W2(2,2), W3(2,2), STAP(2,2)

      STAP(1,1) = 0.0D0
      STAP(1,2) = 0.0D0
      STAP(2,1) = 0.0D0
      STAP(2,2) = 0.0D0

      DO 100 NU = 1, 4
          IF (NU.EQ.MU) GO TO 100

          IXP = IX
          IYP = IY
          IZP = IZ
          IXM = IX
          IYM = IY
          IZM = IZ
          GO TO (11,12,13,14), MU
   11     IXP = MOD(IX,4) + 1
          IXM = MOD(IX+2,4) + 1
          GO TO 19
   12     IYP = MOD(IY,4) + 1
          IYM = MOD(IY+2,4) + 1
          GO TO 19
   13     IZP = MOD(IZ,4) + 1
          IZM = MOD(IZ+2,4) + 1
          GO TO 19
   14     CONTINUE
   19     CONTINUE

          IXNU = IX
          IYNU = IY
          IZNU = IZ
          GO TO (21,22,23,24), NU
   21     IXNU = MOD(IX,4) + 1
          GO TO 29
   22     IYNU = MOD(IY,4) + 1
          GO TO 29
   23     IZNU = MOD(IZ,4) + 1
          GO TO 29
   24     CONTINUE
   29     CONTINUE

          CALL MMULT(U(1,1,NU,IXP,IYP,IZP),
     &               U(1,1,MU,IXNU,IYNU,IZNU),W1)
          CALL MDAG(U(1,1,NU,IX,IY,IZ),W2)
          CALL MMULT(W1,W2,W3)
          STAP(1,1) = STAP(1,1) + W3(1,1)
          STAP(1,2) = STAP(1,2) + W3(1,2)
          STAP(2,1) = STAP(2,1) + W3(2,1)
          STAP(2,2) = STAP(2,2) + W3(2,2)

          IXNM = IX
          IYNM = IY
          IZNM = IZ
          GO TO (31,32,33,34), NU
   31     IXNM = MOD(IX+2,4) + 1
          GO TO 39
   32     IYNM = MOD(IY+2,4) + 1
          GO TO 39
   33     IZNM = MOD(IZ+2,4) + 1
          GO TO 39
   34     CONTINUE
   39     CONTINUE

          CALL MDAG(U(1,1,NU,IXP,IYP,IZP),W1)
          IXPM = IXP
          IYPM = IYP
          IZPM = IZP
          GO TO (41,42,43,44), NU
   41     IXPM = MOD(IXP+2,4) + 1
          GO TO 49
   42     IYPM = MOD(IYP+2,4) + 1
          GO TO 49
   43     IZPM = MOD(IZP+2,4) + 1
          GO TO 49
   44     CONTINUE
   49     CONTINUE

          CALL MDAG(U(1,1,MU,IXNM,IYNM,IZNM),W2)
          CALL MMULT(W1,W2,W3)
          CALL MMULT(W3,U(1,1,NU,IXNM,IYNM,IZNM),W1)
          STAP(1,1) = STAP(1,1) + W1(1,1)
          STAP(1,2) = STAP(1,2) + W1(1,2)
          STAP(2,1) = STAP(2,1) + W1(2,1)
          STAP(2,2) = STAP(2,2) + W1(2,2)

  100 CONTINUE

      SR = STAP(1,1) + STAP(2,2)
      SI = 0.0D0

      RETURN
      END

      SUBROUTINE MMULT(A,B,C)
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      DIMENSION A(2,2), B(2,2), C(2,2)

      C(1,1) = A(1,1)*B(1,1) + A(1,2)*B(2,1)
      C(1,2) = A(1,1)*B(1,2) + A(1,2)*B(2,2)
      C(2,1) = A(2,1)*B(1,1) + A(2,2)*B(2,1)
      C(2,2) = A(2,1)*B(1,2) + A(2,2)*B(2,2)

      RETURN
      END

      SUBROUTINE MDAG(A,B)
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      DIMENSION A(2,2), B(2,2)

      B(1,1) = A(1,1)
      B(1,2) = A(2,1)
      B(2,1) = A(1,2)
      B(2,2) = A(2,2)

      RETURN
      END

      SUBROUTINE PROJSU2(A)
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      DIMENSION A(2,2)

      TR = A(1,1) + A(2,2)
      DT = A(1,1)*A(2,2) - A(1,2)*A(2,1)

      IF (DT - 1.0D-12) 10, 20, 20
   10 A(1,1) = 1.0D0
      A(1,2) = 0.0D0
      A(2,1) = 0.0D0
      A(2,2) = 1.0D0
      RETURN
   20 CONTINUE

      SC = 1.0D0 / DSQRT(DT)
      A(1,1) = A(1,1) * SC
      A(1,2) = A(1,2) * SC
      A(2,1) = A(2,1) * SC
      A(2,2) = A(2,2) * SC

      RETURN
      END
