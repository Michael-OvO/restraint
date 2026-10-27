      SUBROUTINE MEAS
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER NX, NT
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /GLK/ U(2,2,4,4,4,4)
      COMMON /FLD/ PSI(2,4,4,4), CHI(2,4,4,4)
      COMMON /OBS/ PLAQ, POLY, ENRG
      DIMENSION W1(2,2), W2(2,2), W3(2,2)

      PLAQ = 0.0D0
      DO 100 IZ = 1, 4
      DO 100 IY = 1, 4
      DO 100 IX = 1, 4

          CALL MMULT(U(1,1,1,IX,IY,IZ),U(1,1,2,MOD(IX,4)+1,IY,IZ),W1)
          CALL MDAG(U(1,1,1,IX,MOD(IY,4)+1,IZ),W2)
          CALL MMULT(W1,W2,W3)
          CALL MDAG(U(1,1,2,IX,IY,IZ),W1)
          CALL MMULT(W3,W1,W2)
          PLAQ = PLAQ + W2(1,1) + W2(2,2)

          CALL MMULT(U(1,1,1,IX,IY,IZ),U(1,1,3,MOD(IX,4)+1,IY,IZ),W1)
          CALL MDAG(U(1,1,1,IX,IY,MOD(IZ,4)+1),W2)
          CALL MMULT(W1,W2,W3)
          CALL MDAG(U(1,1,3,IX,IY,IZ),W1)
          CALL MMULT(W3,W1,W2)
          PLAQ = PLAQ + W2(1,1) + W2(2,2)

          CALL MMULT(U(1,1,2,IX,IY,IZ),U(1,1,3,IX,MOD(IY,4)+1,IZ),W1)
          CALL MDAG(U(1,1,2,IX,IY,MOD(IZ,4)+1),W2)
          CALL MMULT(W1,W2,W3)
          CALL MDAG(U(1,1,3,IX,IY,IZ),W1)
          CALL MMULT(W3,W1,W2)
          PLAQ = PLAQ + W2(1,1) + W2(2,2)

  100 CONTINUE

      NPLAQ = 3 * 64
      PLAQ = PLAQ / DBLE(2 * NPLAQ)

      POLY = 0.0D0
      DO 200 IZ = 1, 4
      DO 200 IY = 1, 4
      DO 200 IX = 1, 4
          W1(1,1) = 1.0D0
          W1(1,2) = 0.0D0
          W1(2,1) = 0.0D0
          W1(2,2) = 1.0D0

          DO 150 IT = 1, 4
              CALL MMULT(W1,U(1,1,4,IX,IY,IZ),W2)
              W1(1,1) = W2(1,1)
              W1(1,2) = W2(1,2)
              W1(2,1) = W2(2,1)
              W1(2,2) = W2(2,2)
  150     CONTINUE

          POLY = POLY + W1(1,1) + W1(2,2)
  200 CONTINUE
      POLY = POLY / DBLE(2 * 64)

      CALL DSLASH(PSI,CHI)
      CALL FNORM(CHI,ENRG)
      ENRG = ENRG / DBLE(64)

      RETURN
      END

      SUBROUTINE OUTP
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER NX, NT
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /OBS/ PLAQ, POLY, ENRG
      COMMON /ACC/ NACC, NREJ
      COMMON /RNG/ ISEED

      IF (NCFG - 1) 10, 20, 20
   10 PLAQ = 0.0D0
      POLY = 0.0D0
      ENRG = 0.0D0
      GO TO 30
   20 CONTINUE
   30 CONTINUE

      NTOT = NACC + NREJ
      IF (NTOT.GT.0) THEN
          ACCR = DBLE(NACC) / DBLE(NTOT)
      ELSE
          ACCR = 0.0D0
      ENDIF

      OPEN(UNIT=20, FILE='/app/output.dat', STATUS='UNKNOWN')
      WRITE(20,200) NCFG
      WRITE(20,210) PLAQ
      WRITE(20,220) POLY
      WRITE(20,230) ENRG
      WRITE(20,240) ACCR
      WRITE(20,250) ISEED
  200 FORMAT('NCFG=',I8)
  210 FORMAT('PLAQ=',E20.12)
  220 FORMAT('POLY=',E20.12)
  230 FORMAT('ENRG=',E20.12)
  240 FORMAT('ACCR=',E20.12)
  250 FORMAT('SEED=',I12)
      CLOSE(20)

      RETURN
      END
