      PROGRAM LQCD
      IMPLICIT DOUBLE PRECISION (A-H,K,O-Z)
      INTEGER ISEED, NT, NX, NTHERM, NMEAS, NCFG
      COMMON /CFG/ BETA, KAPPA, NX, NT, NCFG, NTHERM, NMEAS
      COMMON /RNG/ ISEED
      COMMON /GLK/ U(2,2,4,4,4,4)
      COMMON /FLD/ PSI(2,4,4,4), CHI(2,4,4,4)
      COMMON /OBS/ PLAQ, POLY, ENRG
      COMMON /ACC/ NACC, NREJ
      SAVE /GLK/, /FLD/, /OBS/, /ACC/

      OPEN(UNIT=10, FILE='/app/data/params.dat', STATUS='OLD')
      READ(10,*) BETA
      READ(10,*) KAPPA
      READ(10,*) NX
      READ(10,*) NT
      READ(10,*) NTHERM
      READ(10,*) NMEAS
      READ(10,*) ISEED
      CLOSE(10)

      NCFG = 0
      NACC = 0
      NREJ = 0
      CALL COLDST
      CALL INITFM

      DO 100 ICFG = 1, NTHERM + NMEAS
          IF (ICFG - NTHERM) 10, 10, 20
   10     CALL HBSWEEP(0)
          GO TO 100
   20     CALL HBSWEEP(1)
          NCFG = NCFG + 1
          CALL MEAS
  100 CONTINUE

      CALL OUTP

      STOP
      END
