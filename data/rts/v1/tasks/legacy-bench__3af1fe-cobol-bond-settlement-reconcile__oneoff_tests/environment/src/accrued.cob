       IDENTIFICATION DIVISION.
       PROGRAM-ID. ACCRUED.
      *================================================================*
      * ACCRUED - Calculate accrued interest for a bond trade          *
      * Reads coupon schedule and bond reference data                  *
      *================================================================*

       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT COUPON-FILE ASSIGN TO "/app/data/coupons.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-COUPON-STATUS.
           SELECT BOND-FILE ASSIGN TO "/app/data/bonds.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-BOND-STATUS.

       DATA DIVISION.
       FILE SECTION.
       FD  COUPON-FILE.
       COPY COUPON REPLACING ==:PREFIX:== BY ==FD-CPN==.

       FD  BOND-FILE.
       COPY BONDREC REPLACING ==:PREFIX:== BY ==FD-BND==.

       WORKING-STORAGE SECTION.
       01  WS-COUPON-STATUS        PIC XX.
       01  WS-BOND-STATUS          PIC XX.
       01  WS-EOF-COUPON           PIC 9 VALUE 0.
       01  WS-EOF-BOND             PIC 9 VALUE 0.
       01  WS-FOUND-COUPON         PIC 9 VALUE 0.
       01  WS-FOUND-BOND           PIC 9 VALUE 0.
       01  WS-LAST-COUPON-DATE     PIC 9(8) VALUE 0.
       01  WS-COUPON-RATE          PIC S9(3)V9(4) COMP-3.
       01  WS-PAR-VALUE            PIC S9(9)V99 COMP-3.
       01  WS-DAY-CONV             PIC X(1).
       01  WS-COUPON-FREQ          PIC 9(1).
       01  WS-DAYS-ACCRUED         PIC S9(5) COMP.
       01  WS-PERIOD-DAYS          PIC S9(5) COMP.
       01  WS-ACCRUED-AMT          PIC S9(9)V99 COMP-3.
       01  WS-TEMP-CALC            PIC S9(15)V9(6) COMP-3.

       COPY COUPON REPLACING ==:PREFIX:== BY ==WS-CPN==.
       COPY BONDREC REPLACING ==:PREFIX:== BY ==WS-BND==.

       LINKAGE SECTION.
       01  LS-CUSIP                PIC X(9).
       01  LS-SETTLE-DATE          PIC 9(8).
       01  LS-QUANTITY             PIC S9(7) COMP-3.
       01  LS-ACCR-INTEREST        PIC S9(9)V99 COMP-3.
       01  LS-DAYS-COUNTED         PIC S9(5) COMP.
       01  LS-DAY-CONV-OUT         PIC X(1).

       PROCEDURE DIVISION USING LS-CUSIP LS-SETTLE-DATE LS-QUANTITY
                                LS-ACCR-INTEREST LS-DAYS-COUNTED
                                LS-DAY-CONV-OUT.
       MAIN-LOGIC.
           MOVE 0 TO LS-ACCR-INTEREST.
           MOVE 0 TO LS-DAYS-COUNTED.
           MOVE SPACES TO LS-DAY-CONV-OUT.
           PERFORM LOOKUP-BOND-DATA.
           IF WS-FOUND-BOND = 1
               PERFORM FIND-LAST-COUPON
               IF WS-FOUND-COUPON = 1
                   PERFORM CALCULATE-ACCRUED
               END-IF
           END-IF.
           GOBACK.

       LOOKUP-BOND-DATA.
           MOVE 0 TO WS-FOUND-BOND.
           MOVE 0 TO WS-EOF-BOND.
           OPEN INPUT BOND-FILE.
           PERFORM UNTIL WS-EOF-BOND = 1 OR WS-FOUND-BOND = 1
               READ BOND-FILE INTO WS-BND-BOND-REC
                   AT END MOVE 1 TO WS-EOF-BOND
                   NOT AT END
                       IF WS-BND-CUSIP = LS-CUSIP
                           MOVE 1 TO WS-FOUND-BOND
                           MOVE WS-BND-COUPON-RATE TO WS-COUPON-RATE
                           MOVE WS-BND-PAR-VALUE TO WS-PAR-VALUE
                           MOVE WS-BND-DAY-COUNT-CONV TO WS-DAY-CONV
                           MOVE WS-BND-COUPON-FREQ TO WS-COUPON-FREQ
                       END-IF
               END-READ
           END-PERFORM.
           CLOSE BOND-FILE.
           MOVE WS-DAY-CONV TO LS-DAY-CONV-OUT.

       FIND-LAST-COUPON.
           MOVE 0 TO WS-FOUND-COUPON.
           MOVE 0 TO WS-EOF-COUPON.
           MOVE 0 TO WS-LAST-COUPON-DATE.
           OPEN INPUT COUPON-FILE.
           PERFORM UNTIL WS-EOF-COUPON = 1
               READ COUPON-FILE INTO WS-CPN-COUPON-REC
                   AT END MOVE 1 TO WS-EOF-COUPON
                   NOT AT END
                       IF WS-CPN-CPN-CUSIP = LS-CUSIP
                           IF WS-CPN-CPN-FREQ = 2
                               IF WS-CPN-CPN-DATE < LS-SETTLE-DATE
                                   IF WS-CPN-CPN-DATE >
                                      WS-LAST-COUPON-DATE
                                       MOVE WS-CPN-CPN-DATE TO
                                           WS-LAST-COUPON-DATE
                                       MOVE 1 TO WS-FOUND-COUPON
                                   END-IF
                               END-IF
                           END-IF
                       END-IF
               END-READ
           END-PERFORM.
           CLOSE COUPON-FILE.

       CALCULATE-ACCRUED.
           CALL "DAYCOUNT" USING WS-LAST-COUPON-DATE
                                 LS-SETTLE-DATE
                                 WS-DAY-CONV
                                 WS-DAYS-ACCRUED.
           MOVE WS-DAYS-ACCRUED TO LS-DAYS-COUNTED.
           EVALUATE WS-DAY-CONV
               WHEN 'A'
                   MOVE 365 TO WS-PERIOD-DAYS
               WHEN 'B'
                   MOVE 360 TO WS-PERIOD-DAYS
               WHEN 'C'
                   MOVE 360 TO WS-PERIOD-DAYS
               WHEN OTHER
                   MOVE 360 TO WS-PERIOD-DAYS
           END-EVALUATE.
           DIVIDE WS-PERIOD-DAYS BY WS-COUPON-FREQ
               GIVING WS-PERIOD-DAYS.
           COMPUTE WS-TEMP-CALC =
               (WS-COUPON-RATE / 100) * WS-PAR-VALUE *
               LS-QUANTITY * WS-DAYS-ACCRUED / WS-PERIOD-DAYS /
               WS-COUPON-FREQ.
           MOVE WS-TEMP-CALC TO LS-ACCR-INTEREST.
