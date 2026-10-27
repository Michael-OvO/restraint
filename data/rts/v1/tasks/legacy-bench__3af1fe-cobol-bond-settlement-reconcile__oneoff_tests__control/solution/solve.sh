#!/bin/bash
cat > /app/daycount.cob << 'ENDCOBOL'
       IDENTIFICATION DIVISION.
       PROGRAM-ID. DAYCOUNT.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-START-YEAR          PIC 9(4).
       01  WS-START-MONTH         PIC 9(2).
       01  WS-START-DAY           PIC 9(2).
       01  WS-END-YEAR            PIC 9(4).
       01  WS-END-MONTH           PIC 9(2).
       01  WS-END-DAY             PIC 9(2).
       01  WS-TEMP-DAYS           PIC S9(5) COMP.
       01  WS-ACTUAL-DAYS         PIC S9(5) COMP.
       01  WS-MONTH-IDX           PIC 9(2).
       01  WS-DAYS-IN-MONTH       PIC 9(2).
       01  WS-IS-LEAP             PIC 9(1).
       01  WS-WORK-YEAR           PIC 9(4).

       LINKAGE SECTION.
       01  LS-START-DATE          PIC 9(8).
       01  LS-END-DATE            PIC 9(8).
       01  LS-DAY-CONV            PIC X(1).
       01  LS-DAYS-RESULT         PIC S9(5) COMP.

       PROCEDURE DIVISION USING LS-START-DATE LS-END-DATE
                                 LS-DAY-CONV LS-DAYS-RESULT.
       MAIN-LOGIC.
           PERFORM PARSE-DATES.
           EVALUATE LS-DAY-CONV
               WHEN 'A'
                   PERFORM CALC-ACTUAL-DAYS
                   MOVE WS-ACTUAL-DAYS TO LS-DAYS-RESULT
               WHEN 'B'
                   PERFORM CALC-30-360-DAYS
                   MOVE WS-TEMP-DAYS TO LS-DAYS-RESULT
               WHEN 'C'
                   PERFORM CALC-ACTUAL-DAYS
                   MOVE WS-ACTUAL-DAYS TO LS-DAYS-RESULT
               WHEN OTHER
                   PERFORM CALC-30-360-DAYS
                   MOVE WS-TEMP-DAYS TO LS-DAYS-RESULT
           END-EVALUATE.
           GOBACK.

       PARSE-DATES.
           MOVE LS-START-DATE(1:4) TO WS-START-YEAR.
           MOVE LS-START-DATE(5:2) TO WS-START-MONTH.
           MOVE LS-START-DATE(7:2) TO WS-START-DAY.
           MOVE LS-END-DATE(1:4) TO WS-END-YEAR.
           MOVE LS-END-DATE(5:2) TO WS-END-MONTH.
           MOVE LS-END-DATE(7:2) TO WS-END-DAY.

       CALC-30-360-DAYS.
           IF WS-START-DAY = 31
               MOVE 30 TO WS-START-DAY
           END-IF.
           IF WS-END-DAY = 31 AND WS-START-DAY >= 30
               MOVE 30 TO WS-END-DAY
           END-IF.
           COMPUTE WS-TEMP-DAYS =
               (WS-END-YEAR - WS-START-YEAR) * 360
               + (WS-END-MONTH - WS-START-MONTH) * 30
               + (WS-END-DAY - WS-START-DAY).

       CALC-ACTUAL-DAYS.
           MOVE 0 TO WS-ACTUAL-DAYS.
           MOVE WS-START-YEAR TO WS-WORK-YEAR.
           MOVE WS-START-MONTH TO WS-MONTH-IDX.
           PERFORM UNTIL WS-WORK-YEAR > WS-END-YEAR OR
               (WS-WORK-YEAR = WS-END-YEAR AND
                WS-MONTH-IDX >= WS-END-MONTH)
               PERFORM GET-DAYS-IN-MONTH
               ADD WS-DAYS-IN-MONTH TO WS-ACTUAL-DAYS
               ADD 1 TO WS-MONTH-IDX
               IF WS-MONTH-IDX > 12
                   MOVE 1 TO WS-MONTH-IDX
                   ADD 1 TO WS-WORK-YEAR
               END-IF
           END-PERFORM.
           SUBTRACT WS-START-DAY FROM WS-ACTUAL-DAYS.
           ADD WS-END-DAY TO WS-ACTUAL-DAYS.

       GET-DAYS-IN-MONTH.
           EVALUATE WS-MONTH-IDX
               WHEN 1 MOVE 31 TO WS-DAYS-IN-MONTH
               WHEN 2
                   PERFORM CHECK-LEAP-YEAR
                   IF WS-IS-LEAP = 1
                       MOVE 29 TO WS-DAYS-IN-MONTH
                   ELSE
                       MOVE 28 TO WS-DAYS-IN-MONTH
                   END-IF
               WHEN 3 MOVE 31 TO WS-DAYS-IN-MONTH
               WHEN 4 MOVE 30 TO WS-DAYS-IN-MONTH
               WHEN 5 MOVE 31 TO WS-DAYS-IN-MONTH
               WHEN 6 MOVE 30 TO WS-DAYS-IN-MONTH
               WHEN 7 MOVE 31 TO WS-DAYS-IN-MONTH
               WHEN 8 MOVE 31 TO WS-DAYS-IN-MONTH
               WHEN 9 MOVE 30 TO WS-DAYS-IN-MONTH
               WHEN 10 MOVE 31 TO WS-DAYS-IN-MONTH
               WHEN 11 MOVE 30 TO WS-DAYS-IN-MONTH
               WHEN 12 MOVE 31 TO WS-DAYS-IN-MONTH
           END-EVALUATE.

       CHECK-LEAP-YEAR.
           MOVE 0 TO WS-IS-LEAP.
           IF FUNCTION MOD(WS-WORK-YEAR, 4) = 0
               IF FUNCTION MOD(WS-WORK-YEAR, 100) NOT = 0
                   MOVE 1 TO WS-IS-LEAP
               ELSE
                   IF FUNCTION MOD(WS-WORK-YEAR, 400) = 0
                       MOVE 1 TO WS-IS-LEAP
                   END-IF
               END-IF
           END-IF.
ENDCOBOL
cat > /app/accrued.cob << 'ENDCOBOL'
       IDENTIFICATION DIVISION.
       PROGRAM-ID. ACCRUED.

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
                           IF WS-CPN-CPN-FREQ = WS-COUPON-FREQ
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
ENDCOBOL
cat > /app/secfee.cob << 'ENDCOBOL'
       IDENTIFICATION DIVISION.
       PROGRAM-ID. SECFEE.

       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT RATE-FILE ASSIGN TO "/app/data/secrates.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-RATE-STATUS.

       DATA DIVISION.
       FILE SECTION.
       FD  RATE-FILE.
       COPY SECRATE REPLACING ==:PREFIX:== BY ==FD-RT==.

       WORKING-STORAGE SECTION.
       01  WS-RATE-STATUS          PIC XX.
       01  WS-EOF-RATE             PIC 9 VALUE 0.
       01  WS-CURRENT-RATE         PIC S9(3)V9(6) COMP-3
                                   VALUE 0.000008.
       01  WS-FOUND-RATE           PIC 9 VALUE 0.
       01  WS-TEMP-FEE             PIC S9(15)V9(6) COMP-3.

       COPY SECRATE REPLACING ==:PREFIX:== BY ==WS-RT==.

       LINKAGE SECTION.
       01  LS-PRINCIPAL            PIC S9(11)V99 COMP-3.
       01  LS-BUY-SELL             PIC X(1).
       01  LS-TRADE-DATE           PIC 9(8).
       01  LS-SEC-FEE              PIC S9(7)V99 COMP-3.

       PROCEDURE DIVISION USING LS-PRINCIPAL LS-BUY-SELL
                                LS-TRADE-DATE LS-SEC-FEE.
       MAIN-LOGIC.
           MOVE 0 TO LS-SEC-FEE.
           IF LS-BUY-SELL = 'S'
               PERFORM LOOKUP-RATE
               PERFORM CALCULATE-FEE
           END-IF.
           GOBACK.

       LOOKUP-RATE.
           MOVE 0 TO WS-FOUND-RATE.
           MOVE 0 TO WS-EOF-RATE.
           OPEN INPUT RATE-FILE.
           PERFORM UNTIL WS-EOF-RATE = 1
               READ RATE-FILE INTO WS-RT-SEC-RATE-REC
                   AT END MOVE 1 TO WS-EOF-RATE
                   NOT AT END
                       IF WS-RT-EFF-DATE <= LS-TRADE-DATE
                           MOVE WS-RT-RATE-PER-MIL TO WS-CURRENT-RATE
                           MOVE 1 TO WS-FOUND-RATE
                       END-IF
               END-READ
           END-PERFORM.
           CLOSE RATE-FILE.

       CALCULATE-FEE.
           COMPUTE WS-TEMP-FEE = LS-PRINCIPAL * WS-CURRENT-RATE.
           IF WS-TEMP-FEE < 0.01
               MOVE 0.01 TO LS-SEC-FEE
           ELSE
               MOVE WS-TEMP-FEE TO LS-SEC-FEE
           END-IF.
ENDCOBOL
cobc -m -I /app/copybooks -o /app/DAYCOUNT.so /app/daycount.cob
cobc -m -I /app/copybooks -o /app/ACCRUED.so /app/accrued.cob
cobc -m -I /app/copybooks -o /app/SECFEE.so /app/secfee.cob
cobc -x -I /app/copybooks -o /app/settle /app/settle.cob
export COB_LIBRARY_PATH=/app
/app/settle
