       IDENTIFICATION DIVISION.
       PROGRAM-ID. SECFEE.
      *================================================================*
      * SECFEE - Calculate SEC Section 31 transaction fee              *
      * Fee applies only to SELL transactions                          *
      *================================================================*

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
       01  WS-FEE-AMOUNT           PIC S9(7)V99 COMP-3.
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
