       IDENTIFICATION DIVISION.
       PROGRAM-ID. SETTLE.
      *================================================================*
      * SETTLE - Main settlement processing program                    *
      * Reads trades, calculates settlement amounts, writes output     *
      *================================================================*

       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT TRADE-FILE ASSIGN TO "/app/data/trades.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-TRADE-STATUS.
           SELECT SETTLE-FILE ASSIGN TO "/app/output/settlement.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-SETTLE-STATUS.
           SELECT REPORT-FILE ASSIGN TO "/app/output/rule606.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-REPORT-STATUS.

       DATA DIVISION.
       FILE SECTION.
       FD  TRADE-FILE.
       COPY TRADEREC REPLACING ==:PREFIX:== BY ==FD-TRD==.

       FD  SETTLE-FILE.
       COPY SETTLOUT REPLACING ==:PREFIX:== BY ==FD-STL==.

       FD  REPORT-FILE.
       COPY RULE606 REPLACING ==:PREFIX:== BY ==FD-RPT==.

       WORKING-STORAGE SECTION.
       01  WS-TRADE-STATUS         PIC XX.
       01  WS-SETTLE-STATUS        PIC XX.
       01  WS-REPORT-STATUS        PIC XX.
       01  WS-EOF-TRADES           PIC 9 VALUE 0.
       01  WS-TRADE-COUNT          PIC 9(6) VALUE 0.

       COPY TRADEREC REPLACING ==:PREFIX:== BY ==WS-TRD==.
       COPY SETTLOUT REPLACING ==:PREFIX:== BY ==WS-STL==.

       01  WS-PRINCIPAL            PIC S9(11)V99 COMP-3.
       01  WS-ACCR-INTEREST        PIC S9(9)V99 COMP-3.
       01  WS-SEC-FEE              PIC S9(7)V99 COMP-3.
       01  WS-DAYS-ACCRUED         PIC S9(5) COMP.
       01  WS-DAY-CONV             PIC X(1).

       01  WS-VENUE-TABLE.
           05  WS-VENUE-ENTRY OCCURS 10 TIMES.
               10  WS-VEN-CODE         PIC X(3).
               10  WS-VEN-TOTAL-ORD    PIC S9(9) COMP-3.
               10  WS-VEN-TOTAL-SHR    PIC S9(11) COMP-3.
               10  WS-VEN-MARKET-ORD   PIC S9(9) COMP-3.
               10  WS-VEN-LIMIT-ORD    PIC S9(9) COMP-3.
               10  WS-VEN-PFOF         PIC S9(9)V99 COMP-3.
       01  WS-VENUE-COUNT          PIC 9(2) VALUE 0.
       01  WS-VENUE-IDX            PIC 9(2).
       01  WS-VENUE-FOUND          PIC 9 VALUE 0.

       COPY RULE606 REPLACING ==:PREFIX:== BY ==WS-RPT==.

       PROCEDURE DIVISION.
       MAIN-LOGIC.
           PERFORM INITIALIZE-VENUES.
           PERFORM PROCESS-TRADES.
           PERFORM WRITE-RULE606-REPORT.
           STOP RUN.

       INITIALIZE-VENUES.
           MOVE 0 TO WS-VENUE-COUNT.
           PERFORM VARYING WS-VENUE-IDX FROM 1 BY 1
               UNTIL WS-VENUE-IDX > 10
               MOVE SPACES TO WS-VEN-CODE(WS-VENUE-IDX)
               MOVE 0 TO WS-VEN-TOTAL-ORD(WS-VENUE-IDX)
               MOVE 0 TO WS-VEN-TOTAL-SHR(WS-VENUE-IDX)
               MOVE 0 TO WS-VEN-MARKET-ORD(WS-VENUE-IDX)
               MOVE 0 TO WS-VEN-LIMIT-ORD(WS-VENUE-IDX)
               MOVE 0 TO WS-VEN-PFOF(WS-VENUE-IDX)
           END-PERFORM.

       PROCESS-TRADES.
           OPEN INPUT TRADE-FILE.
           OPEN OUTPUT SETTLE-FILE.
           MOVE 0 TO WS-EOF-TRADES.
           PERFORM UNTIL WS-EOF-TRADES = 1
               READ TRADE-FILE INTO WS-TRD-TRADE-REC
                   AT END MOVE 1 TO WS-EOF-TRADES
                   NOT AT END
                       ADD 1 TO WS-TRADE-COUNT
                       PERFORM CALCULATE-SETTLEMENT
                       PERFORM UPDATE-VENUE-STATS
                       PERFORM WRITE-SETTLEMENT-RECORD
               END-READ
           END-PERFORM.
           CLOSE TRADE-FILE.
           CLOSE SETTLE-FILE.

       CALCULATE-SETTLEMENT.
           COMPUTE WS-PRINCIPAL =
               WS-TRD-TRADE-QTY * WS-TRD-TRADE-PRICE.
           CALL "ACCRUED" USING WS-TRD-BOND-CUSIP
                                WS-TRD-SETTLE-DATE
                                WS-TRD-TRADE-QTY
                                WS-ACCR-INTEREST
                                WS-DAYS-ACCRUED
                                WS-DAY-CONV.
           CALL "SECFEE" USING WS-PRINCIPAL
                               WS-TRD-BUY-SELL-IND
                               WS-TRD-TRADE-DATE
                               WS-SEC-FEE.
           MOVE WS-TRD-TRADE-ID TO WS-STL-OUT-TRADE-ID.
           MOVE WS-TRD-BOND-CUSIP TO WS-STL-OUT-CUSIP.
           MOVE WS-PRINCIPAL TO WS-STL-OUT-PRINCIPAL.
           MOVE WS-ACCR-INTEREST TO WS-STL-OUT-ACCR-INT.
           MOVE WS-SEC-FEE TO WS-STL-OUT-SEC-FEE.
           MOVE WS-DAYS-ACCRUED TO WS-STL-OUT-DAYS-ACCR.
           IF WS-TRD-BUY-SELL-IND = 'B'
               COMPUTE WS-STL-OUT-NET-AMOUNT =
                   WS-PRINCIPAL + WS-ACCR-INTEREST
           ELSE
               COMPUTE WS-STL-OUT-NET-AMOUNT =
                   WS-PRINCIPAL + WS-ACCR-INTEREST - WS-SEC-FEE
           END-IF.

       UPDATE-VENUE-STATS.
           MOVE 0 TO WS-VENUE-FOUND.
           PERFORM VARYING WS-VENUE-IDX FROM 1 BY 1
               UNTIL WS-VENUE-IDX > WS-VENUE-COUNT
                  OR WS-VENUE-FOUND = 1
               IF WS-VEN-CODE(WS-VENUE-IDX) = WS-TRD-VENUE-CODE
                   MOVE 1 TO WS-VENUE-FOUND
               END-IF
           END-PERFORM.
           IF WS-VENUE-FOUND = 0
               ADD 1 TO WS-VENUE-COUNT
               MOVE WS-VENUE-COUNT TO WS-VENUE-IDX
               MOVE WS-TRD-VENUE-CODE TO WS-VEN-CODE(WS-VENUE-IDX)
           ELSE
               SUBTRACT 1 FROM WS-VENUE-IDX
           END-IF.
           ADD 1 TO WS-VEN-TOTAL-ORD(WS-VENUE-IDX).
           ADD WS-TRD-TRADE-QTY TO WS-VEN-TOTAL-SHR(WS-VENUE-IDX).
           IF WS-TRD-TRADE-PRICE > 0
               ADD 1 TO WS-VEN-LIMIT-ORD(WS-VENUE-IDX)
           ELSE
               ADD 1 TO WS-VEN-MARKET-ORD(WS-VENUE-IDX)
           END-IF.
           COMPUTE WS-VEN-PFOF(WS-VENUE-IDX) =
               WS-VEN-PFOF(WS-VENUE-IDX) +
               (WS-TRD-TRADE-QTY * 0.0001).

       WRITE-SETTLEMENT-RECORD.
           WRITE FD-STL-SETTLE-OUT-REC FROM WS-STL-SETTLE-OUT-REC.

       WRITE-RULE606-REPORT.
           OPEN OUTPUT REPORT-FILE.
           PERFORM VARYING WS-VENUE-IDX FROM 1 BY 1
               UNTIL WS-VENUE-IDX > WS-VENUE-COUNT
               MOVE WS-VEN-CODE(WS-VENUE-IDX) TO WS-RPT-RPT-VENUE
               MOVE WS-VEN-TOTAL-ORD(WS-VENUE-IDX) TO
                   WS-RPT-RPT-TOTAL-ORDERS
               MOVE WS-VEN-TOTAL-SHR(WS-VENUE-IDX) TO
                   WS-RPT-RPT-TOTAL-SHARES
               MOVE WS-VEN-MARKET-ORD(WS-VENUE-IDX) TO
                   WS-RPT-RPT-MARKET-ORD
               MOVE WS-VEN-LIMIT-ORD(WS-VENUE-IDX) TO
                   WS-RPT-RPT-LIMIT-ORD
               MOVE WS-VEN-PFOF(WS-VENUE-IDX) TO WS-RPT-RPT-PFOF-AMT
               WRITE FD-RPT-RULE606-REC FROM WS-RPT-RULE606-REC
           END-PERFORM.
           CLOSE REPORT-FILE.
