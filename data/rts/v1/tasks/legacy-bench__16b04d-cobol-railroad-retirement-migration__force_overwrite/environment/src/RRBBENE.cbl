       IDENTIFICATION DIVISION.
       PROGRAM-ID. RRBBENE.

       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT EMPLOYEE-FILE ASSIGN TO "EMPLOYEE.DAT"
               ORGANIZATION IS LINE SEQUENTIAL
               FILE STATUS IS WS-ES.
           SELECT BENDPTS-FILE ASSIGN TO "BENDPTS.DAT"
               ORGANIZATION IS LINE SEQUENTIAL
               FILE STATUS IS WS-BS.
           SELECT IDXFACT-FILE ASSIGN TO "IDXFACT.DAT"
               ORGANIZATION IS LINE SEQUENTIAL
               FILE STATUS IS WS-IS.
           SELECT TIER2MAX-FILE ASSIGN TO "TIER2MAX.DAT"
               ORGANIZATION IS LINE SEQUENTIAL
               FILE STATUS IS WS-TS.
           SELECT BENEFITS-FILE ASSIGN TO "BENEFITS.DAT"
               ORGANIZATION IS BINARY SEQUENTIAL
               FILE STATUS IS WS-OS.
           SELECT REPORT-FILE ASSIGN TO "BENEFITS.RPT"
               ORGANIZATION IS LINE SEQUENTIAL
               FILE STATUS IS WS-RS.
           SELECT SUMMARY-FILE ASSIGN TO "SUMMARY.DAT"
               ORGANIZATION IS BINARY SEQUENTIAL
               FILE STATUS IS WS-SS.

       DATA DIVISION.
       FILE SECTION.
       FD  EMPLOYEE-FILE.
           COPY EMPREC.

       FD  BENDPTS-FILE.
           COPY BPTREC.

       FD  IDXFACT-FILE.
           COPY IDXREC.

       FD  TIER2MAX-FILE.
           COPY T2MREC.

       FD  BENEFITS-FILE.
           COPY BENREC.

       FD  REPORT-FILE.
           01  REPORT-LINE                    PIC X(132).

       FD  SUMMARY-FILE.
           COPY SUMREC.

       WORKING-STORAGE SECTION.
       01  WS-FSTAT.
           05  WS-ES                          PIC XX.
           05  WS-BS                          PIC XX.
           05  WS-IS                          PIC XX.
           05  WS-TS                          PIC XX.
           05  WS-OS                          PIC XX.
           05  WS-RS                          PIC XX.
           05  WS-SS                          PIC XX.

       01  WS-FLG.
           05  WS-EF1                         PIC 9 VALUE 0.
           05  WS-EF2                         PIC 9 VALUE 0.
           05  WS-EF3                         PIC 9 VALUE 0.
           05  WS-EF4                         PIC 9 VALUE 0.
           05  WS-FIRST-REC                   PIC 9 VALUE 1.

       01  WS-CTR.
           05  WS-I                           PIC 999.
           05  WS-J                           PIC 999.
           05  WS-BC                          PIC 99 VALUE 0.
           05  WS-IC                          PIC 99 VALUE 0.
           05  WS-TC                          PIC 99 VALUE 0.

       01  WS-RPT.
           05  WS-PAGE-NUM                    PIC 9(4) VALUE 0.
           05  WS-LINE-NUM                    PIC 99 VALUE 99.
           05  WS-LINES-PER-PAGE              PIC 99 VALUE 55.
           05  WS-REC-COUNT                   PIC 9(6) VALUE 0.

       01  WS-CTRL.
           05  WS-PREV-RET-YEAR               PIC 9(4) VALUE 0.
           05  WS-CURR-RET-YEAR               PIC 9(4).
           05  WS-YEAR-COUNT                  PIC 9(4) VALUE 0.
           05  WS-YEAR-T1-TOT                 PIC S9(11)V99 VALUE 0.
           05  WS-YEAR-T2-TOT                 PIC S9(11)V99 VALUE 0.
           05  WS-YEAR-TT-TOT                 PIC S9(11)V99 VALUE 0.

       01  WS-TOTALS.
           05  WS-GRAND-T1                    PIC S9(13)V99 VALUE 0.
           05  WS-GRAND-T2                    PIC S9(13)V99 VALUE 0.
           05  WS-GRAND-TT                    PIC S9(13)V99 VALUE 0.
           05  WS-VALID-COUNT                 PIC 9(6) VALUE 0.
           05  WS-ERROR-COUNT                 PIC 9(6) VALUE 0.
           05  WS-T1-MIN                      PIC S9(7)V99 VALUE 999999.
           05  WS-T1-MAX                      PIC S9(7)V99 VALUE 0.
           05  WS-T2-MIN                      PIC S9(7)V99 VALUE 999999.
           05  WS-T2-MAX                      PIC S9(7)V99 VALUE 0.

       01  WS-BPT.
           05  WS-BPE OCCURS 50 TIMES.
               10  WS-BY                      PIC 9(4).
               10  WS-B1                      PIC 9(5)V99.
               10  WS-B2                      PIC 9(5)V99.

       01  WS-IXT.
           05  WS-IXE OCCURS 80 TIMES.
               10  WS-IY                      PIC 9(4).
               10  WS-IA                      PIC 9(7)V99.
               10  WS-IF                      PIC 9(1)V9(6).

       01  WS-T2T.
           05  WS-T2E OCCURS 80 TIMES.
               10  WS-TY                      PIC 9(4).
               10  WS-TM                      PIC 9(7)V99.

       01  WS-CALL-DOB                        PIC 9(8).
       01  WS-CALL-RET                        PIC 9(8).
       01  WS-CALL-SVC                        PIC 9(2)V9.
       01  WS-CALL-EARN.
           05  WS-CE OCCURS 45 TIMES.
               10  WS-CEY                     PIC 9(4).
               10  WS-CEA                     PIC 9(7)V99.
       01  WS-T1-RESULT                       PIC S9(7)V99.
       01  WS-T2-RESULT                       PIC S9(7)V99.

       01  WS-WRK.
           05  WS-RY                          PIC 9(4).

       01  WS-OUT.
           05  WS-ON                          PIC X(9).
           05  WS-OE                          PIC X(30).
           05  WS-O1                          PIC S9(7)V99 COMP-3.
           05  WS-O2                          PIC S9(7)V99 COMP-3.
           05  WS-OT                          PIC S9(7)V99 COMP-3.
           05  WS-ST                          PIC X(1).
           05  WS-MG                          PIC X(50).

       01  WS-O1-NUM                          PIC S9(7)V99.
       01  WS-O2-NUM                          PIC S9(7)V99.
       01  WS-OT-NUM                          PIC S9(7)V99.

       01  WS-HDR1.
           05  FILLER                         PIC X(45) VALUE SPACES.
           05  FILLER                         PIC X(42)
               VALUE "RAILROAD RETIREMENT BOARD BENEFITS REPORT".
           05  FILLER                         PIC X(30) VALUE SPACES.
           05  FILLER                         PIC X(5) VALUE "PAGE ".
           05  WS-HDR1-PAGE                   PIC Z,ZZ9.
           05  FILLER                         PIC X(4) VALUE SPACES.

       01  WS-HDR2.
           05  FILLER                         PIC X(132) VALUE ALL "=".

       01  WS-HDR3.
           05  FILLER                         PIC X(1) VALUE SPACES.
           05  FILLER                         PIC X(9) VALUE "SSN".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(30) VALUE "NAME".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(8) VALUE "RET YEAR".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(12) VALUE "TIER 1".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(12) VALUE "TIER 2".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(12) VALUE "TOTAL".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(6) VALUE "STATUS".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(28) VALUE "MESSAGE".

       01  WS-HDR4.
           05  FILLER                         PIC X(1) VALUE SPACES.
           05  FILLER                         PIC X(9) VALUE ALL "-".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(30) VALUE ALL "-".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(8) VALUE ALL "-".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(12) VALUE ALL "-".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(12) VALUE ALL "-".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(12) VALUE ALL "-".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(6) VALUE ALL "-".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(28) VALUE ALL "-".

       01  WS-DTL.
           05  FILLER                         PIC X(1) VALUE SPACES.
           05  WS-DTL-SSN                     PIC X(9).
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  WS-DTL-NAME                    PIC X(30).
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  WS-DTL-RET-YEAR                PIC 9(4).
           05  FILLER                         PIC X(6) VALUE SPACES.
           05  WS-DTL-T1                      PIC ZZZ,ZZ9.99.
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  WS-DTL-T2                      PIC ZZZ,ZZ9.99.
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  WS-DTL-TOT                     PIC ZZZ,ZZ9.99.
           05  FILLER                         PIC X(4) VALUE SPACES.
           05  WS-DTL-ST                      PIC X(1).
           05  FILLER                         PIC X(7) VALUE SPACES.
           05  WS-DTL-MSG                     PIC X(28).

       01  WS-SUBTOT.
           05  FILLER                         PIC X(1) VALUE SPACES.
           05  FILLER                         PIC X(9) VALUE ALL "*".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(18) VALUE
               "SUBTOTAL FOR YEAR ".
           05  WS-SUB-YEAR                    PIC 9(4).
           05  FILLER                         PIC X(8) VALUE SPACES.
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(8) VALUE SPACES.
           05  WS-SUB-T1                      PIC Z,ZZZ,ZZ9.99.
           05  FILLER                         PIC X(1) VALUE SPACES.
           05  WS-SUB-T2                      PIC Z,ZZZ,ZZ9.99.
           05  FILLER                         PIC X(1) VALUE SPACES.
           05  WS-SUB-TOT                     PIC Z,ZZZ,ZZ9.99.
           05  FILLER                         PIC X(4) VALUE SPACES.
           05  FILLER                         PIC X(5) VALUE "COUNT".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  WS-SUB-CNT                     PIC ZZ,ZZ9.
           05  FILLER                         PIC X(16) VALUE SPACES.

       01  WS-GRAND.
           05  FILLER                         PIC X(1) VALUE SPACES.
           05  FILLER                         PIC X(9) VALUE ALL "=".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(30) VALUE
               "GRAND TOTALS".
           05  FILLER                         PIC X(2) VALUE SPACES.
           05  FILLER                         PIC X(8) VALUE SPACES.
           05  WS-GRD-T1                      PIC Z,ZZZ,ZZ9.99.
           05  FILLER                         PIC X(1) VALUE SPACES.
           05  WS-GRD-T2                      PIC Z,ZZZ,ZZ9.99.
           05  FILLER                         PIC X(1) VALUE SPACES.
           05  WS-GRD-TOT                     PIC Z,ZZZ,ZZ9.99.
           05  FILLER                         PIC X(4) VALUE SPACES.
           05  FILLER                         PIC X(5) VALUE "RECS:".
           05  FILLER                         PIC X(1) VALUE SPACES.
           05  WS-GRD-CNT                     PIC ZZZ,ZZ9.
           05  FILLER                         PIC X(15) VALUE SPACES.

       01  WS-SUMOUT.
           05  WS-SUM-TOT-RECS                PIC 9(6).
           05  WS-SUM-VALID                   PIC 9(6).
           05  WS-SUM-ERROR                   PIC 9(6).
           05  WS-SUM-T1-TOT                  PIC S9(11)V99 COMP-3.
           05  WS-SUM-T2-TOT                  PIC S9(11)V99 COMP-3.
           05  WS-SUM-TT-TOT                  PIC S9(11)V99 COMP-3.
           05  WS-SUM-T1-AVG                  PIC S9(9)V99 COMP-3.
           05  WS-SUM-T2-AVG                  PIC S9(9)V99 COMP-3.
           05  WS-SUM-T1-MIN                  PIC S9(7)V99 COMP-3.
           05  WS-SUM-T1-MAX                  PIC S9(7)V99 COMP-3.
           05  WS-SUM-T2-MIN                  PIC S9(7)V99 COMP-3.
           05  WS-SUM-T2-MAX                  PIC S9(7)V99 COMP-3.

       PROCEDURE DIVISION.
       0000-MAIN.
           PERFORM 1000-INIT
           PERFORM 2000-LOAD
           PERFORM 3000-PROC
           PERFORM 8000-FINAL
           PERFORM 9000-TERM
           STOP RUN.

       1000-INIT.
           OPEN INPUT EMPLOYEE-FILE
           OPEN INPUT BENDPTS-FILE
           OPEN INPUT IDXFACT-FILE
           OPEN INPUT TIER2MAX-FILE
           OPEN OUTPUT BENEFITS-FILE
           OPEN OUTPUT REPORT-FILE
           OPEN OUTPUT SUMMARY-FILE.

       2000-LOAD.
           PERFORM UNTIL WS-EF2 = 1
               READ BENDPTS-FILE
                   AT END
                       MOVE 1 TO WS-EF2
                   NOT AT END
                       ADD 1 TO WS-BC
                       MOVE BP-YEAR TO WS-BY(WS-BC)
                       MOVE BP-FIRST TO WS-B1(WS-BC)
                       MOVE BP-SECOND TO WS-B2(WS-BC)
               END-READ
           END-PERFORM
           CLOSE BENDPTS-FILE

           PERFORM UNTIL WS-EF3 = 1
               READ IDXFACT-FILE
                   AT END
                       MOVE 1 TO WS-EF3
                   NOT AT END
                       ADD 1 TO WS-IC
                       MOVE IDX-YEAR TO WS-IY(WS-IC)
                       MOVE IDX-AWI TO WS-IA(WS-IC)
                       MOVE IDX-FACTOR TO WS-IF(WS-IC)
               END-READ
           END-PERFORM
           CLOSE IDXFACT-FILE

           PERFORM UNTIL WS-EF4 = 1
               READ TIER2MAX-FILE
                   AT END
                       MOVE 1 TO WS-EF4
                   NOT AT END
                       ADD 1 TO WS-TC
                       MOVE T2M-YEAR TO WS-TY(WS-TC)
                       MOVE T2M-MAX TO WS-TM(WS-TC)
               END-READ
           END-PERFORM
           CLOSE TIER2MAX-FILE.

       3000-PROC.
           PERFORM UNTIL WS-EF1 = 1
               READ EMPLOYEE-FILE
                   AT END
                       MOVE 1 TO WS-EF1
                   NOT AT END
                       PERFORM 4000-CALC
                       PERFORM 5000-WRT
                       PERFORM 5500-RPT
               END-READ
           END-PERFORM

           IF WS-REC-COUNT > 0
               PERFORM 6000-YEAR-BREAK
           END-IF

           CLOSE EMPLOYEE-FILE.

       4000-CALC.
           MOVE EMP-SSN TO WS-ON
           MOVE EMP-NAME TO WS-OE
           MOVE "V" TO WS-ST
           MOVE SPACES TO WS-MG

           COMPUTE WS-RY =
               FUNCTION INTEGER-PART(EMP-RET-DATE / 10000)
           MOVE WS-RY TO WS-CURR-RET-YEAR

           IF EMP-SVC-YEARS < 5
               MOVE "E" TO WS-ST
               MOVE "INSUFFICIENT SERVICE YEARS" TO WS-MG
               MOVE 0 TO WS-O1
               MOVE 0 TO WS-O2
               MOVE 0 TO WS-OT
               MOVE 0 TO WS-O1-NUM
               MOVE 0 TO WS-O2-NUM
               MOVE 0 TO WS-OT-NUM
               ADD 1 TO WS-ERROR-COUNT
           ELSE
               MOVE EMP-DOB TO WS-CALL-DOB
               MOVE EMP-RET-DATE TO WS-CALL-RET
               MOVE EMP-SVC-YEARS TO WS-CALL-SVC

               PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > 45
                   MOVE EMP-EARN-YEAR(WS-J) TO WS-CEY(WS-J)
                   MOVE EMP-EARN-AMT(WS-J) TO WS-CEA(WS-J)
               END-PERFORM

               CALL "RRBCL01" USING WS-CALL-DOB WS-CALL-RET
                   WS-CALL-SVC WS-CALL-EARN WS-BC WS-BPT
                   WS-IC WS-IXT WS-T1-RESULT

               CALL "RRBCL02" USING WS-CALL-DOB WS-CALL-RET
                   WS-CALL-SVC WS-CALL-EARN WS-TC WS-T2T
                   WS-T2-RESULT

               MOVE WS-T1-RESULT TO WS-O1
               MOVE WS-T1-RESULT TO WS-O1-NUM
               MOVE WS-T2-RESULT TO WS-O2
               MOVE WS-T2-RESULT TO WS-O2-NUM
               COMPUTE WS-OT = WS-O1 + WS-O2
               COMPUTE WS-OT-NUM = WS-O1-NUM + WS-O2-NUM

               ADD 1 TO WS-VALID-COUNT
               ADD WS-O1-NUM TO WS-GRAND-T1
               ADD WS-O2-NUM TO WS-GRAND-T2
               ADD WS-OT-NUM TO WS-GRAND-TT

               IF WS-O1-NUM < WS-T1-MIN
                   MOVE WS-O1-NUM TO WS-T1-MIN
               END-IF
               IF WS-O1-NUM > WS-T1-MAX
                   MOVE WS-O1-NUM TO WS-T1-MAX
               END-IF
               IF WS-O2-NUM < WS-T2-MIN
                   MOVE WS-O2-NUM TO WS-T2-MIN
               END-IF
               IF WS-O2-NUM > WS-T2-MAX
                   MOVE WS-O2-NUM TO WS-T2-MAX
               END-IF
           END-IF.

       5000-WRT.
           MOVE WS-ON TO BEN-SSN
           MOVE WS-OE TO BEN-NAME
           MOVE WS-O1 TO BEN-TIER1
           MOVE WS-O2 TO BEN-TIER2
           MOVE WS-OT TO BEN-TOTAL
           MOVE WS-ST TO BEN-STATUS
           MOVE WS-MG TO BEN-MSG
           WRITE BENEFIT-RECORD.

       5500-RPT.
           ADD 1 TO WS-REC-COUNT

           IF WS-FIRST-REC = 1
               MOVE WS-CURR-RET-YEAR TO WS-PREV-RET-YEAR
               MOVE 0 TO WS-FIRST-REC
               PERFORM 5600-PAGE-HDR
           ELSE
               IF WS-CURR-RET-YEAR NOT = WS-PREV-RET-YEAR
                   PERFORM 6000-YEAR-BREAK
                   MOVE WS-CURR-RET-YEAR TO WS-PREV-RET-YEAR
               END-IF
           END-IF

           IF WS-LINE-NUM >= WS-LINES-PER-PAGE
               PERFORM 5600-PAGE-HDR
           END-IF

           MOVE WS-ON TO WS-DTL-SSN
           MOVE WS-OE TO WS-DTL-NAME
           MOVE WS-CURR-RET-YEAR TO WS-DTL-RET-YEAR
           MOVE WS-O1-NUM TO WS-DTL-T1
           MOVE WS-O2-NUM TO WS-DTL-T2
           MOVE WS-OT-NUM TO WS-DTL-TOT
           MOVE WS-ST TO WS-DTL-ST
           MOVE WS-MG TO WS-DTL-MSG
           WRITE REPORT-LINE FROM WS-DTL
           ADD 1 TO WS-LINE-NUM

           ADD 1 TO WS-YEAR-COUNT
           ADD WS-O1-NUM TO WS-YEAR-T1-TOT
           ADD WS-O2-NUM TO WS-YEAR-T2-TOT
           ADD WS-OT-NUM TO WS-YEAR-TT-TOT.

       5600-PAGE-HDR.
           ADD 1 TO WS-PAGE-NUM
           MOVE WS-PAGE-NUM TO WS-HDR1-PAGE

           IF WS-PAGE-NUM > 1
               MOVE SPACES TO REPORT-LINE
               WRITE REPORT-LINE AFTER PAGE
           END-IF

           WRITE REPORT-LINE FROM WS-HDR1
           WRITE REPORT-LINE FROM WS-HDR2
           MOVE SPACES TO REPORT-LINE
           WRITE REPORT-LINE
           WRITE REPORT-LINE FROM WS-HDR3
           WRITE REPORT-LINE FROM WS-HDR4
           MOVE 6 TO WS-LINE-NUM.

       6000-YEAR-BREAK.
           MOVE SPACES TO REPORT-LINE
           WRITE REPORT-LINE
           MOVE WS-PREV-RET-YEAR TO WS-SUB-YEAR
           MOVE WS-YEAR-T1-TOT TO WS-SUB-T1
           MOVE WS-YEAR-T2-TOT TO WS-SUB-T2
           MOVE WS-YEAR-TT-TOT TO WS-SUB-TOT
           MOVE WS-YEAR-COUNT TO WS-SUB-CNT
           WRITE REPORT-LINE FROM WS-SUBTOT
           MOVE SPACES TO REPORT-LINE
           WRITE REPORT-LINE
           ADD 3 TO WS-LINE-NUM

           MOVE 0 TO WS-YEAR-COUNT
           MOVE 0 TO WS-YEAR-T1-TOT
           MOVE 0 TO WS-YEAR-T2-TOT
           MOVE 0 TO WS-YEAR-TT-TOT.

       8000-FINAL.
           MOVE SPACES TO REPORT-LINE
           WRITE REPORT-LINE
           WRITE REPORT-LINE FROM WS-HDR2
           MOVE WS-GRAND-T1 TO WS-GRD-T1
           MOVE WS-GRAND-T2 TO WS-GRD-T2
           MOVE WS-GRAND-TT TO WS-GRD-TOT
           MOVE WS-REC-COUNT TO WS-GRD-CNT
           WRITE REPORT-LINE FROM WS-GRAND

           MOVE WS-REC-COUNT TO WS-SUM-TOT-RECS
           MOVE WS-VALID-COUNT TO WS-SUM-VALID
           MOVE WS-ERROR-COUNT TO WS-SUM-ERROR
           MOVE WS-GRAND-T1 TO WS-SUM-T1-TOT
           MOVE WS-GRAND-T2 TO WS-SUM-T2-TOT
           MOVE WS-GRAND-TT TO WS-SUM-TT-TOT

           IF WS-VALID-COUNT > 0
               COMPUTE WS-SUM-T1-AVG =
                   WS-GRAND-T1 / WS-VALID-COUNT
               COMPUTE WS-SUM-T2-AVG =
                   WS-GRAND-T2 / WS-VALID-COUNT
           ELSE
               MOVE 0 TO WS-SUM-T1-AVG
               MOVE 0 TO WS-SUM-T2-AVG
           END-IF

           IF WS-T1-MIN = 999999
               MOVE 0 TO WS-SUM-T1-MIN
           ELSE
               MOVE WS-T1-MIN TO WS-SUM-T1-MIN
           END-IF
           MOVE WS-T1-MAX TO WS-SUM-T1-MAX

           IF WS-T2-MIN = 999999
               MOVE 0 TO WS-SUM-T2-MIN
           ELSE
               MOVE WS-T2-MIN TO WS-SUM-T2-MIN
           END-IF
           MOVE WS-T2-MAX TO WS-SUM-T2-MAX

           WRITE SUMMARY-RECORD FROM WS-SUMOUT.

       9000-TERM.
           CLOSE BENEFITS-FILE
           CLOSE REPORT-FILE
           CLOSE SUMMARY-FILE.
