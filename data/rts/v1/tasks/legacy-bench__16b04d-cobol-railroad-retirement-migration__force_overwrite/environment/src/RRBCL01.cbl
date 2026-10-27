       IDENTIFICATION DIVISION.
       PROGRAM-ID. RRBCL01.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-IDX.
           05  WS-I                           PIC 999.
           05  WS-J                           PIC 999.
           05  WS-K                           PIC 999.
           05  WS-EC                          PIC 99.
           05  WS-FD                          PIC 9.

       01  WS-IEA.
           05  WS-IE OCCURS 45 TIMES         PIC S9(9)V99.

       01  WS-SEA.
           05  WS-SE OCCURS 45 TIMES         PIC S9(9)V99.

       01  WS-WRK.
           05  WS-EY                          PIC 9(4).
           05  WS-DY                          PIC 9(4).
           05  WS-DM                          PIC 99.
           05  WS-RY                          PIC 9(4).
           05  WS-RM                          PIC 99.
           05  WS-AA                          PIC S9(3)V99.
           05  WS-MN                          PIC 999.
           05  WS-TI                          PIC S9(12)V99.
           05  WS-AM                          PIC S9(7).
           05  WS-AW                          PIC S9(12)V99.
           05  WS-PI                          PIC S9(7)V99.
           05  WS-PW                          PIC S9(9)V9(4).
           05  WS-P1                          PIC 9(5)V99.
           05  WS-P2                          PIC 9(5)V99.
           05  WS-T1                          PIC S9(7)V99.
           05  WS-RF                          PIC S9V9(6).
           05  WS-FR                          PIC 99 VALUE 66.
           05  WS-TW                          PIC S9(12)V9(6).
           05  WS-IW                          PIC S9(9)V9(6).

       01  WS-TMP.
           05  WS-TE                          PIC S9(9)V99.

       LINKAGE SECTION.
       01  LK-DOB                             PIC 9(8).
       01  LK-RET-DATE                        PIC 9(8).
       01  LK-SVC-YEARS                       PIC 9(2)V9.
       01  LK-EARN-TBL.
           05  LK-EARN-ENTRY OCCURS 45 TIMES.
               10  LK-EARN-YEAR              PIC 9(4).
               10  LK-EARN-AMT               PIC 9(7)V99.
       01  LK-BPT-CNT                         PIC 99.
       01  LK-BPT-TBL.
           05  LK-BPE OCCURS 50 TIMES.
               10  LK-BY                     PIC 9(4).
               10  LK-B1                     PIC 9(5)V99.
               10  LK-B2                     PIC 9(5)V99.
       01  LK-IDX-CNT                         PIC 99.
       01  LK-IDX-TBL.
           05  LK-IXE OCCURS 80 TIMES.
               10  LK-IY                     PIC 9(4).
               10  LK-IA                     PIC 9(7)V99.
               10  LK-IF                     PIC 9(1)V9(6).
       01  LK-TIER1-OUT                       PIC S9(7)V99.

       PROCEDURE DIVISION USING LK-DOB LK-RET-DATE LK-SVC-YEARS
                                LK-EARN-TBL LK-BPT-CNT LK-BPT-TBL
                                LK-IDX-CNT LK-IDX-TBL LK-TIER1-OUT.
       0000-MAIN.
           PERFORM 1000-CALC-T1
           GOBACK.

       1000-CALC-T1.
           COMPUTE WS-DY =
               FUNCTION INTEGER-PART(LK-DOB / 10000)
           COMPUTE WS-DM =
               FUNCTION MOD(FUNCTION INTEGER-PART(LK-DOB / 100), 100)
           COMPUTE WS-RY =
               FUNCTION INTEGER-PART(LK-RET-DATE / 10000)
           COMPUTE WS-RM =
               FUNCTION MOD(FUNCTION INTEGER-PART(LK-RET-DATE/100), 100)

           COMPUTE WS-EY = WS-DY + 62
           COMPUTE WS-TW = WS-RY - WS-DY + (WS-RM - WS-DM) / 12
           COMPUTE WS-AA =
               FUNCTION INTEGER-PART(WS-TW * 100) / 100

           INITIALIZE WS-IEA
           INITIALIZE WS-SEA
           MOVE 0 TO WS-EC

           PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > 45
               IF LK-EARN-YEAR(WS-J) > 0
                   ADD 1 TO WS-EC
                   MOVE 0 TO WS-FD
                   PERFORM VARYING WS-K FROM 1 BY 1
                       UNTIL WS-K > LK-IDX-CNT OR WS-FD = 1
                       IF LK-IY(WS-K) = LK-EARN-YEAR(WS-J)
                           COMPUTE WS-IW =
                               LK-EARN-AMT(WS-J) * LK-IF(WS-K)
                           COMPUTE WS-IE(WS-EC) =
                               FUNCTION INTEGER-PART(WS-IW * 100) / 100
                           MOVE 1 TO WS-FD
                       END-IF
                   END-PERFORM
                   IF WS-FD = 0
                       MOVE LK-EARN-AMT(WS-J) TO WS-IE(WS-EC)
                   END-IF
               END-IF
           END-PERFORM

           PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > WS-EC
               MOVE WS-IE(WS-J) TO WS-SE(WS-J)
           END-PERFORM

           PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > WS-EC - 1
               PERFORM VARYING WS-K FROM WS-J BY 1 UNTIL WS-K > WS-EC
                   IF WS-SE(WS-K) > WS-SE(WS-J)
                       MOVE WS-SE(WS-J) TO WS-TE
                       MOVE WS-SE(WS-K) TO WS-SE(WS-J)
                       MOVE WS-TE TO WS-SE(WS-K)
                   END-IF
               END-PERFORM
           END-PERFORM

           MOVE 0 TO WS-TI
           PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > 35
               IF WS-J <= WS-EC
                   ADD WS-SE(WS-J) TO WS-TI
               END-IF
           END-PERFORM

           COMPUTE WS-TW = WS-TI / 420
           COMPUTE WS-AW =
               FUNCTION INTEGER-PART(WS-TW * 100) / 100
           COMPUTE WS-AM = FUNCTION INTEGER-PART(WS-AW)

           MOVE 0 TO WS-FD
           MOVE 0 TO WS-P1
           MOVE 0 TO WS-P2
           PERFORM VARYING WS-J FROM 1 BY 1
               UNTIL WS-J > LK-BPT-CNT OR WS-FD = 1
               IF LK-BY(WS-J) = WS-EY
                   MOVE LK-B1(WS-J) TO WS-P1
                   MOVE LK-B2(WS-J) TO WS-P2
                   MOVE 1 TO WS-FD
               END-IF
           END-PERFORM

           IF WS-FD = 0
               MOVE 1174.00 TO WS-P1
               MOVE 7078.00 TO WS-P2
           END-IF

           MOVE 0 TO WS-PW
           IF WS-AM <= WS-P1
               COMPUTE WS-TW = WS-AM * 0.90
               COMPUTE WS-PW =
                   FUNCTION INTEGER-PART(WS-TW * 10000) / 10000
           ELSE IF WS-AM <= WS-P2
               COMPUTE WS-TW =
                   (WS-P1 * 0.90) + ((WS-AM - WS-P1) * 0.32)
               COMPUTE WS-PW =
                   FUNCTION INTEGER-PART(WS-TW * 10000) / 10000
           ELSE
               COMPUTE WS-TW =
                   (WS-P1 * 0.90) + ((WS-P2 - WS-P1) * 0.32) +
                   ((WS-AM - WS-P2) * 0.15)
               COMPUTE WS-PW =
                   FUNCTION INTEGER-PART(WS-TW * 10000) / 10000
           END-IF
           END-IF

           COMPUTE WS-TW = WS-PW * 10
           COMPUTE WS-PI =
               FUNCTION INTEGER-PART(WS-TW) / 10
           COMPUTE WS-T1 =
               FUNCTION INTEGER-PART(WS-PI * 100) / 100

           IF LK-SVC-YEARS >= 30
               CONTINUE
           ELSE
               IF WS-AA < WS-FR
                   COMPUTE WS-MN = (WS-FR - WS-AA) * 12
                   IF WS-MN > 36
                       COMPUTE WS-TW =
                           1 - (36 * 5 / 900) -
                           ((WS-MN - 36) * 5 / 1200)
                       COMPUTE WS-RF =
                           FUNCTION INTEGER-PART(WS-TW * 1000000)
                           / 1000000
                   ELSE
                       COMPUTE WS-TW = 1 - (WS-MN * 5 / 900)
                       COMPUTE WS-RF =
                           FUNCTION INTEGER-PART(WS-TW * 1000000)
                           / 1000000
                   END-IF
                   COMPUTE WS-TW = WS-T1 * WS-RF
                   COMPUTE WS-T1 =
                       FUNCTION INTEGER-PART(WS-TW * 100) / 100
               END-IF
           END-IF

           COMPUTE LK-TIER1-OUT =
               FUNCTION INTEGER-PART(WS-T1 * 100) / 100.
