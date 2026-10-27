       IDENTIFICATION DIVISION.
       PROGRAM-ID. RRBCL02.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-IDX.
           05  WS-I                           PIC 999.
           05  WS-J                           PIC 999.
           05  WS-K                           PIC 999.
           05  WS-MC                          PIC 999.
           05  WS-FD                          PIC 9.

       01  WS-MEA.
           05  WS-ME OCCURS 540 TIMES        PIC S9(7)V99.

       01  WS-WRK.
           05  WS-DY                          PIC 9(4).
           05  WS-DM                          PIC 99.
           05  WS-RY                          PIC 9(4).
           05  WS-RM                          PIC 99.
           05  WS-AA                          PIC S9(3)V99.
           05  WS-H6                          PIC S9(12)V99.
           05  WS-HA                          PIC S9(7)V99.
           05  WS-T2                          PIC S9(7)V99.
           05  WS-R2                          PIC S9V9(6).
           05  WS-M2                          PIC 999.
           05  WS-F2                          PIC 99 VALUE 65.
           05  WS-CP                          PIC S9(7)V99.
           05  WS-WK                          PIC S9(9)V9(6).
           05  WS-TA                          PIC S9(12)V99.
           05  WS-TW                          PIC S9(12)V9(6).

       LINKAGE SECTION.
       01  LK-DOB                             PIC 9(8).
       01  LK-RET-DATE                        PIC 9(8).
       01  LK-SVC-YEARS                       PIC 9(2)V9.
       01  LK-EARN-TBL.
           05  LK-EARN-ENTRY OCCURS 45 TIMES.
               10  LK-EARN-YEAR              PIC 9(4).
               10  LK-EARN-AMT               PIC 9(7)V99.
       01  LK-T2M-CNT                         PIC 99.
       01  LK-T2M-TBL.
           05  LK-T2E OCCURS 80 TIMES.
               10  LK-TY                     PIC 9(4).
               10  LK-TM                     PIC 9(7)V99.
       01  LK-TIER2-OUT                       PIC S9(7)V99.

       PROCEDURE DIVISION USING LK-DOB LK-RET-DATE LK-SVC-YEARS
                                LK-EARN-TBL LK-T2M-CNT LK-T2M-TBL
                                LK-TIER2-OUT.
       0000-MAIN.
           PERFORM 1000-CALC-T2
           GOBACK.

       1000-CALC-T2.
           COMPUTE WS-DY =
               FUNCTION INTEGER-PART(LK-DOB / 10000)
           COMPUTE WS-DM =
               FUNCTION MOD(FUNCTION INTEGER-PART(LK-DOB / 100), 100)
           COMPUTE WS-RY =
               FUNCTION INTEGER-PART(LK-RET-DATE / 10000)
           COMPUTE WS-RM =
               FUNCTION MOD(FUNCTION INTEGER-PART(LK-RET-DATE/100), 100)

           COMPUTE WS-TW =
               WS-RY - WS-DY + (WS-RM - WS-DM) / 12
           COMPUTE WS-AA =
               FUNCTION INTEGER-PART(WS-TW * 100) / 100

           INITIALIZE WS-MEA
           MOVE 0 TO WS-MC

           PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > 45
               IF LK-EARN-YEAR(WS-J) > 0
                   MOVE 0 TO WS-FD
                   MOVE LK-EARN-AMT(WS-J) TO WS-CP
                   PERFORM VARYING WS-K FROM 1 BY 1
                       UNTIL WS-K > LK-T2M-CNT OR WS-FD = 1
                       IF LK-TY(WS-K) = LK-EARN-YEAR(WS-J)
                           IF LK-EARN-AMT(WS-J) > LK-TM(WS-K)
                               MOVE LK-TM(WS-K) TO WS-CP
                           END-IF
                           MOVE 1 TO WS-FD
                       END-IF
                   END-PERFORM

                   COMPUTE WS-TW = WS-CP * 100
                   COMPUTE WS-CP =
                       FUNCTION INTEGER-PART(WS-TW) / 100
                   COMPUTE WS-TW = WS-CP / 12
                   COMPUTE WS-TA =
                       FUNCTION INTEGER-PART(WS-TW * 100) / 100
                   PERFORM 12 TIMES
                       ADD 1 TO WS-MC
                       IF WS-MC <= 540
                           MOVE WS-TA TO WS-ME(WS-MC)
                       END-IF
                   END-PERFORM
               END-IF
           END-PERFORM

           PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > WS-MC - 1
               PERFORM VARYING WS-K FROM WS-J BY 1 UNTIL WS-K > WS-MC
                   IF WS-ME(WS-K) > WS-ME(WS-J)
                       MOVE WS-ME(WS-J) TO WS-TA
                       MOVE WS-ME(WS-K) TO WS-ME(WS-J)
                       MOVE WS-TA TO WS-ME(WS-K)
                   END-IF
               END-PERFORM
           END-PERFORM

           MOVE 0 TO WS-H6
           PERFORM VARYING WS-J FROM 1 BY 1 UNTIL WS-J > 60
               IF WS-J <= WS-MC
                   ADD WS-ME(WS-J) TO WS-H6
               END-IF
           END-PERFORM

           COMPUTE WS-TW = WS-H6 * 100
           COMPUTE WS-H6 = FUNCTION INTEGER-PART(WS-TW) / 100
           COMPUTE WS-TW = WS-H6 / 60
           COMPUTE WS-HA =
               FUNCTION INTEGER-PART(WS-TW * 100) / 100

           COMPUTE WS-TW = 0.007 * WS-HA * LK-SVC-YEARS
           COMPUTE WS-WK =
               FUNCTION INTEGER-PART(WS-TW * 10000) / 10000
           COMPUTE WS-TW = WS-WK * 100
           COMPUTE WS-T2 =
               FUNCTION INTEGER-PART(WS-TW) / 100

           IF LK-SVC-YEARS >= 30
               CONTINUE
           ELSE
               IF WS-AA < WS-F2
                   COMPUTE WS-M2 = (WS-F2 - WS-AA) * 12
                   IF WS-M2 > 36
                       COMPUTE WS-TW =
                           1 - (36 / 180) - ((WS-M2 - 36) / 240)
                       COMPUTE WS-R2 =
                           FUNCTION INTEGER-PART(WS-TW * 1000000)
                           / 1000000
                   ELSE
                       COMPUTE WS-TW = 1 - (WS-M2 / 180)
                       COMPUTE WS-R2 =
                           FUNCTION INTEGER-PART(WS-TW * 1000000)
                           / 1000000
                   END-IF
                   COMPUTE WS-TW = WS-T2 * WS-R2
                   COMPUTE WS-T2 =
                       FUNCTION INTEGER-PART(WS-TW * 100) / 100
               END-IF
           END-IF

           COMPUTE LK-TIER2-OUT =
               FUNCTION INTEGER-PART(WS-T2 * 100) / 100.

