       IDENTIFICATION DIVISION.
       PROGRAM-ID. DAYCOUNT.
      *================================================================*
      * DAYCOUNT - Calculate days between two dates                    *
      * Input: LS-START-DATE, LS-END-DATE, LS-DAY-CONV                *
      * Output: LS-DAYS-RESULT                                         *
      *================================================================*

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-START-YEAR          PIC 9(4).
       01  WS-START-MONTH         PIC 9(2).
       01  WS-START-DAY           PIC 9(2).
       01  WS-END-YEAR            PIC 9(4).
       01  WS-END-MONTH           PIC 9(2).
       01  WS-END-DAY             PIC 9(2).
       01  WS-YEAR-DIFF           PIC S9(4) COMP.
       01  WS-MONTH-DIFF          PIC S9(4) COMP.
       01  WS-DAY-DIFF            PIC S9(4) COMP.
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
           PERFORM CALC-30-360-DAYS.
           MOVE WS-TEMP-DAYS TO LS-DAYS-RESULT.
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
