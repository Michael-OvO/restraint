       IDENTIFICATION DIVISION.
       PROGRAM-ID. MILEAGE.

       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SPECIAL-NAMES.
           ALPHABET EBCDIC-CS IS EBCDIC.

       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT RATE-FILE ASSIGN TO "/app/data/rates.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-RATE-STATUS.

       DATA DIVISION.
       FILE SECTION.
       FD  RATE-FILE
           CODE-SET IS EBCDIC-CS.
       COPY RATEFIL REPLACING ==:PREFIX:== BY ==FD-RT==.

       WORKING-STORAGE SECTION.
       01  WS-RATE-STATUS           PIC XX.
       01  WS-EOF-FLAG              PIC X VALUE 'N'.
           88  WS-RATE-EOF          VALUE 'Y'.
       01  WS-HOUR-IDX              PIC 99.
       01  WS-RATE-IDX              PIC 99.
       01  WS-RATE-COUNT            PIC 99 VALUE 0.
       01  WS-RATE-LOADED           PIC X VALUE 'N'.
       01  WS-FOUND-FLAG            PIC X.
       01  WS-START-JUNCTION        PIC X(4).
       01  WS-END-JUNCTION          PIC X(4).
       01  WS-SEGMENT-MILES         PIC 9(4).
       01  WS-SEGMENT-RATE          PIC 9(3)V99.
       01  WS-SEGMENT-AMT           PIC S9(7)V99.
       01  WS-TOTAL-MILES           PIC 9(5) VALUE 0.
       01  WS-TOTAL-AMT             PIC S9(7)V99 VALUE 0.
       01  WS-JUNC-SPACES           PIC 99 VALUE 0.

       01  WS-RATE-TABLE.
           05  WS-RT-ENTRY OCCURS 50 TIMES
               INDEXED BY WS-RT-INDEX.
               10  WS-RT-FROM-JUNC  PIC X(4).
               10  WS-RT-TO-JUNC    PIC X(4).
               10  WS-RT-DISTANCE   PIC 9(4).
               10  WS-RT-RATE       PIC 9(3)V99.

       LINKAGE SECTION.
       COPY CARLOC REPLACING ==:PREFIX:== BY ==LS-CL==.

       01  LS-MILEAGE-DATA.
           05  LS-ML-AMOUNT         PIC S9(7)V99.
           05  LS-ML-MILES          PIC 9(5).
           05  LS-ML-STATUS         PIC X.

       PROCEDURE DIVISION USING LS-CL-CAR-LOC-REC LS-MILEAGE-DATA.
       MAIN-MILEAGE.
           IF WS-RATE-LOADED = 'N'
               PERFORM LOAD-RATE-TABLE
               MOVE 'Y' TO WS-RATE-LOADED
           END-IF

           MOVE 0 TO WS-TOTAL-MILES
           MOVE 0 TO WS-TOTAL-AMT
           MOVE 'S' TO LS-ML-STATUS

           MOVE LS-CL-HOUR-JUNCTION(1) TO WS-START-JUNCTION

           PERFORM VARYING WS-HOUR-IDX FROM 2 BY 1
               UNTIL WS-HOUR-IDX > 24
               PERFORM CALCULATE-SEGMENT-MILEAGE
           END-PERFORM

           MOVE WS-TOTAL-AMT TO LS-ML-AMOUNT
           MOVE WS-TOTAL-MILES TO LS-ML-MILES

           GOBACK.

       LOAD-RATE-TABLE.
           OPEN INPUT RATE-FILE
           MOVE 0 TO WS-RATE-COUNT
           MOVE 'N' TO WS-EOF-FLAG

           PERFORM UNTIL WS-RATE-EOF OR WS-RATE-COUNT >= 50
               READ RATE-FILE INTO FD-RT-RATE-ENTRY
                   AT END
                       SET WS-RATE-EOF TO TRUE
                   NOT AT END
                       ADD 1 TO WS-RATE-COUNT
                       MOVE FD-RT-FROM-JUNCTION TO
                           WS-RT-FROM-JUNC(WS-RATE-COUNT)
                       MOVE FD-RT-TO-JUNCTION TO
                           WS-RT-TO-JUNC(WS-RATE-COUNT)
                       MOVE FD-RT-DISTANCE-MILES TO
                           WS-RT-DISTANCE(WS-RATE-COUNT)
                       MOVE FD-RT-RATE-PER-MILE TO
                           WS-RT-RATE(WS-RATE-COUNT)
               END-READ
           END-PERFORM

           CLOSE RATE-FILE.

       CALCULATE-SEGMENT-MILEAGE.
           MOVE LS-CL-HOUR-JUNCTION(WS-HOUR-IDX) TO WS-END-JUNCTION

           IF WS-END-JUNCTION NOT = WS-START-JUNCTION
               PERFORM LOOKUP-RATE
               IF WS-FOUND-FLAG = 'Y'
                   ADD WS-SEGMENT-MILES TO WS-TOTAL-MILES
                   COMPUTE WS-SEGMENT-AMT =
                       WS-SEGMENT-MILES * WS-SEGMENT-RATE
                   ADD WS-SEGMENT-AMT TO WS-TOTAL-AMT
               ELSE
                   COMPUTE WS-SEGMENT-AMT = -1000
                   ADD WS-SEGMENT-AMT TO WS-TOTAL-AMT
               END-IF
               MOVE WS-END-JUNCTION TO WS-START-JUNCTION
           END-IF.

       LOOKUP-RATE.
           MOVE 'N' TO WS-FOUND-FLAG

           SEARCH WS-RT-ENTRY
               AT END
                   MOVE 'N' TO WS-FOUND-FLAG
               WHEN WS-RT-FROM-JUNC(WS-RT-INDEX) = WS-START-JUNCTION
                AND WS-RT-TO-JUNC(WS-RT-INDEX) = WS-END-JUNCTION
                   MOVE 'Y' TO WS-FOUND-FLAG
                   MOVE WS-RT-DISTANCE(WS-RT-INDEX) TO WS-SEGMENT-MILES
                   MOVE WS-RT-RATE(WS-RT-INDEX) TO WS-SEGMENT-RATE
           END-SEARCH

           IF WS-FOUND-FLAG = 'N'
               SEARCH WS-RT-ENTRY
                   AT END
                       MOVE 'N' TO WS-FOUND-FLAG
                   WHEN WS-RT-FROM-JUNC(WS-RT-INDEX) = WS-END-JUNCTION
                    AND WS-RT-TO-JUNC(WS-RT-INDEX) = WS-START-JUNCTION
                       MOVE 'Y' TO WS-FOUND-FLAG
                       MOVE WS-RT-DISTANCE(WS-RT-INDEX) TO
                           WS-SEGMENT-MILES
                       MOVE WS-RT-RATE(WS-RT-INDEX) TO WS-SEGMENT-RATE
               END-SEARCH
           END-IF.
