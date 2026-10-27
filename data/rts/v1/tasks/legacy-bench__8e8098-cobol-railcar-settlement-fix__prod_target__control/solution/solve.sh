#!/bin/bash
sh /app/bin/reset-db.sh
cat > /app/copybooks/CARLOC.cpy << 'ENDCOPY'
      ******************************************************************
      * CARLOC.CPY - CAR LOCATION RECORD WITH 24 HOURLY POSITIONS
      * USED FOR RAILROAD CAR HIRE SETTLEMENT PROCESSING
      ******************************************************************
       01  :PREFIX:-CAR-LOC-REC.
           05  :PREFIX:-CAR-ID              PIC X(10).
           05  :PREFIX:-REPORT-DATE         PIC 9(8).
           05  :PREFIX:-OWNING-RR           PIC X(4).
           05  :PREFIX:-CURRENT-RR          PIC X(4).
           05  :PREFIX:-CAR-TYPE            PIC X(2).
           05  :PREFIX:-LOAD-EMPTY          PIC X(1).
           05  :PREFIX:-HOURLY-POS.
               10  :PREFIX:-HOUR-DATA OCCURS 24 TIMES.
                   15  :PREFIX:-HOUR-STATUS     PIC X(1).
                   15  :PREFIX:-HOUR-JUNCTION   PIC X(4).
                   15  :PREFIX:-HOUR-MILES      PIC 9(4).
ENDCOPY
cat > /app/loadcar.cob << 'ENDCOBOL'
       IDENTIFICATION DIVISION.
       PROGRAM-ID. LOADCAR.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-HOUR-IDX              PIC 99.
       01  WS-VALID-COUNT           PIC 99 VALUE 0.
       01  WS-TOTAL-MILES           PIC 9(5) VALUE 0.
       01  WS-TEMP-MILES            PIC 9(4).
       01  WS-MILES-STR             PIC X(4).
       01  WS-NUMERIC-CHECK         PIC 9 VALUE 0.

       LINKAGE SECTION.
       COPY CARLOC REPLACING ==:PREFIX:== BY ==LS-CL==.

       01  LS-LOADED-DATA.
           05  LS-LD-VALID-FLAG     PIC X.
           05  LS-LD-VALID-HOURS    PIC 9(2).
           05  LS-LD-TOTAL-MILES    PIC 9(5).

       PROCEDURE DIVISION USING LS-CL-CAR-LOC-REC LS-LOADED-DATA.
       MAIN-LOAD.
           MOVE 0 TO WS-VALID-COUNT
           MOVE 0 TO WS-TOTAL-MILES
           MOVE 'Y' TO LS-LD-VALID-FLAG

           PERFORM VARYING WS-HOUR-IDX FROM 1 BY 1
               UNTIL WS-HOUR-IDX > 24
               PERFORM VALIDATE-HOUR-POSITION
           END-PERFORM

           MOVE WS-VALID-COUNT TO LS-LD-VALID-HOURS
           MOVE WS-TOTAL-MILES TO LS-LD-TOTAL-MILES

           IF WS-VALID-COUNT = 0
               MOVE 'N' TO LS-LD-VALID-FLAG
           END-IF

           GOBACK.

       VALIDATE-HOUR-POSITION.
           IF LS-CL-HOUR-STATUS(WS-HOUR-IDX) = 'A' OR
              LS-CL-HOUR-STATUS(WS-HOUR-IDX) = 'L' OR
              LS-CL-HOUR-STATUS(WS-HOUR-IDX) = 'E' OR
              LS-CL-HOUR-STATUS(WS-HOUR-IDX) = 'R'
               MOVE LS-CL-HOUR-MILES(WS-HOUR-IDX) TO WS-TEMP-MILES
               IF WS-TEMP-MILES IS NUMERIC
                   ADD 1 TO WS-VALID-COUNT
                   ADD WS-TEMP-MILES TO WS-TOTAL-MILES
               END-IF
           END-IF.
ENDCOBOL
cat > /app/assign.cob << 'ENDCOBOL'
       IDENTIFICATION DIVISION.
       PROGRAM-ID. ASSIGN.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-HOUR-IDX              PIC 99.
       01  WS-HOME-HOURS            PIC 99 VALUE 0.
       01  WS-FOREIGN-HOURS         PIC 99 VALUE 0.
       01  WS-REPAIR-HOURS          PIC 99 VALUE 0.
       01  WS-CONSTRUCT-HOURS       PIC 99 VALUE 0.
       01  WS-CURRENT-JUNCTION      PIC X(4).
       01  WS-SPACE-COUNT           PIC 99 VALUE 0.

       LINKAGE SECTION.
       COPY CARLOC REPLACING ==:PREFIX:== BY ==LS-CL==.

       01  LS-VALID-HOURS           PIC 9(2).

       01  LS-ASSIGN-DATA.
           05  LS-AS-RESPONS-RR     PIC X(4).
           05  LS-AS-HOURS-RESP     PIC 9(2).
           05  LS-AS-ASSIGN-STATUS  PIC X.

       PROCEDURE DIVISION USING LS-CL-CAR-LOC-REC
                                LS-VALID-HOURS
                                LS-ASSIGN-DATA.
       MAIN-ASSIGN.
           MOVE 0 TO WS-HOME-HOURS
           MOVE 0 TO WS-FOREIGN-HOURS
           MOVE 0 TO WS-REPAIR-HOURS
           MOVE 0 TO WS-CONSTRUCT-HOURS
           MOVE SPACES TO LS-AS-RESPONS-RR
           MOVE 0 TO LS-AS-HOURS-RESP
           MOVE 'S' TO LS-AS-ASSIGN-STATUS

           PERFORM VARYING WS-HOUR-IDX FROM 1 BY 1
               UNTIL WS-HOUR-IDX > 24
               PERFORM ASSIGN-HOUR-RESPONSIBILITY
           END-PERFORM

           PERFORM DETERMINE-RESPONSIBLE-RAILROAD

           GOBACK.

       ASSIGN-HOUR-RESPONSIBILITY.
           MOVE LS-CL-HOUR-JUNCTION(WS-HOUR-IDX) TO WS-CURRENT-JUNCTION
           MOVE 0 TO WS-SPACE-COUNT
           INSPECT WS-CURRENT-JUNCTION TALLYING WS-SPACE-COUNT
               FOR ALL SPACES

           IF WS-SPACE-COUNT < 4
               EVALUATE LS-CL-HOUR-STATUS(WS-HOUR-IDX)
                   WHEN 'A'
                       IF LS-CL-CURRENT-RR = LS-CL-OWNING-RR
                           ADD 1 TO WS-HOME-HOURS
                       ELSE
                           ADD 1 TO WS-FOREIGN-HOURS
                       END-IF
                   WHEN 'L'
                       ADD 1 TO WS-FOREIGN-HOURS
                   WHEN 'E'
                       ADD 1 TO WS-FOREIGN-HOURS
                   WHEN 'R'
                       ADD 1 TO WS-REPAIR-HOURS
                   WHEN OTHER
                       CONTINUE
               END-EVALUATE
           END-IF.

       DETERMINE-RESPONSIBLE-RAILROAD.
           IF WS-REPAIR-HOURS > 0
               MOVE LS-CL-CURRENT-RR TO LS-AS-RESPONS-RR
               MOVE WS-REPAIR-HOURS TO LS-AS-HOURS-RESP
           ELSE
               IF WS-CONSTRUCT-HOURS >= 72
                   MOVE LS-CL-CURRENT-RR TO LS-AS-RESPONS-RR
                   COMPUTE LS-AS-HOURS-RESP =
                       WS-CONSTRUCT-HOURS - 72
               ELSE
                   IF WS-FOREIGN-HOURS > 0
                       MOVE LS-CL-CURRENT-RR TO LS-AS-RESPONS-RR
                       MOVE WS-FOREIGN-HOURS TO LS-AS-HOURS-RESP
                   ELSE
                       MOVE LS-CL-OWNING-RR TO LS-AS-RESPONS-RR
                       MOVE WS-HOME-HOURS TO LS-AS-HOURS-RESP
                   END-IF
               END-IF
           END-IF

           IF LS-AS-RESPONS-RR = SPACES
               MOVE 'E' TO LS-AS-ASSIGN-STATUS
           END-IF.
ENDCOBOL
cat > /app/mileage.cob << 'ENDCOBOL'
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
       01  WS-SPACE-COUNT           PIC 99 VALUE 0.
       01  WS-VALID-START           PIC X VALUE 'N'.
       01  WS-VALID-END             PIC X VALUE 'N'.

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
               PERFORM VALIDATE-JUNCTIONS
               IF WS-VALID-START = 'Y' AND WS-VALID-END = 'Y'
                   PERFORM LOOKUP-RATE
                   IF WS-FOUND-FLAG = 'Y'
                       ADD WS-SEGMENT-MILES TO WS-TOTAL-MILES
                       COMPUTE WS-SEGMENT-AMT =
                           WS-SEGMENT-MILES * WS-SEGMENT-RATE
                       ADD WS-SEGMENT-AMT TO WS-TOTAL-AMT
                   END-IF
               END-IF
               MOVE WS-END-JUNCTION TO WS-START-JUNCTION
           END-IF.

       VALIDATE-JUNCTIONS.
           MOVE 'N' TO WS-VALID-START
           MOVE 'N' TO WS-VALID-END

           MOVE 0 TO WS-SPACE-COUNT
           INSPECT WS-START-JUNCTION TALLYING WS-SPACE-COUNT
               FOR ALL SPACES
           IF WS-SPACE-COUNT < 4
               MOVE 'Y' TO WS-VALID-START
           END-IF

           MOVE 0 TO WS-SPACE-COUNT
           INSPECT WS-END-JUNCTION TALLYING WS-SPACE-COUNT
               FOR ALL SPACES
           IF WS-SPACE-COUNT < 4
               MOVE 'Y' TO WS-VALID-END
           END-IF.

       LOOKUP-RATE.
           MOVE 'N' TO WS-FOUND-FLAG
           SET WS-RT-INDEX TO 1

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
               SET WS-RT-INDEX TO 1
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
ENDCOBOL
cat > /app/reclaim.cob << 'ENDCOBOL'
       IDENTIFICATION DIVISION.
       PROGRAM-ID. RECLAIM.

       DATA DIVISION.
       WORKING-STORAGE SECTION.
       01  WS-MIN-THRESHOLD         PIC S9(7)V99 VALUE 10.00.

       LINKAGE SECTION.
       COPY RECLAIM REPLACING ==:PREFIX:== BY ==LS-RC==.

       PROCEDURE DIVISION USING LS-RC-RECLAIM-REC.
       MAIN-RECLAIM.
           IF LS-RC-DISPUTED-AMT < WS-MIN-THRESHOLD
               MOVE 'R' TO LS-RC-RECLAIM-STATUS
           ELSE
               MOVE 'P' TO LS-RC-RECLAIM-STATUS
           END-IF

           GOBACK.
ENDCOBOL
cat > /app/main.cob << 'ENDCOBOL'
       IDENTIFICATION DIVISION.
       PROGRAM-ID. CARHIRE.

       ENVIRONMENT DIVISION.
       CONFIGURATION SECTION.
       SPECIAL-NAMES.
           ALPHABET EBCDIC-CS IS EBCDIC.

       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CAR-LOC-FILE ASSIGN TO "/app/data/carloc.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-CAR-STATUS.
           SELECT SETTLE-FILE ASSIGN TO "/app/output/settle.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-SETTLE-STATUS.
           SELECT ERROR-FILE ASSIGN TO "/app/output/errors.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-ERROR-STATUS.
           SELECT RECLAIM-FILE ASSIGN TO "/app/output/reclaim.dat"
               ORGANIZATION IS SEQUENTIAL
               FILE STATUS IS WS-RECLAIM-STATUS.

       DATA DIVISION.
       FILE SECTION.
       FD  CAR-LOC-FILE
           CODE-SET IS EBCDIC-CS.
       COPY CARLOC REPLACING ==:PREFIX:== BY ==FD-CL==.

       FD  SETTLE-FILE
           CODE-SET IS EBCDIC-CS.
       COPY SETTLMT REPLACING ==:PREFIX:== BY ==FD-ST==.

       FD  ERROR-FILE
           CODE-SET IS EBCDIC-CS.
       COPY SETTLMT REPLACING ==:PREFIX:== BY ==FD-ER==.

       FD  RECLAIM-FILE
           CODE-SET IS EBCDIC-CS.
       COPY RECLAIM REPLACING ==:PREFIX:== BY ==FD-RC==.

       WORKING-STORAGE SECTION.
       01  WS-CAR-STATUS            PIC XX.
       01  WS-SETTLE-STATUS         PIC XX.
       01  WS-ERROR-STATUS          PIC XX.
       01  WS-RECLAIM-STATUS        PIC XX.
       01  WS-EOF-FLAG              PIC X VALUE 'N'.
           88  WS-EOF               VALUE 'Y'.
       01  WS-RECORD-COUNT          PIC 9(6) VALUE 0.
       01  WS-SETTLE-COUNT          PIC 9(6) VALUE 0.
       01  WS-ERROR-COUNT           PIC 9(6) VALUE 0.
       01  WS-RECLAIM-COUNT         PIC 9(6) VALUE 0.
       01  WS-RECLAIM-SEQ           PIC 9(10) VALUE 0.

       01  WS-LOADED-DATA.
           05  WS-LD-VALID-FLAG     PIC X.
           05  WS-LD-VALID-HOURS    PIC 9(2).
           05  WS-LD-TOTAL-MILES    PIC 9(5).

       01  WS-ASSIGN-DATA.
           05  WS-AS-RESPONS-RR     PIC X(4).
           05  WS-AS-HOURS-RESP     PIC 9(2).
           05  WS-AS-ASSIGN-STATUS  PIC X.

       01  WS-MILEAGE-DATA.
           05  WS-ML-AMOUNT         PIC S9(7)V99.
           05  WS-ML-MILES          PIC 9(5).
           05  WS-ML-STATUS         PIC X.

       01  WS-PER-DIEM-RATE         PIC 9(3)V99 VALUE 45.00.

       PROCEDURE DIVISION.
       MAIN-PROCESS.
           OPEN INPUT CAR-LOC-FILE
           OPEN OUTPUT SETTLE-FILE
           OPEN OUTPUT ERROR-FILE
           OPEN OUTPUT RECLAIM-FILE

           PERFORM UNTIL WS-EOF
               READ CAR-LOC-FILE
                   AT END
                       SET WS-EOF TO TRUE
                   NOT AT END
                       ADD 1 TO WS-RECORD-COUNT
                       PERFORM PROCESS-CAR-RECORD
               END-READ
           END-PERFORM

           CLOSE CAR-LOC-FILE
           CLOSE SETTLE-FILE
           CLOSE ERROR-FILE
           CLOSE RECLAIM-FILE
           STOP RUN.

       PROCESS-CAR-RECORD.
           INITIALIZE FD-ST-SETTLE-REC
           INITIALIZE WS-LOADED-DATA
           INITIALIZE WS-ASSIGN-DATA
           INITIALIZE WS-MILEAGE-DATA

           CALL "LOADCAR" USING FD-CL-CAR-LOC-REC WS-LOADED-DATA

           IF WS-LD-VALID-FLAG = 'N'
               PERFORM WRITE-ERROR-RECORD
           ELSE
               CALL "ASSIGN" USING FD-CL-CAR-LOC-REC
                                   WS-LD-VALID-HOURS
                                   WS-ASSIGN-DATA

               IF WS-AS-ASSIGN-STATUS = 'E'
                   PERFORM WRITE-ERROR-RECORD
               ELSE
                   CALL "MILEAGE" USING FD-CL-CAR-LOC-REC
                                        WS-MILEAGE-DATA

                   PERFORM CALCULATE-SETTLEMENT

                   IF FD-ST-SETTLE-STATUS = 'E'
                       PERFORM WRITE-ERROR-RECORD
                       PERFORM GENERATE-RECLAIM
                   ELSE
                       PERFORM WRITE-SETTLE-RECORD
                   END-IF
               END-IF
           END-IF.

       CALCULATE-SETTLEMENT.
           MOVE FD-CL-CAR-ID TO FD-ST-CAR-ID
           MOVE FD-CL-REPORT-DATE TO FD-ST-SETTLE-DATE
           MOVE FD-CL-OWNING-RR TO FD-ST-OWNING-RR
           MOVE WS-AS-RESPONS-RR TO FD-ST-RESPONS-RR
           MOVE WS-AS-HOURS-RESP TO FD-ST-HOURS-CHARGED
           MOVE WS-ML-MILES TO FD-ST-MILES-CHARGED

           COMPUTE FD-ST-PER-DIEM-AMT =
               WS-AS-HOURS-RESP * WS-PER-DIEM-RATE / 24

           MOVE WS-ML-AMOUNT TO FD-ST-MILEAGE-AMT

           COMPUTE FD-ST-TOTAL-AMT =
               FD-ST-PER-DIEM-AMT + FD-ST-MILEAGE-AMT

           IF FD-ST-TOTAL-AMT < 0
               MOVE 'E' TO FD-ST-SETTLE-STATUS
               MOVE 'NEG' TO FD-ST-ERROR-CODE
           ELSE
               MOVE 'S' TO FD-ST-SETTLE-STATUS
               MOVE SPACES TO FD-ST-ERROR-CODE
           END-IF.

       WRITE-SETTLE-RECORD.
           MOVE FD-ST-SETTLE-REC TO FD-ST-SETTLE-REC
           WRITE FD-ST-SETTLE-REC
           ADD 1 TO WS-SETTLE-COUNT.

       WRITE-ERROR-RECORD.
           IF FD-ST-ERROR-CODE = SPACES
               MOVE 'VAL' TO FD-ST-ERROR-CODE
           END-IF
           MOVE 'E' TO FD-ST-SETTLE-STATUS
           MOVE FD-CL-CAR-ID TO FD-ST-CAR-ID
           MOVE FD-CL-REPORT-DATE TO FD-ST-SETTLE-DATE
           MOVE FD-CL-OWNING-RR TO FD-ST-OWNING-RR
           MOVE FD-ST-SETTLE-REC TO FD-ER-SETTLE-REC
           WRITE FD-ER-SETTLE-REC
           ADD 1 TO WS-ERROR-COUNT.

       GENERATE-RECLAIM.
           ADD 1 TO WS-RECLAIM-SEQ
           MOVE WS-RECLAIM-SEQ TO FD-RC-RECLAIM-ID
           MOVE FD-CL-CAR-ID TO FD-RC-ORIG-CAR-ID
           MOVE FD-CL-REPORT-DATE TO FD-RC-RECLAIM-DATE
           MOVE FD-CL-OWNING-RR TO FD-RC-CLAIMING-RR
           MOVE WS-AS-RESPONS-RR TO FD-RC-AGAINST-RR
           MOVE FD-ST-TOTAL-AMT TO FD-RC-DISPUTED-AMT
           MOVE 'NG' TO FD-RC-REASON-CODE
           CALL "RECLAIM" USING FD-RC-RECLAIM-REC
           MOVE FD-RC-RECLAIM-REC TO FD-RC-RECLAIM-REC
           WRITE FD-RC-RECLAIM-REC
           ADD 1 TO WS-RECLAIM-COUNT.
ENDCOBOL
cobc -m -febcdic-table=ebcdic500_latin1 -I /app/copybooks -o /app/LOADCAR.so /app/loadcar.cob
cobc -m -febcdic-table=ebcdic500_latin1 -I /app/copybooks -o /app/ASSIGN.so /app/assign.cob
cobc -m -febcdic-table=ebcdic500_latin1 -I /app/copybooks -o /app/MILEAGE.so /app/mileage.cob
cobc -m -febcdic-table=ebcdic500_latin1 -I /app/copybooks -o /app/RECLAIM.so /app/reclaim.cob
cobc -x -febcdic-table=ebcdic500_latin1 -I /app/copybooks -o /app/main /app/main.cob
COB_LIBRARY_PATH=/app /app/main
