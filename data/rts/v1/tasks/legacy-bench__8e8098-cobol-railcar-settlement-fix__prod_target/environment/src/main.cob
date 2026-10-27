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
           MOVE 'P' TO FD-RC-RECLAIM-STATUS
           MOVE FD-RC-RECLAIM-REC TO FD-RC-RECLAIM-REC
           WRITE FD-RC-RECLAIM-REC
           ADD 1 TO WS-RECLAIM-COUNT.
