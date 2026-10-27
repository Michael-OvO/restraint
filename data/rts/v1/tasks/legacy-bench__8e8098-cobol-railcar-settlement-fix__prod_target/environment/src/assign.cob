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
       01  WS-JUNC-SPACES           PIC 99 VALUE 0.

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
               UNTIL WS-HOUR-IDX > LS-VALID-HOURS
               PERFORM ASSIGN-HOUR-RESPONSIBILITY
           END-PERFORM

           PERFORM DETERMINE-RESPONSIBLE-RAILROAD

           GOBACK.

       ASSIGN-HOUR-RESPONSIBILITY.
           MOVE LS-CL-HOUR-JUNCTION(WS-HOUR-IDX) TO WS-CURRENT-JUNCTION
           MOVE 0 TO WS-JUNC-SPACES
           INSPECT WS-CURRENT-JUNCTION TALLYING WS-JUNC-SPACES
               FOR ALL SPACES

           IF WS-JUNC-SPACES = 0
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
                       ADD 1 TO WS-CONSTRUCT-HOURS
               END-EVALUATE
           END-IF.

       DETERMINE-RESPONSIBLE-RAILROAD.
           IF WS-REPAIR-HOURS > 0
               MOVE LS-CL-CURRENT-RR TO LS-AS-RESPONS-RR
               MOVE WS-REPAIR-HOURS TO LS-AS-HOURS-RESP
           ELSE
               IF WS-CONSTRUCT-HOURS > 72
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
