       01  EMPLOYEE-RECORD.
           05  EMP-SSN                     PIC X(9).
           05  EMP-NAME                    PIC X(30).
           05  EMP-DOB                     PIC 9(8).
           05  EMP-RET-DATE                PIC 9(8).
           05  EMP-SVC-YEARS               PIC 9(2)V9.
           05  EMP-EARNINGS-TABLE.
               10  EMP-EARN-ENTRY OCCURS 45 TIMES.
                   15  EMP-EARN-YEAR       PIC 9(4).
                   15  EMP-EARN-AMT        PIC 9(7)V99.
