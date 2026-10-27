      ******************************************************************
      * SETTLMT.CPY - SETTLEMENT OUTPUT RECORD
      * PER DIEM AND MILEAGE SETTLEMENT RESULTS
      ******************************************************************
       01  :PREFIX:-SETTLE-REC.
           05  :PREFIX:-CAR-ID              PIC X(10).
           05  :PREFIX:-SETTLE-DATE         PIC 9(8).
           05  :PREFIX:-OWNING-RR           PIC X(4).
           05  :PREFIX:-RESPONS-RR          PIC X(4).
           05  :PREFIX:-PER-DIEM-AMT        PIC S9(7)V99 COMP-3.
           05  :PREFIX:-MILEAGE-AMT         PIC S9(7)V99 COMP-3.
           05  :PREFIX:-TOTAL-AMT           PIC S9(7)V99 COMP-3.
           05  :PREFIX:-HOURS-CHARGED       PIC 9(2).
           05  :PREFIX:-MILES-CHARGED       PIC 9(5).
           05  :PREFIX:-SETTLE-STATUS       PIC X(1).
           05  :PREFIX:-ERROR-CODE          PIC X(3).
