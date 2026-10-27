      ******************************************************************
      * RECLAIM.CPY - RECLAIM REQUEST RECORD FOR DISPUTED CHARGES
      * GENERATED WHEN SETTLEMENT ERRORS ARE DETECTED
      ******************************************************************
       01  :PREFIX:-RECLAIM-REC.
           05  :PREFIX:-RECLAIM-ID          PIC 9(10).
           05  :PREFIX:-ORIG-CAR-ID         PIC X(10).
           05  :PREFIX:-RECLAIM-DATE        PIC 9(8).
           05  :PREFIX:-CLAIMING-RR         PIC X(4).
           05  :PREFIX:-AGAINST-RR          PIC X(4).
           05  :PREFIX:-DISPUTED-AMT        PIC S9(7)V99 COMP-3.
           05  :PREFIX:-REASON-CODE         PIC X(2).
           05  :PREFIX:-RECLAIM-STATUS      PIC X(1).
