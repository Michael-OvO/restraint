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
