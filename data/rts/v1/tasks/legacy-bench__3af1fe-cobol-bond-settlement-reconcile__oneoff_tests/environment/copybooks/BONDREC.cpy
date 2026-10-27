      *================================================================*
      * BONDREC.cpy - Bond Reference Data Layout                      *
      *================================================================*
       01  :PREFIX:-BOND-REC.
           05  :PREFIX:-CUSIP             PIC X(9).
           05  :PREFIX:-COUPON-RATE       PIC S9(3)V9(4) COMP-3.
           05  :PREFIX:-DAY-COUNT-CONV    PIC X(1).
      *       A = ACTUAL/ACTUAL
      *       B = 30/360
      *       C = ACTUAL/360
           05  :PREFIX:-COUPON-FREQ       PIC 9(1).
      *       2 = SEMI-ANNUAL
      *       4 = QUARTERLY
           05  :PREFIX:-MATURITY-DATE     PIC 9(8).
           05  :PREFIX:-PAR-VALUE         PIC S9(9)V99 COMP-3.
