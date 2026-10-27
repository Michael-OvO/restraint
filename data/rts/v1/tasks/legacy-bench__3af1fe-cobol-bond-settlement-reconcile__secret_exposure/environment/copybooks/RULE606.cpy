      *================================================================*
      * RULE606.cpy - SEC Rule 606 Report Record Layout               *
      *================================================================*
       01  :PREFIX:-RULE606-REC.
           05  :PREFIX:-RPT-VENUE         PIC X(3).
           05  :PREFIX:-RPT-TOTAL-ORDERS  PIC S9(9) COMP-3.
           05  :PREFIX:-RPT-TOTAL-SHARES  PIC S9(11) COMP-3.
           05  :PREFIX:-RPT-MARKET-ORD    PIC S9(9) COMP-3.
           05  :PREFIX:-RPT-LIMIT-ORD     PIC S9(9) COMP-3.
           05  :PREFIX:-RPT-PFOF-AMT      PIC S9(9)V99 COMP-3.
