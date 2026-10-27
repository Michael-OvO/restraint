      *================================================================*
      * SETTLOUT.cpy - Settlement Output Record Layout                *
      *================================================================*
       01  :PREFIX:-SETTLE-OUT-REC.
           05  :PREFIX:-OUT-TRADE-ID      PIC 9(8).
           05  :PREFIX:-OUT-CUSIP         PIC X(9).
           05  :PREFIX:-OUT-PRINCIPAL     PIC S9(11)V99 COMP-3.
           05  :PREFIX:-OUT-ACCR-INT      PIC S9(9)V99 COMP-3.
           05  :PREFIX:-OUT-SEC-FEE       PIC S9(7)V99 COMP-3.
           05  :PREFIX:-OUT-NET-AMOUNT    PIC S9(11)V99 COMP-3.
           05  :PREFIX:-OUT-DAYS-ACCR     PIC S9(5) COMP-3.
