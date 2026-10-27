      *================================================================*
      * TRADEREC.cpy - Trade Record Layout with REPLACING pattern     *
      *================================================================*
       01  :PREFIX:-TRADE-REC.
           05  :PREFIX:-TRADE-ID           PIC 9(8).
           05  :PREFIX:-BOND-CUSIP         PIC X(9).
           05  :PREFIX:-TRADE-DATE         PIC 9(8).
           05  :PREFIX:-SETTLE-DATE        PIC 9(8).
           05  :PREFIX:-TRADE-QTY          PIC S9(7) COMP-3.
           05  :PREFIX:-TRADE-PRICE        PIC S9(5)V9(4) COMP-3.
           05  :PREFIX:-BUY-SELL-IND       PIC X(1).
           05  :PREFIX:-BROKER-ID          PIC X(4).
           05  :PREFIX:-VENUE-CODE         PIC X(3).
