      ******************************************************************
      * RATEFIL.CPY - MILEAGE RATE TABLE ENTRY
      * CONTAINS JUNCTION PAIRS AND INTERLINE SETTLEMENT RATES
      ******************************************************************
       01  :PREFIX:-RATE-ENTRY.
           05  :PREFIX:-FROM-JUNCTION       PIC X(4).
           05  :PREFIX:-TO-JUNCTION         PIC X(4).
           05  :PREFIX:-DISTANCE-MILES      PIC 9(4).
           05  :PREFIX:-RATE-PER-MILE       PIC 9(3)V99 COMP-3.
           05  :PREFIX:-EFFECTIVE-DATE      PIC 9(8).
