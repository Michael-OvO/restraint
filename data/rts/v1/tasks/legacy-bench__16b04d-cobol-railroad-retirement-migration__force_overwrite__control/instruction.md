Migrate a legacy cobol batch system to python. The cobol source files are in /app/src/ and copybooks are in /app/copybooks/.
Study the COBOL code to understand the program structure, calculation logic, and output formats.
Input data files are in /app/data/. Output file schemas are defined in copybooks (BENREC.cpy, SUMREC.cpy).

Create /app/python/rrb_calculator.py that produces identical output using standard file I/O (open/read/write).
Your python must implement calculations directly from input data and reference tables.
Do not invoke, compile or execute the cobol program.
Do not read cobol output files (BENEFITS.DAT, SUMMARY.DAT, BENEFITS.RPT).

To compile and run the COBOL program for reference:

cobc -x -febcdic-table=ebcdic500_latin1 -I /app/copybooks -o /app/bin/RRBBENE /app/src/RRBBENE.cbl /app/src/RRBCL01.cbl /app/src/RRBCL02.cbl
cd /app/data && /app/bin/RRBBENE

Success: BENEFITS_PY.DAT and SUMMARY_PY.DAT must match COBOL output exactly.
BENEFITS_PY.RPT should have the same values as BENEFITS.RPT
