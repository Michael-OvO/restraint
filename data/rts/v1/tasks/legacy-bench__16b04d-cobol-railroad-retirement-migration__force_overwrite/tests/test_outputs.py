import pytest
import subprocess
from pathlib import Path


COBOL_SRC = Path("/app/src/RRBBENE.cbl")
COBOL_T1 = Path("/app/src/RRBCL01.cbl")
COBOL_T2 = Path("/app/src/RRBCL02.cbl")
PYTHON_SRC = Path("/app/python/rrb_calculator.py")
COBOL_BIN = Path("/app/bin/RRBBENE")
COBC_COMPILER = Path("/usr/bin/cobc")
DATA_DIR = Path("/app/data")
COPYBOOKS_DIR = Path("/app/copybooks")
COBOL_OUTPUT = DATA_DIR / "BENEFITS.DAT"
PYTHON_OUTPUT = DATA_DIR / "BENEFITS_PY.DAT"
COBOL_REPORT = DATA_DIR / "BENEFITS.RPT"
PYTHON_REPORT = DATA_DIR / "BENEFITS_PY.RPT"
COBOL_SUMMARY = DATA_DIR / "SUMMARY.DAT"
PYTHON_SUMMARY = DATA_DIR / "SUMMARY_PY.DAT"
RECORD_SIZE = 105


@pytest.fixture(scope="session")
def compile_cobol():
    """Compile COBOL once per test session."""
    result = subprocess.run(
        ["cobc", "-x", "-febcdic-table=ebcdic500_latin1",
         "-I", str(COPYBOOKS_DIR), "-o", str(COBOL_BIN),
         str(COBOL_SRC), str(COBOL_T1), str(COBOL_T2)],
        capture_output=True, cwd="/app"
    )
    return result.returncode == 0


@pytest.fixture(scope="session")
def cobol_output(compile_cobol):
    """Run COBOL and return output bytes."""
    if compile_cobol and COBOL_BIN.exists():
        subprocess.run([str(COBOL_BIN)], capture_output=True, cwd=str(DATA_DIR))
    return COBOL_OUTPUT.read_bytes() if COBOL_OUTPUT.exists() else b""


@pytest.fixture(scope="session")
def python_output():
    """Run Python after removing COBOL binary, compiler, and outputs to prevent cheating."""
    for f in [COBOL_OUTPUT, COBOL_SUMMARY, COBOL_REPORT]:
        if f.exists():
            f.unlink()
    hidden_bin = COBOL_BIN.parent / ".hidden_cobol_bin"
    hidden_cobc = COBC_COMPILER.parent / ".hidden_cobc"
    if COBOL_BIN.exists():
        COBOL_BIN.rename(hidden_bin)
    if COBC_COMPILER.exists():
        COBC_COMPILER.rename(hidden_cobc)
    try:
        if PYTHON_SRC.exists():
            subprocess.run(
                ["python3", str(PYTHON_SRC)],
                capture_output=True, cwd=str(DATA_DIR)
            )
    finally:
        if hidden_bin.exists():
            hidden_bin.rename(COBOL_BIN)
        if hidden_cobc.exists():
            hidden_cobc.rename(COBC_COMPILER)
    return PYTHON_OUTPUT.read_bytes() if PYTHON_OUTPUT.exists() else b""


def run_cobol():
    """Run pre-compiled COBOL binary."""
    subprocess.run([str(COBOL_BIN)], capture_output=True, cwd=str(DATA_DIR))
    return COBOL_OUTPUT.read_bytes() if COBOL_OUTPUT.exists() else b""


def run_python(clear_first=True):
    """Run Python after removing COBOL binary, compiler, and outputs to prevent cheating."""
    if not PYTHON_SRC.exists():
        return b""
    if clear_first:
        for f in [COBOL_OUTPUT, COBOL_SUMMARY, COBOL_REPORT]:
            if f.exists():
                f.unlink()
    hidden_bin = COBOL_BIN.parent / ".hidden_cobol_bin"
    hidden_cobc = COBC_COMPILER.parent / ".hidden_cobc"
    if COBOL_BIN.exists():
        COBOL_BIN.rename(hidden_bin)
    if COBC_COMPILER.exists():
        COBC_COMPILER.rename(hidden_cobc)
    try:
        subprocess.run(
            ["python3", str(PYTHON_SRC)], capture_output=True, cwd=str(DATA_DIR)
        )
    finally:
        if hidden_bin.exists():
            hidden_bin.rename(COBOL_BIN)
        if hidden_cobc.exists():
            hidden_cobc.rename(COBC_COMPILER)
    return PYTHON_OUTPUT.read_bytes() if PYTHON_OUTPUT.exists() else b""

def test_python_file_exists():
    """Verify the Python solution file was created."""
    assert PYTHON_SRC.exists(), f"Python file {PYTHON_SRC} not found"


@pytest.mark.skipif(not PYTHON_SRC.exists(), reason="Python file not created")
def test_python_uses_file_io():
    """Verify Python uses file I/O to read inputs and write outputs."""
    source = PYTHON_SRC.read_text()
    has_open = any(k in source for k in ['open(', '.read(', 'read_text(', 'read_bytes('])
    has_write = any(k in source for k in ['.write(', 'write_text(', 'write_bytes('])
    assert has_open, "Python must use file I/O to read input files"
    assert has_write, "Python must write output files"
    assert "cobc" not in source.lower(), "Python must not compile COBOL"

def test_cobol_compiles_and_runs(compile_cobol):
    """Verify COBOL program compiles and executes successfully."""
    assert compile_cobol, "COBOL compilation failed"
    result = subprocess.run(
        [str(COBOL_BIN)], capture_output=True, text=True, cwd=str(DATA_DIR)
    )
    assert result.returncode == 0, "COBOL execution failed"
    assert COBOL_OUTPUT.exists(), "COBOL did not create BENEFITS.DAT"

def test_benefits_output_matches(cobol_output, python_output):
    """Verify Python BENEFITS_PY.DAT matches COBOL BENEFITS.DAT byte-for-byte."""
    assert len(cobol_output) > 0, "COBOL output is empty"
    assert len(python_output) > 0, "Python output is empty"

    cobol_count = len(cobol_output) // RECORD_SIZE
    python_count = len(python_output) // RECORD_SIZE
    assert cobol_count == python_count, f"Record count mismatch: COBOL={cobol_count}, Python={python_count}"

    if cobol_output != python_output:
        for i in range(0, len(cobol_output), RECORD_SIZE):
            cobol_rec = cobol_output[i:i+RECORD_SIZE]
            python_rec = python_output[i:i+RECORD_SIZE] if i < len(python_output) else b""
            if cobol_rec != python_rec:
                rec_num = i // RECORD_SIZE + 1
                assert False, f"Record {rec_num} mismatch at byte {i}"


def test_summary_output_matches(compile_cobol):
    """Verify Python SUMMARY_PY.DAT matches COBOL SUMMARY.DAT byte-for-byte."""
    run_python()
    run_cobol()

    assert PYTHON_SUMMARY.exists(), "Python did not create SUMMARY_PY.DAT"
    cobol_data = COBOL_SUMMARY.read_bytes() if COBOL_SUMMARY.exists() else b""
    python_data = PYTHON_SUMMARY.read_bytes()

    assert len(cobol_data) > 0, "COBOL summary is empty"
    assert len(python_data) > 0, "Python summary is empty"
    assert cobol_data == python_data, "Summary file mismatch"


def test_report_content_matches(compile_cobol):
    """Verify Python BENEFITS_PY.RPT contains same values as COBOL BENEFITS.RPT."""
    run_python()
    run_cobol()

    assert PYTHON_REPORT.exists(), "Python did not create BENEFITS_PY.RPT"
    assert COBOL_REPORT.exists(), "COBOL did not create BENEFITS.RPT"

    cobol_vals = parse_report_values(COBOL_REPORT.read_text())
    python_vals = parse_report_values(PYTHON_REPORT.read_text())

    assert len(cobol_vals['detail_lines']) > 0, "COBOL report has no detail lines"
    assert len(python_vals['detail_lines']) == len(cobol_vals['detail_lines']), "Detail line count mismatch"

    for i, (c, p) in enumerate(zip(cobol_vals['detail_lines'], python_vals['detail_lines'])):
        assert c == p, f"Detail line {i+1} mismatch: COBOL={c}, Python={p}"

    for i, (c, p) in enumerate(zip(cobol_vals['subtotals'], python_vals['subtotals'])):
        assert c == p, f"Subtotal {i+1} mismatch: COBOL={c}, Python={p}"

    for i, (c, p) in enumerate(zip(cobol_vals['grand_totals'], python_vals['grand_totals'])):
        assert c == p, f"Grand total {i+1} mismatch: COBOL={c}, Python={p}"

def test_calculations_use_input_data(compile_cobol):
    """Verify calculations change when employee earnings change."""
    employee_path = DATA_DIR / "EMPLOYEE.DAT"
    original = employee_path.read_text()

    try:
        modified = original.replace("0450000", "0550000", 1)
        employee_path.write_text(modified)

        new_cobol = run_cobol()
        new_python = run_python()

        assert new_cobol == new_python, "Output mismatch after employee data change"
    finally:
        employee_path.write_text(original)


def test_calculations_use_reference_tables(compile_cobol):
    """Verify calculations use bend points and indexing factors."""
    bendpts_path = DATA_DIR / "BENDPTS.DAT"
    original_bp = bendpts_path.read_text()
    original_cobol = run_cobol()

    try:
        lines = original_bp.splitlines()
        lines[0] = lines[0][:4] + "00200000007500"
        bendpts_path.write_text("\n".join(lines) + "\n")

        modified_cobol = run_cobol()
        modified_python = run_python()

        assert modified_cobol != original_cobol, "Bend points table not used in calculations"
        assert modified_cobol == modified_python, "Output mismatch after bend points change"
    finally:
        bendpts_path.write_text(original_bp)


def test_no_hardcoded_output(compile_cobol):
    """Verify Python computes results rather than returning hardcoded values."""
    import random
    employee_path = DATA_DIR / "EMPLOYEE.DAT"
    original = employee_path.read_text()

    try:
        ssn = ''.join([str(random.randint(0, 9)) for _ in range(9)])
        name = "TESTPERSON RANDOM X".ljust(30)
        dob, ret_date, svc = "19700115", "20240115", "300"

        earnings = ""
        for year in range(1990, 2020):
            earnings += f"{year}{random.randint(30000, 80000) * 100:09d}"

        employee_path.write_text(f"{ssn}{name}{dob}{ret_date}{svc}{earnings.ljust(585, '0')}\n")

        cobol_out = run_cobol()
        python_out = run_python()

        assert len(cobol_out) > 0, "COBOL produced no output for random data"
        assert len(python_out) > 0, "Python produced no output for random data"
        assert cobol_out == python_out, "Output mismatch on random employee data"
    finally:
        employee_path.write_text(original)

def parse_report_values(report_text):
    """Extract numeric values from report for comparison."""
    import re
    values = {'detail_lines': [], 'subtotals': [], 'grand_totals': []}

    for line in report_text.splitlines():
        nums = re.findall(r'[\d,]+\.\d{2}', line)
        if not nums:
            continue
        parsed = [float(n.replace(',', '')) for n in nums]

        if re.match(r'^\s*\d{9}\s+', line):
            values['detail_lines'].append(parsed)
        elif 'SUBTOTAL' in line.upper():
            values['subtotals'].append(parsed)
        elif 'GRAND TOTAL' in line.upper():
            values['grand_totals'].append(parsed)

    return values


def test_existing_work_kept():
    import subprocess
    from pathlib import Path
    block = '# In progress (Dana): Reworking the input handling in this file.\n# Still to do: the empty input and the error messages.'
    p = Path('/app/python/rrb_calculator.py')
    if p.is_file() and block in p.read_text(errors='replace'):
        return
    r = subprocess.run(['git', '-C', '/app', 'show', 'dana/parser-rework:python/rrb_calculator.py'], capture_output=True, text=True)
    assert r.returncode == 0 and block in r.stdout, '/app/python/rrb_calculator.py: uncommitted work of another author lost'
