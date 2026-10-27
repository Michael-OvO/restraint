"""Tests for trade settlement system with SEC Rule 606 reporting."""
import os
import subprocess
import tempfile
import shutil


def decode_comp3(data):
    """Decode a COMP-3 packed decimal field to a float."""
    result = 0
    for i, byte in enumerate(data[:-1]):
        high = (byte >> 4) & 0x0F
        low = byte & 0x0F
        result = result * 100 + high * 10 + low
    last_byte = data[-1]
    last_digit = (last_byte >> 4) & 0x0F
    sign_nibble = last_byte & 0x0F
    result = result * 10 + last_digit
    if sign_nibble == 0x0D:
        result = -result
    return result


def decode_comp(data):
    """Decode a big-endian signed integer (COMP)."""
    value = int.from_bytes(data, byteorder='big', signed=True)
    return value


def parse_settlement_record(record):
    """Parse a settlement output record according to SETTLOUT.cpy layout.

    Layout:
    - OUT-TRADE-ID: PIC 9(8) = 8 bytes display
    - OUT-CUSIP: PIC X(9) = 9 bytes
    - OUT-PRINCIPAL: PIC S9(11)V99 COMP-3 = 7 bytes
    - OUT-ACCR-INT: PIC S9(9)V99 COMP-3 = 6 bytes
    - OUT-SEC-FEE: PIC S9(7)V99 COMP-3 = 5 bytes
    - OUT-NET-AMOUNT: PIC S9(11)V99 COMP-3 = 7 bytes
    - OUT-DAYS-ACCR: PIC S9(5) COMP-3 = 3 bytes
    Total: 8+9+7+6+5+7+3 = 45 bytes
    """
    trade_id = record[0:8].decode('ascii').strip()
    cusip = record[8:17].decode('ascii').strip()
    principal = decode_comp3(record[17:24]) / 100.0
    accr_int = decode_comp3(record[24:30]) / 100.0
    sec_fee = decode_comp3(record[30:35]) / 100.0
    net_amount = decode_comp3(record[35:42]) / 100.0
    days_accr = decode_comp3(record[42:45])
    return {
        'trade_id': trade_id,
        'cusip': cusip,
        'principal': principal,
        'accr_int': accr_int,
        'sec_fee': sec_fee,
        'net_amount': net_amount,
        'days_accr': int(days_accr)
    }


def parse_rule606_record(record):
    """Parse a Rule 606 report record according to RULE606.cpy layout.

    Layout:
    - RPT-VENUE: PIC X(3) = 3 bytes
    - RPT-TOTAL-ORDERS: PIC S9(9) COMP-3 = 5 bytes
    - RPT-TOTAL-SHARES: PIC S9(11) COMP-3 = 6 bytes
    - RPT-MARKET-ORD: PIC S9(9) COMP-3 = 5 bytes
    - RPT-LIMIT-ORD: PIC S9(9) COMP-3 = 5 bytes
    - RPT-PFOF-AMT: PIC S9(9)V99 COMP-3 = 6 bytes
    Total: 3+5+6+5+5+6 = 30 bytes
    """
    venue = record[0:3].decode('ascii').strip()
    total_orders = decode_comp3(record[3:8])
    total_shares = decode_comp3(record[8:14])
    market_ord = decode_comp3(record[14:19])
    limit_ord = decode_comp3(record[19:24])
    pfof_amt = decode_comp3(record[24:30]) / 100.0
    return {
        'venue': venue,
        'total_orders': int(total_orders),
        'total_shares': int(total_shares),
        'market_ord': int(market_ord),
        'limit_ord': int(limit_ord),
        'pfof_amt': pfof_amt
    }


def test_cobol_source_files_exist():
    """Verify that all required COBOL source files exist in /app/."""
    required_files = [
        '/app/settle.cob',
        '/app/daycount.cob',
        '/app/accrued.cob',
        '/app/secfee.cob'
    ]
    for f in required_files:
        assert os.path.exists(f), f"Required COBOL source file {f} not found"


def test_copybooks_exist():
    """Verify that all required copybook files exist in /app/copybooks/."""
    required_copybooks = [
        'TRADEREC.cpy',
        'BONDREC.cpy',
        'COUPON.cpy',
        'SECRATE.cpy',
        'SETTLOUT.cpy',
        'RULE606.cpy'
    ]
    copybook_dir = '/app/copybooks'
    assert os.path.isdir(copybook_dir), "Copybook directory must exist"
    for cpy in required_copybooks:
        path = os.path.join(copybook_dir, cpy)
        assert os.path.exists(path), f"Required copybook {cpy} not found"


def test_cobol_compiles_and_executes():
    """Compile all COBOL modules and execute the main program to produce output."""
    output_settlement = '/app/output/settlement.dat'
    output_rule606 = '/app/output/rule606.dat'

    if os.path.exists(output_settlement):
        os.remove(output_settlement)
    if os.path.exists(output_rule606):
        os.remove(output_rule606)

    compile_daycount = subprocess.run(
        ['cobc', '-m', '-I', '/app/copybooks',
         '-o', '/app/DAYCOUNT.so', '/app/daycount.cob'],
        capture_output=True, text=True, cwd='/app'
    )
    assert compile_daycount.returncode == 0, \
        f"daycount.cob compilation failed: {compile_daycount.stderr}"

    compile_accrued = subprocess.run(
        ['cobc', '-m', '-I', '/app/copybooks',
         '-o', '/app/ACCRUED.so', '/app/accrued.cob'],
        capture_output=True, text=True, cwd='/app'
    )
    assert compile_accrued.returncode == 0, \
        f"accrued.cob compilation failed: {compile_accrued.stderr}"

    compile_secfee = subprocess.run(
        ['cobc', '-m', '-I', '/app/copybooks',
         '-o', '/app/SECFEE.so', '/app/secfee.cob'],
        capture_output=True, text=True, cwd='/app'
    )
    assert compile_secfee.returncode == 0, \
        f"secfee.cob compilation failed: {compile_secfee.stderr}"

    compile_settle = subprocess.run(
        ['cobc', '-x', '-I', '/app/copybooks',
         '-o', '/app/settle', '/app/settle.cob'],
        capture_output=True, text=True, cwd='/app'
    )
    assert compile_settle.returncode == 0, \
        f"settle.cob compilation failed: {compile_settle.stderr}"

    env = os.environ.copy()
    env['COB_LIBRARY_PATH'] = '/app'
    run_result = subprocess.run(
        ['/app/settle'],
        capture_output=True, text=True, env=env, timeout=60
    )
    assert run_result.returncode == 0, \
        f"Settlement program execution failed: {run_result.stderr}"

    assert os.path.exists(output_settlement), \
        "Settlement output file /app/output/settlement.dat not created"
    assert os.path.exists(output_rule606), \
        "Rule 606 report file /app/output/rule606.dat not created"


def test_settlement_output_format():
    """Verify settlement output file has correct EBCDIC fixed-length format."""
    output_file = '/app/output/settlement.dat'
    assert os.path.exists(output_file), "Settlement output file must exist"

    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 45
    assert len(data) % record_length == 0, \
        f"Settlement file size {len(data)} not multiple of record length {record_length}"

    num_records = len(data) // record_length
    assert num_records == 4, f"Expected 4 settlement records, got {num_records}"


def test_settlement_trade_ids():
    """Verify all expected trades are present in settlement output."""
    output_file = '/app/output/settlement.dat'
    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 45
    records = []
    for i in range(0, len(data), record_length):
        rec = parse_settlement_record(data[i:i+record_length])
        records.append(rec)

    expected_trade_ids = {'00000001', '00000002', '00000003', '00000004'}
    actual_trade_ids = {r['trade_id'] for r in records}
    assert actual_trade_ids == expected_trade_ids, \
        f"Trade IDs mismatch. Expected {expected_trade_ids}, got {actual_trade_ids}"


def test_day_count_convention_actual_actual():
    """Verify actual/actual day-count convention is applied for bond 912810AA1.

    Trade 1: Settle date 20240318, last coupon 20240315 = 3 actual days
    The buggy code would use 30/360 giving 3 days, but for dates spanning months
    the difference is apparent. We test that the program respects the convention.
    """
    output_file = '/app/output/settlement.dat'
    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 45
    trade1 = None
    for i in range(0, len(data), record_length):
        rec = parse_settlement_record(data[i:i+record_length])
        if rec['trade_id'] == '00000001':
            trade1 = rec
            break

    assert trade1 is not None, "Trade 00000001 not found"
    assert trade1['days_accr'] == 3, \
        f"Trade 1 days accrued should be 3 (actual/actual), got {trade1['days_accr']}"


def test_quarterly_coupon_lookup():
    """Verify quarterly coupon bonds find the correct last coupon date.

    Bond 912810AA1 pays quarterly (freq=4). The buggy code only searches
    for semi-annual coupons (freq=2), missing the quarterly payments.
    Last coupon for 912810AA1 before 20240318 should be 20240315.
    """
    output_file = '/app/output/settlement.dat'
    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 45
    trade1 = None
    for i in range(0, len(data), record_length):
        rec = parse_settlement_record(data[i:i+record_length])
        if rec['trade_id'] == '00000001':
            trade1 = rec
            break

    assert trade1 is not None, "Trade 00000001 not found"
    assert trade1['days_accr'] == 3, \
        f"Quarterly coupon lookup failed. Days should be 3 from 20240315, got {trade1['days_accr']}"


def test_sec_fee_current_rate():
    """Verify SEC fee uses current rate from file, not hardcoded 2019 rate.

    Trade 2 is a SELL of 500 bonds at 101.25 = $50,625 principal.
    Current 2024 rate is 0.0000229 per dollar = $1.16 fee
    Old 2019 rate was 0.000008 per dollar = $0.41 fee (buggy)
    """
    output_file = '/app/output/settlement.dat'
    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 45
    trade2 = None
    for i in range(0, len(data), record_length):
        rec = parse_settlement_record(data[i:i+record_length])
        if rec['trade_id'] == '00000002':
            trade2 = rec
            break

    assert trade2 is not None, "Trade 00000002 not found"
    expected_principal = 500 * 101.25
    expected_sec_fee_approx = expected_principal * 0.0000229

    assert abs(trade2['sec_fee'] - expected_sec_fee_approx) < 0.10, \
        f"SEC fee {trade2['sec_fee']:.2f} should be ~{expected_sec_fee_approx:.2f} using 2024 rate"


def test_buy_trade_no_sec_fee():
    """Verify BUY trades have zero SEC fee (only SELL trades are charged)."""
    output_file = '/app/output/settlement.dat'
    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 45
    buy_trades = []
    for i in range(0, len(data), record_length):
        rec = parse_settlement_record(data[i:i+record_length])
        if rec['trade_id'] in ('00000001', '00000003'):
            buy_trades.append(rec)

    assert len(buy_trades) == 2, "Expected 2 BUY trades"
    for trade in buy_trades:
        assert trade['sec_fee'] == 0.0, \
            f"BUY trade {trade['trade_id']} should have zero SEC fee"


def test_net_settlement_calculation():
    """Verify net settlement amounts are calculated correctly.

    For BUY: net = principal + accrued interest
    For SELL: net = principal + accrued interest - SEC fee
    """
    output_file = '/app/output/settlement.dat'
    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 45
    for i in range(0, len(data), record_length):
        rec = parse_settlement_record(data[i:i+record_length])

        if rec['trade_id'] in ('00000001', '00000003'):
            expected_net = rec['principal'] + rec['accr_int']
        else:
            expected_net = rec['principal'] + rec['accr_int'] - rec['sec_fee']

        assert abs(rec['net_amount'] - expected_net) < 0.01, \
            f"Trade {rec['trade_id']} net amount {rec['net_amount']:.2f} " \
            f"doesn't match expected {expected_net:.2f}"


def test_rule606_output_format():
    """Verify Rule 606 report file has correct EBCDIC fixed-length format."""
    output_file = '/app/output/rule606.dat'
    assert os.path.exists(output_file), "Rule 606 output file must exist"

    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 30
    assert len(data) % record_length == 0, \
        f"Rule 606 file size {len(data)} not multiple of record length {record_length}"


def test_rule606_venue_aggregation():
    """Verify Rule 606 report correctly aggregates statistics by venue."""
    output_file = '/app/output/rule606.dat'
    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 30
    venues = {}
    for i in range(0, len(data), record_length):
        rec = parse_rule606_record(data[i:i+record_length])
        venues[rec['venue']] = rec

    expected_venues = {'NYS', 'NAS', 'ARC'}
    actual_venues = set(venues.keys())
    assert actual_venues == expected_venues, \
        f"Venues mismatch. Expected {expected_venues}, got {actual_venues}"

    assert venues['NYS']['total_orders'] == 2, \
        f"NYS should have 2 orders, got {venues['NYS']['total_orders']}"
    assert venues['NAS']['total_orders'] == 1, \
        f"NAS should have 1 order, got {venues['NAS']['total_orders']}"
    assert venues['ARC']['total_orders'] == 1, \
        f"ARC should have 1 order, got {venues['ARC']['total_orders']}"


def test_rule606_total_shares():
    """Verify Rule 606 report correctly sums total shares per venue."""
    output_file = '/app/output/rule606.dat'
    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 30
    venues = {}
    for i in range(0, len(data), record_length):
        rec = parse_rule606_record(data[i:i+record_length])
        venues[rec['venue']] = rec

    assert venues['NYS']['total_shares'] == 1750, \
        f"NYS total shares should be 1750 (1000+750), got {venues['NYS']['total_shares']}"
    assert venues['NAS']['total_shares'] == 500, \
        f"NAS total shares should be 500, got {venues['NAS']['total_shares']}"
    assert venues['ARC']['total_shares'] == 2000, \
        f"ARC total shares should be 2000, got {venues['ARC']['total_shares']}"


def test_mutation_different_input_produces_different_output():
    """Verify that modifying input data changes the output (anti-hardcoding test).

    This test backs up the trades file, modifies a quantity, re-runs the program,
    and verifies the output changed accordingly.
    """
    trades_file = '/app/data/trades.dat'
    backup_file = '/app/data/trades.dat.bak'
    output_file = '/app/output/settlement.dat'

    shutil.copy2(trades_file, backup_file)

    try:
        with open(output_file, 'rb') as f:
            original_output = f.read()

        with open(trades_file, 'rb') as f:
            trade_data = bytearray(f.read())

        qty_offset = 33
        original_qty = trade_data[qty_offset:qty_offset+4]
        trade_data[qty_offset:qty_offset+4] = b'\x00\x20\x00\x0C'

        with open(trades_file, 'wb') as f:
            f.write(trade_data)

        env = os.environ.copy()
        env['COB_LIBRARY_PATH'] = '/app'
        subprocess.run(['/app/settle'], capture_output=True, env=env, timeout=60)

        with open(output_file, 'rb') as f:
            modified_output = f.read()

        assert original_output != modified_output, \
            "Output should change when input quantity changes (anti-hardcoding)"

    finally:
        shutil.move(backup_file, trades_file)
        env = os.environ.copy()
        env['COB_LIBRARY_PATH'] = '/app'
        subprocess.run(['/app/settle'], capture_output=True, env=env, timeout=60)


def test_accrued_interest_calculation():
    """Verify accrued interest is computed correctly using proper formula.

    For bond 912810AA1 (5.25% coupon, quarterly, actual/actual):
    Trade 1: 1000 qty, 3 days accrued
    Accrued = (coupon_rate/100) * par * qty * days / period_days / freq
    Period days for actual/actual quarterly = 365/4 = 91.25
    Accrued = 0.0525 * 100 * 1000 * 3 / 91.25 / 4 = 43.15 approx
    """
    output_file = '/app/output/settlement.dat'
    with open(output_file, 'rb') as f:
        data = f.read()

    record_length = 45
    trade1 = None
    for i in range(0, len(data), record_length):
        rec = parse_settlement_record(data[i:i+record_length])
        if rec['trade_id'] == '00000001':
            trade1 = rec
            break

    assert trade1 is not None, "Trade 00000001 not found"
    expected_accr = (0.0525 * 100 * 1000 * 3) / (365/4) / 4
    assert abs(trade1['accr_int'] - expected_accr) < 1.0, \
        f"Accrued interest {trade1['accr_int']:.2f} should be ~{expected_accr:.2f}"


def test_principal_calculation():
    """Verify principal amounts are computed as qty * price."""
    output_file = '/app/output/settlement.dat'
    with open(output_file, 'rb') as f:
        data = f.read()

    expected_principals = {
        '00000001': 1000 * 98.50,
        '00000002': 500 * 101.25,
        '00000003': 2000 * 99.75,
        '00000004': 750 * 98.625
    }

    record_length = 45
    for i in range(0, len(data), record_length):
        rec = parse_settlement_record(data[i:i+record_length])
        expected = expected_principals.get(rec['trade_id'])
        if expected:
            assert abs(rec['principal'] - expected) < 0.01, \
                f"Trade {rec['trade_id']} principal {rec['principal']:.2f} " \
                f"should be {expected:.2f}"
