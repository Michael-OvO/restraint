#!/usr/bin/env python3
"""Generate test data files for trade settlement task."""
import struct
import os

def encode_ascii(s, length):
    """Encode string to ASCII, padded to length."""
    encoded = s.encode('ascii')
    if len(encoded) < length:
        encoded = encoded + b' ' * (length - len(encoded))
    return encoded[:length]

def encode_comp3(value, total_bytes, decimal_places=0):
    """Encode a number as COMP-3 (packed decimal)."""
    if decimal_places > 0:
        value = int(round(value * (10 ** decimal_places)))
    sign_nibble = 0x0C if value >= 0 else 0x0D
    value = abs(value)
    digits = str(value)
    num_digits = (total_bytes * 2) - 1
    digits = digits.zfill(num_digits)
    result = bytearray()
    for i in range(0, len(digits) - 1, 2):
        high = int(digits[i])
        low = int(digits[i + 1])
        result.append((high << 4) | low)
    last_digit = int(digits[-1])
    result.append((last_digit << 4) | sign_nibble)
    return bytes(result)

def encode_comp(value, num_bytes):
    """Encode as COMP (binary)."""
    if num_bytes == 2:
        return struct.pack('>h', value)
    elif num_bytes == 4:
        return struct.pack('>i', value)
    return struct.pack('>q', value)[:num_bytes]

os.makedirs('/app/data', exist_ok=True)

# Trade records (TRADEREC.cpy layout):
# TRADE-ID: PIC 9(8) = 8 bytes display
# BOND-CUSIP: PIC X(9) = 9 bytes
# TRADE-DATE: PIC 9(8) = 8 bytes display
# SETTLE-DATE: PIC 9(8) = 8 bytes display
# TRADE-QTY: PIC S9(7) COMP-3 = 4 bytes
# TRADE-PRICE: PIC S9(5)V9(4) COMP-3 = 5 bytes
# BUY-SELL-IND: PIC X(1) = 1 byte
# BROKER-ID: PIC X(4) = 4 bytes
# VENUE-CODE: PIC X(3) = 3 bytes
# Total: 8+9+8+8+4+5+1+4+3 = 50 bytes

trades = [
    # Trade 1: Bond with ACTUAL/ACTUAL day-count, quarterly coupon
    ("00000001", "912810AA1", "20240315", "20240318", 1000, 98.5000, "B", "GSCO", "NYS"),
    # Trade 2: Bond with 30/360 day-count, semi-annual coupon (no day-count bug)
    ("00000002", "912810BB2", "20240315", "20240318", 500, 101.2500, "S", "MSCO", "NAS"),
    # Trade 3: Bond with ACTUAL/360, quarterly coupon
    ("00000003", "912810CC3", "20240315", "20240318", 2000, 99.7500, "B", "JPMC", "ARC"),
    # Trade 4: Another ACTUAL/ACTUAL quarterly bond
    ("00000004", "912810AA1", "20240316", "20240319", 750, 98.6250, "S", "GSCO", "NYS"),
]

with open('/app/data/trades.dat', 'wb') as f:
    for trade in trades:
        record = b''
        record += encode_ascii(trade[0], 8)  # TRADE-ID
        record += encode_ascii(trade[1], 9)  # BOND-CUSIP
        record += encode_ascii(trade[2], 8)  # TRADE-DATE
        record += encode_ascii(trade[3], 8)  # SETTLE-DATE
        record += encode_comp3(trade[4], 4, 0)  # TRADE-QTY
        record += encode_comp3(trade[5], 5, 4)  # TRADE-PRICE
        record += encode_ascii(trade[6], 1)  # BUY-SELL-IND
        record += encode_ascii(trade[7], 4)  # BROKER-ID
        record += encode_ascii(trade[8], 3)  # VENUE-CODE
        f.write(record)

# Bond records (BONDREC.cpy layout):
# CUSIP: PIC X(9) = 9 bytes
# COUPON-RATE: PIC S9(3)V9(4) COMP-3 = 4 bytes
# DAY-COUNT-CONV: PIC X(1) = 1 byte (A=actual/actual, B=30/360, C=actual/360)
# COUPON-FREQ: PIC 9(1) = 1 byte (2=semi-annual, 4=quarterly)
# MATURITY-DATE: PIC 9(8) = 8 bytes
# PAR-VALUE: PIC S9(9)V99 COMP-3 = 6 bytes
# Total: 9+4+1+1+8+6 = 29 bytes

bonds = [
    # Bond 1: ACTUAL/ACTUAL, quarterly coupon - will expose all bugs
    ("912810AA1", 5.2500, "A", "4", "20340315", 100.00),
    # Bond 2: 30/360, semi-annual - standard bond
    ("912810BB2", 4.7500, "B", "2", "20290615", 100.00),
    # Bond 3: ACTUAL/360, quarterly
    ("912810CC3", 6.0000, "C", "4", "20310915", 100.00),
]

with open('/app/data/bonds.dat', 'wb') as f:
    for bond in bonds:
        record = b''
        record += encode_ascii(bond[0], 9)  # CUSIP
        record += encode_comp3(bond[1], 4, 4)  # COUPON-RATE
        record += encode_ascii(bond[2], 1)  # DAY-COUNT-CONV
        record += encode_ascii(bond[3], 1)  # COUPON-FREQ
        record += encode_ascii(bond[4], 8)  # MATURITY-DATE
        record += encode_comp3(bond[5], 6, 2)  # PAR-VALUE
        f.write(record)

# Coupon schedule (COUPON.cpy layout):
# CPN-CUSIP: PIC X(9) = 9 bytes
# CPN-DATE: PIC 9(8) = 8 bytes
# CPN-FREQ: PIC 9(1) = 1 byte
# Total: 9+8+1 = 18 bytes

coupons = [
    # Bond 912810AA1 - quarterly coupons (freq=4)
    ("912810AA1", "20231215", "4"),
    ("912810AA1", "20240315", "4"),
    ("912810AA1", "20240615", "4"),
    # Bond 912810BB2 - semi-annual coupons (freq=2)
    ("912810BB2", "20231215", "2"),
    ("912810BB2", "20240615", "2"),
    # Bond 912810CC3 - quarterly coupons (freq=4)
    ("912810CC3", "20231215", "4"),
    ("912810CC3", "20240315", "4"),
    ("912810CC3", "20240615", "4"),
]

with open('/app/data/coupons.dat', 'wb') as f:
    for coupon in coupons:
        record = b''
        record += encode_ascii(coupon[0], 9)  # CPN-CUSIP
        record += encode_ascii(coupon[1], 8)  # CPN-DATE
        record += encode_ascii(coupon[2], 1)  # CPN-FREQ
        f.write(record)

# SEC rate file (SECRATE.cpy layout):
# EFF-DATE: PIC 9(8) = 8 bytes
# RATE-PER-MIL: PIC S9(3)V9(6) COMP-3 = 5 bytes
# Total: 8+5 = 13 bytes

# Current SEC rate as of 2024 is approximately $22.90 per million
# That's 0.0000229 per dollar or 0.000022900000
sec_rates = [
    ("20190101", 0.000008),   # Old rate (what buggy code uses)
    ("20230101", 0.000020),   # 2023 rate
    ("20240101", 0.0000229),  # Current 2024 rate
]

with open('/app/data/secrates.dat', 'wb') as f:
    for rate in sec_rates:
        record = b''
        record += encode_ascii(rate[0], 8)  # EFF-DATE
        record += encode_comp3(rate[1], 5, 6)  # RATE-PER-MIL
        f.write(record)

print("Data files generated successfully")
