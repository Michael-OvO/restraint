#!/usr/bin/env python3
"""Generate EBCDIC test data for Railroad Car Hire Settlement task."""

import struct

def encode_ebcdic(text, length):
    """Encode text to EBCDIC, padded to specified length."""
    padded = text.ljust(length)[:length]
    return padded.encode('cp500')

def encode_comp3(value, num_bytes):
    """Encode a decimal value to COMP-3 packed decimal format."""
    if value < 0:
        sign_nibble = 0x0D
        value = abs(value)
    else:
        sign_nibble = 0x0C

    int_value = int(value * 100)
    digits = str(int_value).zfill(num_bytes * 2 - 1)

    result = bytearray()
    for i in range(0, len(digits) - 1, 2):
        byte = (int(digits[i]) << 4) | int(digits[i + 1])
        result.append(byte)

    last_digit = int(digits[-1])
    last_byte = (last_digit << 4) | sign_nibble
    result.append(last_byte)

    return bytes(result)

def create_car_location_record(car_id, report_date, owning_rr, current_rr,
                                car_type, load_empty, hourly_data):
    """Create a car location record with 24 hourly positions."""
    record = bytearray()

    record.extend(encode_ebcdic(car_id, 10))
    record.extend(encode_ebcdic(report_date, 8))
    record.extend(encode_ebcdic(owning_rr, 4))
    record.extend(encode_ebcdic(current_rr, 4))
    record.extend(encode_ebcdic(car_type, 2))
    record.extend(encode_ebcdic(load_empty, 1))

    for hour in range(24):
        if hour < len(hourly_data):
            status, junction, miles = hourly_data[hour]
        else:
            status, junction, miles = ' ', '    ', 0

        record.extend(encode_ebcdic(status, 1))
        record.extend(encode_ebcdic(junction, 4))
        record.extend(encode_ebcdic(str(miles).zfill(4), 4))

    return bytes(record)

def create_rate_record(from_junc, to_junc, distance, rate, eff_date):
    """Create a rate table record."""
    record = bytearray()

    record.extend(encode_ebcdic(from_junc, 4))
    record.extend(encode_ebcdic(to_junc, 4))
    record.extend(encode_ebcdic(str(distance).zfill(4), 4))
    record.extend(encode_comp3(rate, 3))
    record.extend(encode_ebcdic(eff_date, 8))

    return bytes(record)

car_records = [
    ('BNSF001234', '20260115', 'BNSF', 'UP  ', 'BX', 'L',
     [('A', 'CHIC', 0)] * 10 + [('L', 'KANS', 250)] * 14),

    ('UP00056789', '20260115', 'UP  ', 'BNSF', 'TK', 'E',
     [('E', 'DENV', 0)] * 8 + [('A', 'SLAK', 150)] * 16),

    ('CSX0012345', '20260115', 'CSX ', 'NS  ', 'GP', 'L',
     [('R', 'ATLA', 0)] * 24),

    ('NS00098765', '20260115', 'NS  ', 'CSX ', 'BX', 'L',
     [('A', 'CLEV', 0)] * 18 + [('L', 'PITT', 180)] * 6),

    ('BNSF005678', '20260115', 'BNSF', 'UP  ', 'TK', 'L',
     [('A', 'CHIC', 0)] * 6 + [('L', 'OMAH', 300)] * 6 +
     [('L', 'DENV', 200)] * 6 + [('A', 'SLAK', 150)] * 6),

    ('UP00067890', '20260115', 'UP  ', 'CSX ', 'BX', 'E',
     [('L', 'JACK', 120)] * 12 + [('A', 'ATLA', 90)] * 12),

    ('CSX0023456', '20260115', 'CSX ', 'BNSF', 'GP', 'L',
     [('A', 'PITT', 0)] * 4 + [('L', 'CLEV', 135)] * 10 +
     [('E', 'CHIC', 0)] * 10),
]

rate_records = [
    ('CHIC', 'KANS', 550, 0.85, '20260101'),
    ('CHIC', 'OMAH', 470, 0.90, '20260101'),
    ('OMAH', 'DENV', 540, 0.88, '20260101'),
    ('DENV', 'SLAK', 520, 0.82, '20260101'),
    ('CHIC', 'DENV', 1010, 0.75, '20260101'),
    ('DENV', 'KANS', 600, 0.80, '20260101'),
    ('ATLA', 'CLEV', 720, 0.78, '20260101'),
    ('CLEV', 'PITT', 135, 0.95, '20260101'),
    ('SLAK', 'DENV', 520, 0.82, '20260101'),
    ('ATLA', 'PITT', 680, 0.77, '20260101'),
    ('PITT', 'CLEV', 135, 0.95, '20260101'),
    ('CLEV', 'CHIC', 345, 0.87, '20260101'),
]

with open('/app/data/carloc.dat', 'wb') as f:
    for car in car_records:
        record = create_car_location_record(*car)
        f.write(record)

with open('/app/data/rates.dat', 'wb') as f:
    for rate in rate_records:
        record = create_rate_record(*rate)
        f.write(record)
