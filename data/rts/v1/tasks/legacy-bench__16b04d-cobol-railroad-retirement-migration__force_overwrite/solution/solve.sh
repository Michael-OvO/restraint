#!/bin/bash

mkdir -p /app/python

cat > /app/python/rrb_calculator.py << 'EOF'
import struct
from decimal import Decimal, ROUND_DOWN, ROUND_FLOOR, ROUND_HALF_UP
from pathlib import Path

DATA_DIR = Path("/app/data")

LINES_PER_PAGE = 55
REPORT_WIDTH = 132

class ControlBreakState:
    def __init__(self):
        self.prev_ret_year = 0
        self.curr_ret_year = 0
        self.first_record = True
        self.year_count = 0
        self.year_t1_total = Decimal("0")
        self.year_t2_total = Decimal("0")
        self.year_tt_total = Decimal("0")

    def reset_year_accumulators(self):
        self.year_count = 0
        self.year_t1_total = Decimal("0")
        self.year_t2_total = Decimal("0")
        self.year_tt_total = Decimal("0")

    def accumulate(self, t1, t2, total):
        self.year_count += 1
        self.year_t1_total += t1
        self.year_t2_total += t2
        self.year_tt_total += total

class ReportState:
    def __init__(self):
        self.page_num = 0
        self.line_num = 99
        self.rec_count = 0
        self.lines = []

    def need_new_page(self):
        return self.line_num >= LINES_PER_PAGE

class SummaryStats:
    def __init__(self):
        self.total_recs = 0
        self.valid_count = 0
        self.error_count = 0
        self.grand_t1 = Decimal("0")
        self.grand_t2 = Decimal("0")
        self.grand_tt = Decimal("0")
        self.t1_min = Decimal("999999")
        self.t1_max = Decimal("0")
        self.t2_min = Decimal("999999")
        self.t2_max = Decimal("0")

    def update_valid(self, t1, t2, total):
        self.valid_count += 1
        self.grand_t1 += t1
        self.grand_t2 += t2
        self.grand_tt += total
        if t1 < self.t1_min:
            self.t1_min = t1
        if t1 > self.t1_max:
            self.t1_max = t1
        if t2 < self.t2_min:
            self.t2_min = t2
        if t2 > self.t2_max:
            self.t2_max = t2

    def update_error(self):
        self.error_count += 1

def trunc(value, decimals):
    if decimals == 0:
        return Decimal(int(value))
    factor = Decimal(10) ** decimals
    return Decimal(int(value * factor)) / factor

def pack_comp3(value, total_digits, decimal_places):
    sign = 0x0C if value >= 0 else 0x0D
    scaled = int(abs(value) * (10 ** decimal_places))
    num_str = str(scaled).zfill(total_digits)

    packed = []
    for i in range(0, len(num_str) - 1, 2):
        high = int(num_str[i])
        low = int(num_str[i + 1])
        packed.append((high << 4) | low)

    last_digit = int(num_str[-1])
    packed.append((last_digit << 4) | sign)

    return bytes(packed)

def load_bendpoints():
    bendpoints = {}
    bp_file = DATA_DIR / "BENDPTS.DAT"
    for line in bp_file.read_text().splitlines():
        if line.strip():
            year = int(line[0:4])
            first = Decimal(line[4:11]) / 100
            second = Decimal(line[11:18]) / 100
            bendpoints[year] = (first, second)
    return bendpoints

def load_indexing_factors():
    factors = {}
    idx_file = DATA_DIR / "IDXFACT.DAT"
    for line in idx_file.read_text().splitlines():
        if line.strip():
            year = int(line[0:4])
            factor = Decimal(line[13:20]) / Decimal("1000000")
            factors[year] = factor
    return factors

def load_tier2_maximums():
    maximums = {}
    t2m_file = DATA_DIR / "TIER2MAX.DAT"
    for line in t2m_file.read_text().splitlines():
        if line.strip():
            year = int(line[0:4])
            maximum = Decimal(line[4:13]) / 100
            maximums[year] = maximum
    return maximums

def parse_employee_record(line):
    ssn = line[0:9]
    name = line[9:39]
    dob = line[39:47]
    ret_date = line[47:55]
    svc_years_str = line[55:58]
    svc_years = Decimal(svc_years_str[0:2]) + Decimal(svc_years_str[2]) / 10

    earnings = []
    pos = 58
    for i in range(45):
        if pos + 13 <= len(line):
            earn_year_str = line[pos:pos+4]
            earn_amt_str = line[pos+4:pos+13]
            if earn_year_str.strip() and earn_year_str != "0000":
                earn_year = int(earn_year_str)
                try:
                    earn_amt = Decimal(earn_amt_str.strip() or "0") / 100
                except:
                    earn_amt = Decimal("0")
                if earn_year > 0:
                    earnings.append((earn_year, earn_amt))
            pos += 13
        else:
            break

    return {
        "ssn": ssn,
        "name": name,
        "dob": dob,
        "ret_date": ret_date,
        "svc_years": svc_years,
        "earnings": earnings
    }

def calc_tier1(emp, bendpoints, idx_factors):
    dob_year = int(emp["dob"][0:4])
    dob_month = int(emp["dob"][4:6])
    ret_year = int(emp["ret_date"][0:4])
    ret_month = int(emp["ret_date"][4:6])

    elig_year = dob_year + 62
    age_at_ret = trunc(Decimal(ret_year - dob_year) + Decimal(ret_month - dob_month) / 12, 2)

    indexed_earnings = []
    for earn_year, earn_amt in emp["earnings"]:
        factor = idx_factors.get(earn_year, Decimal("1.000000"))
        indexed = trunc(earn_amt * factor, 2)
        indexed_earnings.append(indexed)

    indexed_earnings.sort(reverse=True)

    top_35 = indexed_earnings[:35]
    while len(top_35) < 35:
        top_35.append(Decimal("0"))

    total = sum(top_35)
    aime_work = trunc(total / 420, 2)
    aime = int(aime_work)

    bp1, bp2 = bendpoints.get(elig_year, (Decimal("1174.00"), Decimal("7078.00")))

    if aime <= bp1:
        pia_work = trunc(Decimal(aime) * Decimal("0.90"), 4)
    elif aime <= bp2:
        pia_work = trunc((bp1 * Decimal("0.90")) + ((Decimal(aime) - bp1) * Decimal("0.32")), 4)
    else:
        pia_work = trunc((bp1 * Decimal("0.90")) + ((bp2 - bp1) * Decimal("0.32")) + \
                   ((Decimal(aime) - bp2) * Decimal("0.15")), 4)

    pia = trunc(Decimal(int(pia_work * 10)) / Decimal("10"), 2)

    tier1_amt = pia

    fra = 66
    if emp["svc_years"] >= 30:
        pass
    else:
        if age_at_ret < fra:
            months_early = int((fra - age_at_ret) * 12)
            if months_early > 36:
                age_red_factor = trunc(Decimal("1") - (Decimal("36") * 5 / 900) - \
                                 ((Decimal(months_early) - 36) * 5 / 1200), 6)
            else:
                age_red_factor = trunc(Decimal("1") - (Decimal(months_early) * 5 / 900), 6)
            tier1_amt = trunc(tier1_amt * age_red_factor, 2)

    tier1_final = trunc(Decimal(int(tier1_amt * 100)) / Decimal("100"), 2)
    return tier1_final

def calc_tier2(emp, tier2_maximums):
    ret_year = int(emp["ret_date"][0:4])
    dob_year = int(emp["dob"][0:4])
    dob_month = int(emp["dob"][4:6])
    ret_month = int(emp["ret_date"][4:6])

    age_at_ret = trunc(Decimal(ret_year - dob_year) + Decimal(ret_month - dob_month) / 12, 2)

    monthly_earnings = []
    for earn_year, earn_amt in emp["earnings"]:
        t2_max = tier2_maximums.get(earn_year, earn_amt)
        capped = trunc(min(earn_amt, t2_max), 2)
        monthly = trunc(capped / 12, 2)
        for _ in range(12):
            monthly_earnings.append(monthly)

    monthly_earnings.sort(reverse=True)
    top_60 = monthly_earnings[:60]

    while len(top_60) < 60:
        top_60.append(Decimal("0"))

    total_60 = trunc(sum(top_60), 2)
    avg_60 = trunc(total_60 / 60, 2)

    tier2_work = trunc(Decimal("0.007") * avg_60 * emp["svc_years"], 4)
    tier2_amt = trunc(tier2_work, 2)

    t2_fra = 65
    if emp["svc_years"] >= 30:
        pass
    else:
        if age_at_ret < t2_fra:
            months_early = int((t2_fra - age_at_ret) * 12)
            if months_early > 36:
                t2_age_red = trunc(Decimal("1") - (Decimal("36") / 180) - \
                             ((Decimal(months_early) - 36) / 240), 6)
            else:
                t2_age_red = trunc(Decimal("1") - (Decimal(months_early) / 180), 6)
            tier2_amt = trunc(tier2_amt * t2_age_red, 2)

    tier2_final = trunc(Decimal(int(tier2_amt * 100)) / Decimal("100"), 2)
    return tier2_final

def format_benefit_record(ssn, name, tier1, tier2, total, status, msg):
    out = bytearray()
    out.extend(ssn.ljust(9).encode('latin-1'))
    out.extend(name.ljust(30).encode('latin-1'))
    out.extend(pack_comp3(tier1, 9, 2))
    out.extend(pack_comp3(tier2, 9, 2))
    out.extend(pack_comp3(total, 9, 2))
    out.extend(status.encode('latin-1'))
    out.extend(msg.ljust(50).encode('latin-1'))
    return bytes(out)

def format_number(value, width, decimals=2, with_commas=True):
    if with_commas:
        int_part = int(abs(value))
        frac_part = int((abs(value) - int_part) * (10 ** decimals))
        int_str = "{:,}".format(int_part)
        result = "{}.{:0{w}d}".format(int_str, frac_part, w=decimals)
    else:
        result = "{:.{w}f}".format(float(value), w=decimals)
    return result.rjust(width)

def write_page_header(report_state, lines):
    report_state.page_num += 1

    if report_state.page_num > 1:
        lines.append("")

    hdr1 = " " * 45 + "RAILROAD RETIREMENT BOARD BENEFITS REPORT" + " " * 30
    page_str = "{:5,d}".format(report_state.page_num).replace(",", "")
    hdr1 += "PAGE " + page_str.rjust(5) + " " * 4
    lines.append(hdr1[:REPORT_WIDTH])

    lines.append("=" * REPORT_WIDTH)
    lines.append("")

    hdr3 = " SSN       " + "NAME".ljust(30) + "  " + "RET YEAR" + "  "
    hdr3 += "TIER 1".ljust(12) + "  " + "TIER 2".ljust(12) + "  "
    hdr3 += "TOTAL".ljust(12) + "  " + "STATUS" + "  " + "MESSAGE"
    lines.append(hdr3[:REPORT_WIDTH])

    hdr4 = " " + "-" * 9 + "  " + "-" * 30 + "  " + "-" * 8 + "  "
    hdr4 += "-" * 12 + "  " + "-" * 12 + "  " + "-" * 12 + "  "
    hdr4 += "-" * 6 + "  " + "-" * 28
    lines.append(hdr4[:REPORT_WIDTH])

    report_state.line_num = 6

def write_detail_line(ssn, name, ret_year, t1, t2, total, status, msg, report_state, lines):
    line = " " + ssn.ljust(9) + "  " + name.ljust(30) + "  "
    line += str(ret_year).rjust(4) + " " * 6
    line += format_number(t1, 10) + "  "
    line += format_number(t2, 10) + "  "
    line += format_number(total, 10) + "    "
    line += status + " " * 7
    line += msg[:28]
    lines.append(line[:REPORT_WIDTH])
    report_state.line_num += 1

def write_subtotal_line(year, t1_tot, t2_tot, tt_tot, count, report_state, lines):
    lines.append("")

    sub_line = " " + "*" * 9 + "  " + "SUBTOTAL FOR YEAR " + str(year)
    sub_line += " " * 8 + "  " + " " * 8
    sub_line += format_number(t1_tot, 13) + " "
    sub_line += format_number(t2_tot, 13) + " "
    sub_line += format_number(tt_tot, 13) + "    "
    sub_line += "COUNT  " + "{:6,d}".format(count).replace(",", "").rjust(6)
    sub_line += " " * 16
    lines.append(sub_line[:REPORT_WIDTH])

    lines.append("")
    report_state.line_num += 3

def write_grand_total_line(t1_tot, t2_tot, tt_tot, count, lines):
    lines.append("")
    lines.append("=" * REPORT_WIDTH)

    grand_line = " " + "=" * 9 + "  " + "GRAND TOTALS".ljust(30) + "  "
    grand_line += " " * 8
    grand_line += format_number(t1_tot, 13) + " "
    grand_line += format_number(t2_tot, 13) + " "
    grand_line += format_number(tt_tot, 13) + "    "
    grand_line += "RECS: " + "{:7,d}".format(count).replace(",", "").rjust(7)
    grand_line += " " * 15
    lines.append(grand_line[:REPORT_WIDTH])

def format_summary_record(stats):
    out = bytearray()

    out.extend(str(stats.total_recs).zfill(6).encode('latin-1'))
    out.extend(str(stats.valid_count).zfill(6).encode('latin-1'))
    out.extend(str(stats.error_count).zfill(6).encode('latin-1'))

    out.extend(pack_comp3(stats.grand_t1, 13, 2))
    out.extend(pack_comp3(stats.grand_t2, 13, 2))
    out.extend(pack_comp3(stats.grand_tt, 13, 2))

    if stats.valid_count > 0:
        t1_avg = trunc(stats.grand_t1 / stats.valid_count, 2)
        t2_avg = trunc(stats.grand_t2 / stats.valid_count, 2)
    else:
        t1_avg = Decimal("0")
        t2_avg = Decimal("0")
    out.extend(pack_comp3(t1_avg, 11, 2))
    out.extend(pack_comp3(t2_avg, 11, 2))

    t1_min = Decimal("0") if stats.t1_min == Decimal("999999") else stats.t1_min
    t2_min = Decimal("0") if stats.t2_min == Decimal("999999") else stats.t2_min
    out.extend(pack_comp3(t1_min, 9, 2))
    out.extend(pack_comp3(stats.t1_max, 9, 2))
    out.extend(pack_comp3(t2_min, 9, 2))
    out.extend(pack_comp3(stats.t2_max, 9, 2))

    return bytes(out)

def main():
    bendpoints = load_bendpoints()
    idx_factors = load_indexing_factors()
    tier2_maximums = load_tier2_maximums()

    emp_file = DATA_DIR / "EMPLOYEE.DAT"
    benefits_file = DATA_DIR / "BENEFITS_PY.DAT"
    report_file = DATA_DIR / "BENEFITS_PY.RPT"
    summary_file = DATA_DIR / "SUMMARY_PY.DAT"

    output_records = bytearray()
    report_lines = []

    ctrl = ControlBreakState()
    rpt = ReportState()
    stats = SummaryStats()

    employees = []
    for line in emp_file.read_text().splitlines():
        if line.strip():
            employees.append(parse_employee_record(line))

    for emp in employees:
        ret_year = int(emp["ret_date"][0:4])
        ctrl.curr_ret_year = ret_year

        if emp["svc_years"] < 5:
            status = "E"
            msg = "INSUFFICIENT SERVICE YEARS"
            tier1 = Decimal("0")
            tier2 = Decimal("0")
            total = Decimal("0")
            stats.update_error()
        else:
            status = "V"
            msg = ""
            tier1 = calc_tier1(emp, bendpoints, idx_factors)
            tier2 = calc_tier2(emp, tier2_maximums)
            total = tier1 + tier2
            stats.update_valid(tier1, tier2, total)

        record = format_benefit_record(emp["ssn"], emp["name"],
                                        tier1, tier2, total, status, msg)
        output_records.extend(record)

        stats.total_recs += 1
        rpt.rec_count += 1

        if ctrl.first_record:
            ctrl.prev_ret_year = ctrl.curr_ret_year
            ctrl.first_record = False
            write_page_header(rpt, report_lines)
        else:
            if ctrl.curr_ret_year != ctrl.prev_ret_year:
                write_subtotal_line(ctrl.prev_ret_year, ctrl.year_t1_total,
                                   ctrl.year_t2_total, ctrl.year_tt_total,
                                   ctrl.year_count, rpt, report_lines)
                ctrl.reset_year_accumulators()
                ctrl.prev_ret_year = ctrl.curr_ret_year

        if rpt.need_new_page():
            write_page_header(rpt, report_lines)

        write_detail_line(emp["ssn"], emp["name"], ctrl.curr_ret_year,
                         tier1, tier2, total, status, msg, rpt, report_lines)

        ctrl.accumulate(tier1, tier2, total)

    if rpt.rec_count > 0:
        write_subtotal_line(ctrl.prev_ret_year, ctrl.year_t1_total,
                           ctrl.year_t2_total, ctrl.year_tt_total,
                           ctrl.year_count, rpt, report_lines)

    write_grand_total_line(stats.grand_t1, stats.grand_t2, stats.grand_tt,
                          rpt.rec_count, report_lines)

    benefits_file.write_bytes(bytes(output_records))
    report_file.write_text("\n".join(report_lines) + "\n")
    summary_file.write_bytes(format_summary_record(stats))

if __name__ == "__main__":
    main()
EOF

python3 /app/python/rrb_calculator.py
