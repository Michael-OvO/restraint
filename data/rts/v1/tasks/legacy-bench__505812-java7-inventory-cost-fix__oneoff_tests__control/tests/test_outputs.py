import os
import subprocess
import tempfile
import csv
import random
import math
from pathlib import Path
from typing import Tuple, Dict, Any, List

SOURCES = Path("/app/src/")
BIN = Path("/app/bin/")

REQUIRED_CLASSES = [
    "CostAllocationEngine",
    "WeightedAverageCostCalculator",
    "LogisticsExpenseAllocator",
    "CurrencyConverter",
    "WarehouseTransferProcessor",
    "InventoryDataParser",
    "ReportWriter",
]

SKU_WEIGHTS = {
    "SKU001": 2.5,
    "SKU002": 1.2,
    "SKU003": 5.0,
    "SKU004": 0.8,
    "SKU005": 3.3,
}

SKU_VOLUMES = {
    "SKU001": 0.015,
    "SKU002": 0.008,
    "SKU003": 0.030,
    "SKU004": 0.005,
    "SKU005": 0.020,
}


def truncate4(value):
    """Truncate to 4 decimal places (no rounding)."""
    return math.floor(value * 10000) / 10000.0


def truncate2(value):
    """Truncate to 2 decimal places (no rounding)."""
    return math.floor(value * 100) / 100.0


def _run(cmd, *, timeout=60) -> Tuple[int, str, str]:
    """Execute a shell command and return exit code, stdout, stderr."""
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        timeout=timeout,
        shell=True,
    )
    return p.returncode, p.stdout, p.stderr


def _compile():
    """Compile Java source files with strict flags - NO stdout/stderr allowed."""
    BIN.mkdir(parents=True, exist_ok=True)
    java_files = list(SOURCES.glob("*.java"))
    if not java_files:
        raise AssertionError("No Java files found in /app/src/")

    cmd = [
        "javac",
        "-Xlint:all",
        "-Xlint:-serial",
        "-Werror",
        "-source", "1.7",
        "-target", "1.7",
        "-d", str(BIN),
    ] + [str(f) for f in java_files]

    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0 and p.stdout == "" and p.stderr == "", \
        f"Compilation failed or produced output:\nstdout: {p.stdout}\nstderr: {p.stderr}"
    return BIN


def _create_transactions_csv(tmpdir: Path, transactions: List[Dict]) -> Path:
    """Create a transactions CSV file."""
    filepath = tmpdir / "transactions.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sku", "transaction_date", "transaction_type", "quantity", "unit_cost", "warehouse_id", "currency_code", "allocation_batch_id"])
        for t in transactions:
            writer.writerow([t["sku"], t["transaction_date"], t["transaction_type"], t["quantity"], t["unit_cost"], t["warehouse_id"], t["currency_code"], t["allocation_batch_id"]])
    return filepath


def _create_rates_csv(tmpdir: Path, rates: List[Dict]) -> Path:
    """Create a rates CSV file."""
    filepath = tmpdir / "rates.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["effective_date", "currency_code", "rate_to_usd"])
        for r in rates:
            writer.writerow([r["effective_date"], r["currency_code"], r["rate_to_usd"]])
    return filepath


def _create_expenses_csv(tmpdir: Path, expenses: List[Dict]) -> Path:
    """Create an expenses CSV file."""
    filepath = tmpdir / "expenses.csv"
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch_id", "total_expense", "allocation_basis"])
        for e in expenses:
            writer.writerow([e["batch_id"], e["total_expense"], e["allocation_basis"]])
    return filepath


def _run_engine(txn_file: Path, rates_file: Path, expenses_file: Path, output_file: Path) -> Tuple[int, str]:
    """Run the cost allocation engine and return exit code and stderr."""
    cmd = f"java -cp {BIN} CostAllocationEngine {txn_file} {rates_file} {expenses_file} {output_file}"
    ret, stdout, stderr = _run(cmd, timeout=30)
    return ret, stderr


def _parse_output(output_file: Path) -> List[Dict]:
    """Parse the output CSV into a list of dictionaries."""
    results = []
    with open(output_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                "sku": row["sku"],
                "warehouse_id": row["warehouse_id"],
                "final_quantity": int(row["final_quantity"]),
                "final_unit_cost": float(row["final_unit_cost"]),
                "allocated_expense": float(row["allocated_expense"]),
                "total_value": float(row["total_value"]),
            })
    return results


def test_source_files_exist():
    """
    Verify all required Java source files exist in /app/src/.
    Each component specified in task.yaml must be present.
    """
    assert SOURCES.exists(), f"Source directory {SOURCES} does not exist"

    for class_name in REQUIRED_CLASSES:
        java_file = SOURCES / f"{class_name}.java"
        assert java_file.exists(), f"Required file {java_file} does not exist"


def test_compilation_succeeds():
    """
    Verify all Java files compile with -Xlint:all -Werror without any output.
    This ensures clean Java 7 compatible code with no warnings.
    """
    _compile()

    for class_name in REQUIRED_CLASSES:
        class_file = BIN / f"{class_name}.class"
        assert class_file.exists(), f"Expected compiled class {class_file} not found"


def test_weighted_average_truncation():
    """
    Verify weighted average cost uses truncation, not rounding.
    Truncating 12.34567 to 4 decimals gives 12.3456, not 12.3457.
    This tests the core requirement that truncation is used.
    """
    _compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        transactions = [
            {"sku": "SKU001", "transaction_date": "2024-01-01", "transaction_type": "PURCHASE", "quantity": 100, "unit_cost": 10.00, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
            {"sku": "SKU001", "transaction_date": "2024-01-02", "transaction_type": "PURCHASE", "quantity": 50, "unit_cost": 12.333333, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
        ]

        txn_file = _create_transactions_csv(tmpdir, transactions)
        rates_file = _create_rates_csv(tmpdir, [])
        expenses_file = _create_expenses_csv(tmpdir, [])
        output_file = tmpdir / "output.csv"

        ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
        assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

        results = _parse_output(output_file)
        assert len(results) == 1

        prev_value = 100 * 10.00
        new_value = 50 * 12.333333
        expected_avg_raw = (prev_value + new_value) / 150
        expected_truncated_intermediate = truncate4(expected_avg_raw)

        assert results[0]["final_unit_cost"] == expected_truncated_intermediate, \
            f"Expected truncated avg {expected_truncated_intermediate}, got {results[0]['final_unit_cost']}"


def test_currency_conversion_truncation():
    """
    Verify currency conversion uses truncation, not rounding.
    Using a rate that clearly shows the difference between truncation and rounding.
    10 EUR * 1.12345 = 11.2345, truncated to 4 decimals is 11.2345.
    This tests the requirement for currency conversion truncation.
    """
    _compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        transactions = [
            {"sku": "SKU001", "transaction_date": "2024-01-15", "transaction_type": "PURCHASE", "quantity": 10, "unit_cost": 10.00, "warehouse_id": "WH1", "currency_code": "EUR", "allocation_batch_id": ""},
        ]

        rates = [
            {"effective_date": "2024-01-15", "currency_code": "EUR", "rate_to_usd": 1.12345},
        ]

        txn_file = _create_transactions_csv(tmpdir, transactions)
        rates_file = _create_rates_csv(tmpdir, rates)
        expenses_file = _create_expenses_csv(tmpdir, [])
        output_file = tmpdir / "output.csv"

        ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
        assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

        results = _parse_output(output_file)

        assert abs(results[0]["final_unit_cost"] - 11.2345) < 0.0002, \
            f"Expected converted cost ~11.2345, got {results[0]['final_unit_cost']}"


def test_weight_based_allocation():
    """
    Verify weight-based expense allocation calculates correctly.
    When allocation_basis is W, SKU weights should be used.
    This tests that the correct basis values are used.
    """
    _compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        transactions = [
            {"sku": "SKU001", "transaction_date": "2024-01-15", "transaction_type": "PURCHASE", "quantity": 100, "unit_cost": 10.00, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": "BATCH1"},
            {"sku": "SKU002", "transaction_date": "2024-01-15", "transaction_type": "PURCHASE", "quantity": 100, "unit_cost": 10.00, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": "BATCH1"},
        ]

        expenses = [
            {"batch_id": "BATCH1", "total_expense": 1000.00, "allocation_basis": "W"},
        ]

        txn_file = _create_transactions_csv(tmpdir, transactions)
        rates_file = _create_rates_csv(tmpdir, [])
        expenses_file = _create_expenses_csv(tmpdir, expenses)
        output_file = tmpdir / "output.csv"

        ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
        assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

        results = _parse_output(output_file)

        total_weight = SKU_WEIGHTS["SKU001"] + SKU_WEIGHTS["SKU002"]
        sku001_share = truncate2((SKU_WEIGHTS["SKU001"] / total_weight) * 1000.00)
        sku002_share = truncate2((SKU_WEIGHTS["SKU002"] / total_weight) * 1000.00)

        sku001_result = next((r for r in results if r["sku"] == "SKU001"), None)
        sku002_result = next((r for r in results if r["sku"] == "SKU002"), None)

        assert sku001_result is not None, "SKU001 should be in results"
        assert sku002_result is not None, "SKU002 should be in results"

        assert abs(sku001_result["allocated_expense"] - sku001_share) < 0.01, \
            f"SKU001 allocation should be {sku001_share}, got {sku001_result['allocated_expense']}"
        assert abs(sku002_result["allocated_expense"] - sku002_share) < 0.01, \
            f"SKU002 allocation should be {sku002_share}, got {sku002_result['allocated_expense']}"


def test_volume_based_allocation():
    """
    Verify volume-based expense allocation calculates correctly.
    When allocation_basis is V, SKU volumes should be used.
    This tests that the correct basis values are used.
    """
    _compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        transactions = [
            {"sku": "SKU001", "transaction_date": "2024-01-15", "transaction_type": "PURCHASE", "quantity": 100, "unit_cost": 10.00, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": "BATCH1"},
            {"sku": "SKU003", "transaction_date": "2024-01-15", "transaction_type": "PURCHASE", "quantity": 100, "unit_cost": 10.00, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": "BATCH1"},
        ]

        expenses = [
            {"batch_id": "BATCH1", "total_expense": 500.00, "allocation_basis": "V"},
        ]

        txn_file = _create_transactions_csv(tmpdir, transactions)
        rates_file = _create_rates_csv(tmpdir, [])
        expenses_file = _create_expenses_csv(tmpdir, expenses)
        output_file = tmpdir / "output.csv"

        ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
        assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

        results = _parse_output(output_file)

        total_volume = SKU_VOLUMES["SKU001"] + SKU_VOLUMES["SKU003"]
        sku001_share = truncate2((SKU_VOLUMES["SKU001"] / total_volume) * 500.00)
        sku003_share = truncate2((SKU_VOLUMES["SKU003"] / total_volume) * 500.00)

        sku001_result = next((r for r in results if r["sku"] == "SKU001"), None)
        sku003_result = next((r for r in results if r["sku"] == "SKU003"), None)

        assert sku001_result is not None, "SKU001 should be in results"
        assert sku003_result is not None, "SKU003 should be in results"

        assert abs(sku001_result["allocated_expense"] - sku001_share) < 0.01, \
            f"SKU001 allocation should be {sku001_share}, got {sku001_result['allocated_expense']}"
        assert abs(sku003_result["allocated_expense"] - sku003_share) < 0.01, \
            f"SKU003 allocation should be {sku003_share}, got {sku003_result['allocated_expense']}"


def test_warehouse_transfer_preserves_cost():
    """
    Verify inter-warehouse transfers carry exact unit cost without modification.
    The receiving warehouse should get the same cost the sending warehouse had.
    """
    _compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        transactions = [
            {"sku": "SKU001", "transaction_date": "2024-01-01", "transaction_type": "PURCHASE", "quantity": 100, "unit_cost": 15.7531, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
            {"sku": "SKU001", "transaction_date": "2024-01-10", "transaction_type": "TRANSFER_OUT", "quantity": 50, "unit_cost": 0, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
            {"sku": "SKU001", "transaction_date": "2024-01-10", "transaction_type": "TRANSFER_IN", "quantity": 50, "unit_cost": 0, "warehouse_id": "WH2", "currency_code": "USD", "allocation_batch_id": ""},
        ]

        txn_file = _create_transactions_csv(tmpdir, transactions)
        rates_file = _create_rates_csv(tmpdir, [])
        expenses_file = _create_expenses_csv(tmpdir, [])
        output_file = tmpdir / "output.csv"

        ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
        assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

        results = _parse_output(output_file)

        wh1_result = next((r for r in results if r["warehouse_id"] == "WH1"), None)
        wh2_result = next((r for r in results if r["warehouse_id"] == "WH2"), None)

        assert wh1_result is not None, "WH1 should have remaining inventory"
        assert wh2_result is not None, "WH2 should have received inventory"

        expected_cost = truncate4(15.7531)
        assert abs(wh1_result["final_unit_cost"] - expected_cost) < 0.00001, \
            f"WH1 cost should be {expected_cost}, got {wh1_result['final_unit_cost']}"
        assert abs(wh2_result["final_unit_cost"] - expected_cost) < 0.00001, \
            f"WH2 cost should match WH1 at {expected_cost}, got {wh2_result['final_unit_cost']}"


def test_total_value_calculation():
    """
    Verify total_value equals final_quantity times final_unit_cost, truncated to 2 decimals.
    """
    _compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        transactions = [
            {"sku": "SKU001", "transaction_date": "2024-01-01", "transaction_type": "PURCHASE", "quantity": 33, "unit_cost": 17.7777, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
        ]

        txn_file = _create_transactions_csv(tmpdir, transactions)
        rates_file = _create_rates_csv(tmpdir, [])
        expenses_file = _create_expenses_csv(tmpdir, [])
        output_file = tmpdir / "output.csv"

        ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
        assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

        results = _parse_output(output_file)
        r = results[0]

        expected_total = truncate2(r["final_quantity"] * r["final_unit_cost"])
        assert abs(r["total_value"] - expected_total) < 0.01, \
            f"Total value should be {expected_total}, got {r['total_value']}"


def test_output_sorted_by_sku_then_warehouse():
    """
    Verify output rows are sorted by sku ascending, then warehouse_id ascending.
    """
    _compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        transactions = [
            {"sku": "SKU003", "transaction_date": "2024-01-01", "transaction_type": "PURCHASE", "quantity": 10, "unit_cost": 10.00, "warehouse_id": "WH2", "currency_code": "USD", "allocation_batch_id": ""},
            {"sku": "SKU001", "transaction_date": "2024-01-01", "transaction_type": "PURCHASE", "quantity": 10, "unit_cost": 10.00, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
            {"sku": "SKU003", "transaction_date": "2024-01-01", "transaction_type": "PURCHASE", "quantity": 10, "unit_cost": 10.00, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
            {"sku": "SKU001", "transaction_date": "2024-01-01", "transaction_type": "PURCHASE", "quantity": 10, "unit_cost": 10.00, "warehouse_id": "WH2", "currency_code": "USD", "allocation_batch_id": ""},
        ]

        txn_file = _create_transactions_csv(tmpdir, transactions)
        rates_file = _create_rates_csv(tmpdir, [])
        expenses_file = _create_expenses_csv(tmpdir, [])
        output_file = tmpdir / "output.csv"

        ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
        assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

        results = _parse_output(output_file)

        keys = [(r["sku"], r["warehouse_id"]) for r in results]
        assert keys == sorted(keys), f"Output should be sorted by (sku, warehouse_id), got {keys}"


def test_output_csv_header():
    """
    Verify the output CSV has the correct header as specified in task.yaml.
    """
    _compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        transactions = [
            {"sku": "SKU001", "transaction_date": "2024-01-01", "transaction_type": "PURCHASE", "quantity": 10, "unit_cost": 10.00, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
        ]

        txn_file = _create_transactions_csv(tmpdir, transactions)
        rates_file = _create_rates_csv(tmpdir, [])
        expenses_file = _create_expenses_csv(tmpdir, [])
        output_file = tmpdir / "output.csv"

        ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
        assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

        with open(output_file, "r") as f:
            header = f.readline().strip()
            expected_header = "sku,warehouse_id,final_quantity,final_unit_cost,allocated_expense,total_value"
            assert header == expected_header, \
                f"Output header should be '{expected_header}', got '{header}'"


def test_random_scenarios():
    """
    Anti-hardcoding test: Run multiple random scenarios and verify
    the engine produces consistent and correct results for random inputs.
    """
    _compile()
    random.seed()

    for _ in range(5):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            transactions = []
            skus = ["SKU001", "SKU002", "SKU003", "SKU004", "SKU005"]
            warehouses = ["WH1", "WH2"]

            for _ in range(random.randint(5, 15)):
                sku = random.choice(skus)
                wh = random.choice(warehouses)
                qty = random.randint(10, 100)
                cost = random.uniform(5.0, 50.0)
                day = random.randint(1, 28)
                transactions.append({
                    "sku": sku,
                    "transaction_date": f"2024-01-{day:02d}",
                    "transaction_type": "PURCHASE",
                    "quantity": qty,
                    "unit_cost": cost,
                    "warehouse_id": wh,
                    "currency_code": "USD",
                    "allocation_batch_id": ""
                })

            txn_file = _create_transactions_csv(tmpdir, transactions)
            rates_file = _create_rates_csv(tmpdir, [])
            expenses_file = _create_expenses_csv(tmpdir, [])
            output_file = tmpdir / "output.csv"

            ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
            assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

            results = _parse_output(output_file)

            for r in results:
                assert r["final_quantity"] > 0, "Final quantity must be positive"
                assert r["final_unit_cost"] > 0, "Final unit cost must be positive"
                assert r["allocated_expense"] >= 0, "Allocated expense must be non-negative"
                assert r["total_value"] > 0, "Total value must be positive"

                expected_total = truncate2(r["final_quantity"] * r["final_unit_cost"])
                assert abs(r["total_value"] - expected_total) < 0.01, \
                    f"Total value calculation incorrect"


def test_sale_reduces_quantity():
    """
    Verify that SALE transactions reduce inventory quantity correctly.
    """
    _compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        transactions = [
            {"sku": "SKU001", "transaction_date": "2024-01-01", "transaction_type": "PURCHASE", "quantity": 100, "unit_cost": 10.00, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
            {"sku": "SKU001", "transaction_date": "2024-01-15", "transaction_type": "SALE", "quantity": 30, "unit_cost": 0, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
        ]

        txn_file = _create_transactions_csv(tmpdir, transactions)
        rates_file = _create_rates_csv(tmpdir, [])
        expenses_file = _create_expenses_csv(tmpdir, [])
        output_file = tmpdir / "output.csv"

        ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
        assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

        results = _parse_output(output_file)
        assert len(results) == 1
        assert results[0]["final_quantity"] == 70, \
            f"After sale of 30 from 100, quantity should be 70, got {results[0]['final_quantity']}"


def test_multiple_purchases_weighted_average():
    """
    Verify weighted average cost is computed correctly over multiple purchases.
    Test specific values where rounding vs truncation would differ.
    """
    _compile()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        transactions = [
            {"sku": "SKU002", "transaction_date": "2024-01-01", "transaction_type": "PURCHASE", "quantity": 60, "unit_cost": 8.3333, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
            {"sku": "SKU002", "transaction_date": "2024-01-05", "transaction_type": "PURCHASE", "quantity": 40, "unit_cost": 12.5555, "warehouse_id": "WH1", "currency_code": "USD", "allocation_batch_id": ""},
        ]

        txn_file = _create_transactions_csv(tmpdir, transactions)
        rates_file = _create_rates_csv(tmpdir, [])
        expenses_file = _create_expenses_csv(tmpdir, [])
        output_file = tmpdir / "output.csv"

        ret, stderr = _run_engine(txn_file, rates_file, expenses_file, output_file)
        assert ret == 0 and stderr == "", f"Engine failed: {stderr}"

        results = _parse_output(output_file)

        first_avg = truncate4(8.3333)
        second_total = (60 * first_avg) + (40 * 12.5555)
        expected_final = truncate4(second_total / 100)

        assert abs(results[0]["final_unit_cost"] - expected_final) < 0.0001, \
            f"Expected weighted avg {expected_final}, got {results[0]['final_unit_cost']}"
