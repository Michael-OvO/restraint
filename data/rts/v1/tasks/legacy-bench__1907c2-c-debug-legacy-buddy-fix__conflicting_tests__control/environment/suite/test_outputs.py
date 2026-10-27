import subprocess
import random
import re
import os
from pathlib import Path
from subprocess import PIPE
from typing import List, Tuple, Optional

SOURCES = Path("/app/src/")
BINARY = Path("/app/test_binary")
BINARY_ASAN = Path("/app/test_binary_asan")

PAGES_PER_ZONE = 64
NUM_ZONES = 2
MAX_ORDER = 4


def _compile(asan: bool = False) -> Path:
    """Compile all sources. Use asan=True for AddressSanitizer build."""
    c_files = list(SOURCES.glob("*.c"))
    assert c_files, f"{SOURCES} has no .c files."

    binary = BINARY_ASAN if asan else BINARY
    sanitizer_flags = "-fsanitize=address,undefined -fno-omit-frame-pointer" if asan else ""
    sources = " ".join(str(f) for f in c_files)

    cmd = f"gcc -Wall -Wextra -Werror -std=c23 -g {sanitizer_flags} {sources} -o {binary}"
    p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    assert p.returncode == 0 and p.stdout == "" and p.stderr == "", (
        f"Compilation failed:\nCommand: {cmd}\nExit: {p.returncode}\nstderr: {p.stderr}"
    )
    return binary


# =============================================================================
# ORACLE CLASS - Simulates correct buddy allocator behavior
# =============================================================================
class BuddyOracle:
    """Python oracle that simulates correct buddy allocator behavior.

    The oracle processes the SAME random commands as the program.
    Tests verify program output matches oracle output for EVERY command.
    """

    def __init__(self):
        self.zones: List[dict] = []
        for z in range(NUM_ZONES):
            zone = {
                'id': z,
                'bitmap': [0] * PAGES_PER_ZONE,  # 0=free, 1=allocated, 2=reserved
                'free_lists': [[] for _ in range(MAX_ORDER + 1)],
                'alloc_count': 0,
                'free_count': 0,
                'split_count': 0,
                'coalesce_count': 0,
            }
            # Initially all memory is in order-4 blocks (16 pages each)
            for i in range(0, PAGES_PER_ZONE, 1 << MAX_ORDER):
                zone['free_lists'][MAX_ORDER].append(i)
            self.zones.append(zone)
        self.regions: List[dict] = []
        self.total_pages = PAGES_PER_ZONE * NUM_ZONES
        self.reserved_pages = 0

    def add_region(self, base: int, count: int, rtype: int) -> str:
        """Add a memory region. Returns 'OK' or error."""
        if len(self.regions) >= 16:
            return "ERROR: region_limit"

        self.regions.append({'base': base, 'count': count, 'type': rtype})

        if rtype != 0:  # Not usable
            for pg in range(base, base + count):
                zone_id = pg // PAGES_PER_ZONE
                if zone_id < NUM_ZONES:
                    page_idx = pg % PAGES_PER_ZONE
                    if self.zones[zone_id]['bitmap'][page_idx] == 0:
                        self.zones[zone_id]['bitmap'][page_idx] = 2
                        self.reserved_pages += 1
                        # Remove from free lists
                        self._remove_from_freelists(zone_id, page_idx)
        return "OK"

    def _remove_from_freelists(self, zone_id: int, page_idx: int):
        """Remove a page from free lists when marking as reserved.

        For buddy allocator: find which block contains this page, remove it from
        its free list, then add back the parts that don't contain the reserved page.
        """
        zone = self.zones[zone_id]
        # Check from largest order down to find which block contains page_idx
        for order in range(MAX_ORDER, -1, -1):
            block_size = 1 << order
            aligned_start = (page_idx // block_size) * block_size
            if aligned_start in zone['free_lists'][order]:
                zone['free_lists'][order].remove(aligned_start)
                # Now recursively split and keep the parts that aren't reserved
                self._split_and_reserve(zone, aligned_start, order, page_idx)
                return

    def _split_and_reserve(self, zone: dict, block_start: int, order: int, reserved_page: int):
        """Split a block and add non-reserved parts to free lists."""
        if order == 0:
            # Base case - page is reserved, don't add to free list
            return

        # Split into two halves
        half_size = 1 << (order - 1)
        left_start = block_start
        right_start = block_start + half_size

        # Figure out which half contains the reserved page
        if reserved_page < right_start:
            # Reserved page is in left half - add right to free list
            zone['free_lists'][order - 1].append(right_start)
            self._split_and_reserve(zone, left_start, order - 1, reserved_page)
        else:
            # Reserved page is in right half - add left to free list
            zone['free_lists'][order - 1].append(left_start)
            self._split_and_reserve(zone, right_start, order - 1, reserved_page)

    def alloc_pages(self, zone_id: int, order: int) -> str:
        """Allocate 2^order pages. Returns 'PAGE=N' or error."""
        if zone_id < 0 or zone_id >= NUM_ZONES:
            return "ERROR: invalid_zone"
        if order < 0 or order > MAX_ORDER:
            return "ERROR: invalid_order"

        zone = self.zones[zone_id]
        block_size = 1 << order

        # Find a block - try each order from requested up to MAX
        for cur_order in range(order, MAX_ORDER + 1):
            # Try each block at this order
            tried_indices = []
            while zone['free_lists'][cur_order]:
                page_idx = zone['free_lists'][cur_order].pop(0)

                # Check for reserved pages in this block
                has_reserved = False
                for i in range(1 << cur_order):
                    if zone['bitmap'][page_idx + i] == 2:
                        has_reserved = True
                        break

                if has_reserved:
                    # Block contains reserved pages, save it and try next
                    tried_indices.append(page_idx)
                    continue

                # Put back blocks we couldn't use
                zone['free_lists'][cur_order] = tried_indices + zone['free_lists'][cur_order]

                # Split down to required order
                split_order = cur_order
                while split_order > order:
                    split_order -= 1
                    zone['split_count'] += 1
                    buddy = page_idx + (1 << split_order)
                    zone['free_lists'][split_order].append(buddy)

                # Mark pages as allocated
                for i in range(block_size):
                    zone['bitmap'][page_idx + i] = 1

                zone['alloc_count'] += 1
                return f"PAGE={page_idx}"

            # Put back blocks we tried at this order
            zone['free_lists'][cur_order] = tried_indices + zone['free_lists'][cur_order]

        return "ERROR: no_memory"

    def free_pages(self, zone_id: int, page_idx: int, order: int) -> str:
        """Free 2^order pages. Returns 'OK' or error."""
        if zone_id < 0 or zone_id >= NUM_ZONES:
            return "ERROR: invalid_zone"
        if order < 0 or order > MAX_ORDER:
            return "ERROR: invalid_order"

        zone = self.zones[zone_id]
        block_size = 1 << order

        if page_idx < 0 or page_idx + block_size > PAGES_PER_ZONE:
            return "ERROR: invalid_page"

        if (page_idx & ((1 << order) - 1)) != 0:
            return "ERROR: unaligned"

        # Check all pages in block
        for i in range(block_size):
            if zone['bitmap'][page_idx + i] == 2:
                return "ERROR: reserved"
            if zone['bitmap'][page_idx + i] == 0:
                return "ERROR: double_free"

        # Mark as free
        for i in range(block_size):
            zone['bitmap'][page_idx + i] = 0

        zone['free_count'] += 1

        # Coalesce with buddies
        while order < MAX_ORDER:
            buddy = page_idx ^ (1 << order)

            if buddy < 0 or buddy >= PAGES_PER_ZONE:
                break

            # Check if buddy is completely free
            buddy_free = True
            buddy_size = 1 << order
            for i in range(buddy_size):
                if zone['bitmap'][buddy + i] != 0:
                    buddy_free = False
                    break

            if not buddy_free:
                break

            # Check if buddy is in free list
            if buddy not in zone['free_lists'][order]:
                break

            # Remove buddy from free list
            zone['free_lists'][order].remove(buddy)
            zone['coalesce_count'] += 1

            # Merge
            if buddy < page_idx:
                page_idx = buddy
            order += 1

        zone['free_lists'][order].append(page_idx)
        return "OK"

    def query_page(self, zone_id: int, page_idx: int) -> str:
        """Query a page's state."""
        if zone_id < 0 or zone_id >= NUM_ZONES:
            return "ERROR: invalid_zone"
        if page_idx < 0 or page_idx >= PAGES_PER_ZONE:
            return "ERROR: invalid_page"

        state = self.zones[zone_id]['bitmap'][page_idx]
        if state == 0:
            return "FREE"
        elif state == 1:
            return "ALLOCATED"
        else:
            return "RESERVED"

    def zone_stats(self, zone_id: int) -> str:
        """Get zone statistics."""
        if zone_id < 0 or zone_id >= NUM_ZONES:
            return "ERROR: invalid_zone"

        zone = self.zones[zone_id]
        allocated = sum(1 for b in zone['bitmap'] if b == 1)
        reserved = sum(1 for b in zone['bitmap'] if b == 2)
        free = sum(1 for b in zone['bitmap'] if b == 0)

        return (f"zone={zone_id} allocs={zone['alloc_count']} frees={zone['free_count']} "
                f"splits={zone['split_count']} coalesces={zone['coalesce_count']} "
                f"allocated={allocated} reserved={reserved} free={free}")

    def global_stats(self) -> str:
        """Get global statistics."""
        total_allocs = sum(z['alloc_count'] for z in self.zones)
        total_frees = sum(z['free_count'] for z in self.zones)
        total_splits = sum(z['split_count'] for z in self.zones)
        total_coalesces = sum(z['coalesce_count'] for z in self.zones)

        total_allocated = sum(sum(1 for b in z['bitmap'] if b == 1) for z in self.zones)
        total_reserved = sum(sum(1 for b in z['bitmap'] if b == 2) for z in self.zones)
        usable = self.total_pages - total_reserved

        return (f"total={self.total_pages} usable={usable} reserved={total_reserved} "
                f"allocs={total_allocs} frees={total_frees} splits={total_splits} coalesces={total_coalesces}")

    def freelist_count(self, zone_id: int, order: int) -> str:
        """Get count of blocks in free list."""
        if zone_id < 0 or zone_id >= NUM_ZONES:
            return "ERROR: invalid_zone"
        if order < 0 or order > MAX_ORDER:
            return "ERROR: invalid_order"

        count = len(self.zones[zone_id]['free_lists'][order])
        return f"COUNT={count}"

    @property
    def allocated_pages(self) -> List[Tuple[int, int]]:
        """Return list of (zone_id, page_idx) for allocated pages."""
        result = []
        for z_id, zone in enumerate(self.zones):
            for i, state in enumerate(zone['bitmap']):
                if state == 1:
                    result.append((z_id, i))
        return result


# =============================================================================
# INTERACTIVE TEST HARNESS
# =============================================================================
class ProgramDriver:
    """Drives the program via stdin/stdout for interactive testing."""

    def __init__(self, binary: Path, env: dict = None):
        run_env = os.environ.copy()
        if env:
            run_env.update(env)
        self.proc = subprocess.Popen(
            [str(binary)], stdin=PIPE, stdout=PIPE, stderr=PIPE,
            text=True, bufsize=1, env=run_env
        )

    def send(self, cmd: str) -> str:
        """Send command and return response."""
        self.proc.stdin.write(cmd + "\n")
        self.proc.stdin.flush()
        return self.proc.stdout.readline().strip()

    def close(self) -> Tuple[int, str]:
        """Close and return (returncode, stderr)."""
        self.proc.stdin.close()
        self.proc.wait(timeout=10)
        stderr = self.proc.stderr.read()
        return self.proc.returncode, stderr


# =============================================================================
# COMPILATION TESTS
# =============================================================================
def test_compiles_cleanly():
    """Verify source compiles with -Wall -Wextra -Werror and no output."""
    binary = _compile(asan=False)
    assert binary.exists()


def test_compiles_with_asan():
    """Verify source compiles with AddressSanitizer enabled."""
    binary = _compile(asan=True)
    assert binary.exists()


# =============================================================================
# RANDOM SEQUENCE TESTS (ANTI-CHEAT)
# =============================================================================
def test_random_alloc_free_sequence():
    """Drive program with random alloc/free sequence, verify each result."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    for _ in range(random.randint(50, 100)):
        op = random.choices(['alloc', 'free', 'query'], weights=[0.5, 0.3, 0.2])[0]

        if op == 'alloc':
            zone = random.randint(0, NUM_ZONES - 1)
            order = random.randint(0, MAX_ORDER)
            result = driver.send(f"alloc {zone} {order}")
            expected = oracle.alloc_pages(zone, order)
            assert result == expected, f"alloc {zone} {order}: got '{result}', expected '{expected}'"

        elif op == 'free':
            if oracle.allocated_pages and random.random() < 0.7:
                zone, page = random.choice(oracle.allocated_pages)
                order = random.randint(0, 2)  # Small orders for testing
            else:
                zone = random.randint(-1, NUM_ZONES)
                page = random.randint(-5, PAGES_PER_ZONE + 5)
                order = random.randint(-1, MAX_ORDER + 1)
            result = driver.send(f"free {zone} {page} {order}")
            expected = oracle.free_pages(zone, page, order)
            assert result == expected, f"free {zone} {page} {order}: got '{result}', expected '{expected}'"

        elif op == 'query':
            zone = random.randint(0, NUM_ZONES - 1)
            page = random.randint(0, PAGES_PER_ZONE - 1)
            result = driver.send(f"query {zone} {page}")
            expected = oracle.query_page(zone, page)
            assert result == expected, f"query {zone} {page}: got '{result}', expected '{expected}'"

    # Verify final stats
    for z in range(NUM_ZONES):
        result = driver.send(f"zstats {z}")
        expected = oracle.zone_stats(z)
        assert result == expected, f"zstats {z}: got '{result}', expected '{expected}'"

    result = driver.send("stats")
    expected = oracle.global_stats()
    assert result == expected, f"stats: got '{result}', expected '{expected}'"

    ret, stderr = driver.close()
    assert ret == 0, f"Program exited with {ret}"
    assert stderr == "", f"Unexpected stderr: {stderr}"


def test_random_sequence_with_invariant_checks():
    """Random sequence with invariant verification after each operation."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    for _ in range(random.randint(30, 60)):
        if random.random() < 0.6:
            zone = random.randint(0, NUM_ZONES - 1)
            order = random.randint(0, MAX_ORDER)
            result = driver.send(f"alloc {zone} {order}")
            expected = oracle.alloc_pages(zone, order)
            assert result == expected
        else:
            if oracle.allocated_pages:
                zone, page = random.choice(oracle.allocated_pages)
                order = 0
                result = driver.send(f"free {zone} {page} {order}")
                expected = oracle.free_pages(zone, page, order)
                assert result == expected

        # Check invariant: zone stats must match oracle
        for z in range(NUM_ZONES):
            result = driver.send(f"zstats {z}")
            expected = oracle.zone_stats(z)
            assert result == expected, f"Invariant violated: {result} != {expected}"

    driver.close()


def test_state_interrogation():
    """Probe internal state with random queries to verify correctness."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    # Do some allocations
    for _ in range(random.randint(10, 20)):
        zone = random.randint(0, NUM_ZONES - 1)
        order = random.randint(0, 2)
        driver.send(f"alloc {zone} {order}")
        oracle.alloc_pages(zone, order)

    # Free some randomly
    if oracle.allocated_pages:
        sample_size = min(5, len(oracle.allocated_pages))
        for zone, page in random.sample(oracle.allocated_pages, k=sample_size):
            driver.send(f"free {zone} {page} 0")
            oracle.free_pages(zone, page, 0)

    # Query all pages and verify
    for z in range(NUM_ZONES):
        for p in range(PAGES_PER_ZONE):
            result = driver.send(f"query {z} {p}")
            expected = oracle.query_page(z, p)
            assert result == expected, f"State mismatch at zone {z} page {p}: {result} != {expected}"

    driver.close()


def test_cross_validate_stats_vs_queries():
    """Cross-validate: stats counts must match individual query results."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    # Random operations
    for _ in range(random.randint(20, 40)):
        if random.random() < 0.6:
            zone = random.randint(0, NUM_ZONES - 1)
            order = random.randint(0, 2)
            driver.send(f"alloc {zone} {order}")
            oracle.alloc_pages(zone, order)
        else:
            if oracle.allocated_pages:
                zone, page = random.choice(oracle.allocated_pages)
                driver.send(f"free {zone} {page} 0")
                oracle.free_pages(zone, page, 0)

    # Get zone stats and verify via queries
    for z in range(NUM_ZONES):
        stats_result = driver.send(f"zstats {z}")
        match = re.search(r"allocated=(\d+) reserved=(\d+) free=(\d+)", stats_result)
        assert match, f"Stats format invalid: {stats_result}"
        reported_allocated = int(match.group(1))
        reported_reserved = int(match.group(2))
        reported_free = int(match.group(3))

        # Count via queries
        allocated_count = 0
        reserved_count = 0
        free_count = 0
        for p in range(PAGES_PER_ZONE):
            q = driver.send(f"query {z} {p}")
            if q == "ALLOCATED":
                allocated_count += 1
            elif q == "RESERVED":
                reserved_count += 1
            elif q == "FREE":
                free_count += 1

        assert allocated_count == reported_allocated, f"Zone {z}: queried allocated {allocated_count} != reported {reported_allocated}"
        assert reserved_count == reported_reserved, f"Zone {z}: queried reserved {reserved_count} != reported {reported_reserved}"
        assert free_count == reported_free, f"Zone {z}: queried free {free_count} != reported {reported_free}"

        # Invariant: must sum to PAGES_PER_ZONE
        total = allocated_count + reserved_count + free_count
        assert total == PAGES_PER_ZONE, f"Zone {z}: counts sum to {total}, expected {PAGES_PER_ZONE}"

    driver.close()


# =============================================================================
# RESERVED REGION TESTS
# =============================================================================
def test_reserved_regions_not_allocatable():
    """Verify that reserved regions cannot be allocated."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    # Mark some pages as reserved
    result = driver.send("region 0 16 1")  # Reserved in zone 0
    expected = oracle.add_region(0, 16, 1)
    assert result == expected

    result = driver.send("region 64 8 2")  # ACPI in zone 1
    expected = oracle.add_region(64, 8, 2)
    assert result == expected

    # Query reserved pages
    for p in range(16):
        result = driver.send(f"query 0 {p}")
        assert result == "RESERVED", f"Page 0:{p} should be RESERVED: {result}"

    for p in range(8):
        result = driver.send(f"query 1 {p}")
        assert result == "RESERVED", f"Page 1:{p} should be RESERVED: {result}"

    # Try to allocate - should work in unreserved areas
    result = driver.send("alloc 0 0")
    expected = oracle.alloc_pages(0, 0)
    assert result == expected, f"Alloc in zone 0: got '{result}', expected '{expected}'"

    # Verify stats show correct reserved count
    result = driver.send("stats")
    expected = oracle.global_stats()
    assert result == expected, f"Stats mismatch: {result} != {expected}"

    driver.close()


def test_reserved_regions_not_freeable():
    """Verify that reserved regions cannot be freed."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    # Mark pages as reserved
    driver.send("region 0 4 1")
    oracle.add_region(0, 4, 1)

    # Try to free reserved page
    result = driver.send("free 0 0 0")
    expected = oracle.free_pages(0, 0, 0)
    assert result == expected, f"Free reserved: got '{result}', expected '{expected}'"
    assert result == "ERROR: reserved"

    driver.close()


# =============================================================================
# BUDDY COALESCING TESTS
# =============================================================================
def test_buddy_coalescing():
    """Verify that freed buddies coalesce correctly."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    # Allocate two order-0 blocks that should be buddies
    result1 = driver.send("alloc 0 0")
    oracle.alloc_pages(0, 0)
    result2 = driver.send("alloc 0 0")
    oracle.alloc_pages(0, 0)

    # Get starting indices
    page1 = int(result1.split("=")[1])
    page2 = int(result2.split("=")[1])

    # Free both - should coalesce if they're buddies
    result = driver.send(f"free 0 {page1} 0")
    oracle.free_pages(0, page1, 0)
    assert result == "OK"

    result = driver.send(f"free 0 {page2} 0")
    oracle.free_pages(0, page2, 0)
    assert result == "OK"

    # Check zone stats for coalesce count
    result = driver.send("zstats 0")
    expected = oracle.zone_stats(0)
    assert result == expected, f"Stats mismatch: {result} != {expected}"

    # Verify coalesces counter was updated - when freeing two adjacent order-0
    # blocks that came from splitting an order-4, they coalesce back up
    match = re.search(r'coalesces=(\d+)', result)
    assert match, f"No coalesces in stats: {result}"
    coalesce_count = int(match.group(1))
    assert coalesce_count > 0, f"Expected coalesces > 0, got: {result}"

    driver.close()


def test_split_counting():
    """Verify that splits are counted correctly when allocating smaller blocks."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    # Initially we have order-4 blocks. Allocating order-0 requires 4 splits
    result = driver.send("alloc 0 0")
    expected = oracle.alloc_pages(0, 0)
    assert result == expected

    # Check split count
    result = driver.send("zstats 0")
    expected = oracle.zone_stats(0)
    assert result == expected, f"Stats mismatch: {result} != {expected}"
    assert "splits=4" in result, f"Expected 4 splits, got: {result}"

    driver.close()


def test_freelist_counts():
    """Verify freelist counts are accurate after operations."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    # Initially: 4 blocks at order-4 (since 64 pages / 16 pages per block = 4)
    result = driver.send("freelist 0 4")
    expected = oracle.freelist_count(0, 4)
    assert result == expected, f"Initial freelist: {result} != {expected}"
    assert result == "COUNT=4"

    # Allocate order-0, creates splits
    driver.send("alloc 0 0")
    oracle.alloc_pages(0, 0)

    # Now check all orders
    for order in range(MAX_ORDER + 1):
        result = driver.send(f"freelist 0 {order}")
        expected = oracle.freelist_count(0, order)
        assert result == expected, f"Freelist order {order}: {result} != {expected}"

    driver.close()


# =============================================================================
# ADDRESS SANITIZER TESTS
# =============================================================================
def test_asan_random_sequence():
    """Run random sequence under ASan to catch memory bugs."""
    binary = _compile(asan=True)
    env = {"ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1"}
    driver = ProgramDriver(binary, env=env)
    oracle = BuddyOracle()

    for _ in range(random.randint(50, 100)):
        if random.random() < 0.6:
            zone = random.randint(0, NUM_ZONES - 1)
            order = random.randint(0, MAX_ORDER)
            driver.send(f"alloc {zone} {order}")
            oracle.alloc_pages(zone, order)
        else:
            if oracle.allocated_pages:
                zone, page = random.choice(oracle.allocated_pages)
                driver.send(f"free {zone} {page} 0")
                oracle.free_pages(zone, page, 0)

    driver.send("stats")
    ret, stderr = driver.close()
    assert "AddressSanitizer" not in stderr, f"Memory error: {stderr}"
    assert "UndefinedBehaviorSanitizer" not in stderr, f"UB error: {stderr}"
    assert ret == 0


def test_asan_repeated_sequences():
    """Run multiple random sequences under ASan."""
    binary = _compile(asan=True)
    env = {"ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1"}

    for iteration in range(10):
        driver = ProgramDriver(binary, env=env)
        oracle = BuddyOracle()

        for _ in range(random.randint(20, 50)):
            if random.random() < 0.5:
                zone = random.randint(0, NUM_ZONES - 1)
                order = random.randint(0, MAX_ORDER)
                driver.send(f"alloc {zone} {order}")
                oracle.alloc_pages(zone, order)
            else:
                if oracle.allocated_pages:
                    zone, page = random.choice(oracle.allocated_pages)
                    driver.send(f"free {zone} {page} 0")
                    oracle.free_pages(zone, page, 0)

        ret, stderr = driver.close()
        assert "AddressSanitizer" not in stderr, f"Iteration {iteration}: {stderr}"
        assert ret == 0


def test_asan_with_reserved_regions():
    """Test reserved region handling under ASan."""
    binary = _compile(asan=True)
    env = {"ASAN_OPTIONS": "detect_leaks=1:abort_on_error=1"}
    driver = ProgramDriver(binary, env=env)
    oracle = BuddyOracle()

    # Add some reserved regions
    driver.send("region 0 8 1")
    oracle.add_region(0, 8, 1)
    driver.send("region 32 4 2")
    oracle.add_region(32, 4, 2)

    # Random operations avoiding reserved pages
    for _ in range(30):
        if random.random() < 0.6:
            zone = random.randint(0, NUM_ZONES - 1)
            order = random.randint(0, 2)
            driver.send(f"alloc {zone} {order}")
            oracle.alloc_pages(zone, order)
        else:
            if oracle.allocated_pages:
                zone, page = random.choice(oracle.allocated_pages)
                driver.send(f"free {zone} {page} 0")
                oracle.free_pages(zone, page, 0)

    ret, stderr = driver.close()
    assert "AddressSanitizer" not in stderr
    assert ret == 0


# =============================================================================
# EDGE CASE TESTS
# =============================================================================
def test_double_free_rejected():
    """Verify double-free attempts are correctly rejected."""
    binary = _compile()
    driver = ProgramDriver(binary)

    result1 = driver.send("alloc 0 0")
    assert result1.startswith("PAGE="), f"Alloc failed: {result1}"
    page = int(result1.split("=")[1])

    result2 = driver.send(f"free 0 {page} 0")
    assert result2 == "OK", f"Free failed: {result2}"

    result3 = driver.send(f"free 0 {page} 0")
    assert result3 == "ERROR: double_free", f"Double-free not detected: {result3}"

    driver.close()


def test_invalid_arguments():
    """Verify invalid arguments are rejected."""
    binary = _compile()
    driver = ProgramDriver(binary)

    # Invalid zone
    result = driver.send("alloc -1 0")
    assert result == "ERROR: invalid_zone"
    result = driver.send("alloc 5 0")
    assert result == "ERROR: invalid_zone"

    # Invalid order
    result = driver.send("alloc 0 -1")
    assert result == "ERROR: invalid_order"
    result = driver.send("alloc 0 10")
    assert result == "ERROR: invalid_order"

    # Invalid page indices
    result = driver.send("free 0 -1 0")
    assert result == "ERROR: invalid_page"
    result = driver.send("free 0 100 0")
    assert result == "ERROR: invalid_page"

    # Unaligned free
    result = driver.send("alloc 0 1")  # Allocate order-1 (2 pages)
    page = int(result.split("=")[1])
    result = driver.send(f"free 0 {page + 1} 1")  # Try to free unaligned
    assert result == "ERROR: unaligned" or result == "ERROR: double_free"

    driver.close()


def test_pool_exhaustion():
    """Verify pool exhaustion is handled correctly."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    # Allocate all memory in zone 0
    allocated = []
    for _ in range(PAGES_PER_ZONE):
        result = driver.send("alloc 0 0")
        expected = oracle.alloc_pages(0, 0)
        assert result == expected, f"Alloc: {result} != {expected}"
        if result.startswith("PAGE="):
            allocated.append(int(result.split("=")[1]))
        else:
            break

    # Next alloc should fail
    result = driver.send("alloc 0 0")
    assert result == "ERROR: no_memory", f"Expected no_memory, got: {result}"

    # Free one, then alloc should work
    if allocated:
        driver.send(f"free 0 {allocated[0]} 0")
        oracle.free_pages(0, allocated[0], 0)
        result = driver.send("alloc 0 0")
        expected = oracle.alloc_pages(0, 0)
        assert result == expected

    driver.close()


def test_empty_zones():
    """Verify queries on empty zones return FREE for all pages."""
    binary = _compile()
    driver = ProgramDriver(binary)

    for z in range(NUM_ZONES):
        for p in range(PAGES_PER_ZONE):
            result = driver.send(f"query {z} {p}")
            assert result == "FREE", f"Zone {z} page {p} should be FREE: {result}"

    driver.close()


# =============================================================================
# MUTATION / PERTURBATION TESTS
# =============================================================================
def test_repeated_random_sequences():
    """Run multiple different random sequences to catch intermittent bugs."""
    binary = _compile()

    for iteration in range(15):
        driver = ProgramDriver(binary)
        oracle = BuddyOracle()

        for _ in range(random.randint(30, 80)):
            op = random.choices(['alloc', 'free', 'query', 'zstats', 'freelist'],
                               weights=[0.4, 0.3, 0.15, 0.1, 0.05])[0]

            if op == 'alloc':
                zone = random.randint(0, NUM_ZONES - 1)
                order = random.randint(0, MAX_ORDER)
                result = driver.send(f"alloc {zone} {order}")
                expected = oracle.alloc_pages(zone, order)
                assert result == expected, f"Iter {iteration}: alloc mismatch"
            elif op == 'free':
                if oracle.allocated_pages:
                    zone, page = random.choice(oracle.allocated_pages)
                    result = driver.send(f"free {zone} {page} 0")
                    expected = oracle.free_pages(zone, page, 0)
                    assert result == expected, f"Iter {iteration}: free mismatch"
            elif op == 'query':
                zone = random.randint(0, NUM_ZONES - 1)
                page = random.randint(0, PAGES_PER_ZONE - 1)
                result = driver.send(f"query {zone} {page}")
                expected = oracle.query_page(zone, page)
                assert result == expected, f"Iter {iteration}: query mismatch"
            elif op == 'zstats':
                zone = random.randint(0, NUM_ZONES - 1)
                result = driver.send(f"zstats {zone}")
                expected = oracle.zone_stats(zone)
                assert result == expected, f"Iter {iteration}: zstats mismatch"
            else:
                zone = random.randint(0, NUM_ZONES - 1)
                order = random.randint(0, MAX_ORDER)
                result = driver.send(f"freelist {zone} {order}")
                expected = oracle.freelist_count(zone, order)
                assert result == expected, f"Iter {iteration}: freelist mismatch"

        ret, _ = driver.close()
        assert ret == 0, f"Iteration {iteration} failed"


def test_alloc_free_alloc_pattern():
    """Test alloc-free-alloc patterns to verify block reuse."""
    binary = _compile()
    driver = ProgramDriver(binary)
    oracle = BuddyOracle()

    # Allocate many blocks
    for _ in range(20):
        zone = random.randint(0, NUM_ZONES - 1)
        result = driver.send(f"alloc {zone} 0")
        expected = oracle.alloc_pages(zone, 0)
        assert result == expected

    # Free random blocks
    to_free = random.sample(oracle.allocated_pages, k=min(10, len(oracle.allocated_pages)))
    for zone, page in to_free:
        result = driver.send(f"free {zone} {page} 0")
        expected = oracle.free_pages(zone, page, 0)
        assert result == expected

    # Allocate again - should reuse freed blocks
    for _ in range(len(to_free)):
        zone = random.randint(0, NUM_ZONES - 1)
        result = driver.send(f"alloc {zone} 0")
        expected = oracle.alloc_pages(zone, 0)
        assert result == expected

    # Verify final state
    result = driver.send("stats")
    expected = oracle.global_stats()
    assert result == expected

    driver.close()


# =============================================================================
# UNSAFE FUNCTION/PATTERN CHECKS
# =============================================================================
UNSAFE_PATTERNS = [
    (r'\bsprintf\s*\(', 'sprintf', 'snprintf'),
    (r'\bvsprintf\s*\(', 'vsprintf', 'vsnprintf'),
    (r'\bstrcpy\s*\(', 'strcpy', 'strncpy or strlcpy'),
    (r'\bstrcat\s*\(', 'strcat', 'strncat or strlcat'),
    (r'\bgets\s*\(', 'gets', 'fgets'),
    (r'\bNULL\b', 'NULL', 'nullptr'),
]


def test_no_unsafe_patterns():
    """Verify unsafe C functions/patterns are replaced with safe variants."""
    c_files = list(SOURCES.glob("*.c")) + list(SOURCES.glob("*.h"))
    for c_file in c_files:
        code = c_file.read_text()
        for pattern, unsafe, safe in UNSAFE_PATTERNS:
            matches = re.findall(pattern, code)
            assert not matches, (
                f"Unsafe pattern '{unsafe}' found in {c_file.name}. "
                f"Use '{safe}' instead."
            )
