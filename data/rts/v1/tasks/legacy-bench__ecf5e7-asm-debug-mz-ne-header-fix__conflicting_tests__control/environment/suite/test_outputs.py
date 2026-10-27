"""Tests for DOS MZ/NE executable analyzer.

Tests verify the assembly correctly parses MZ and NE executable formats,
extracts header information, segment tables, and relocation entries.
All tests run under both normal and ASan modes.
"""

import os
import random
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

SOURCES = Path("/app/src/")
BINARY = Path("/app/test_binary")
BINARY_ASAN = Path("/app/test_binary_asan")


def _compile(asan: bool = False) -> Path:
    """Compile assembly with optional AddressSanitizer instrumentation."""
    binary = BINARY_ASAN if asan else BINARY
    s_files = list(SOURCES.glob("*.s"))
    assert s_files, f"{SOURCES} has no .s files"

    if asan:
        for s_file in s_files:
            obj = s_file.with_suffix(".o")
            p = subprocess.run(f"gcc -c {s_file} -o {obj}", shell=True, capture_output=True, text=True)
            if p.returncode != 0:
                pytest.skip(f"ASan compilation not supported: {p.stderr}")
        objs = " ".join(str(f.with_suffix(".o")) for f in s_files)
        cmd = f"gcc -no-pie -fsanitize=address {objs} -o {binary}"
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if p.returncode != 0:
            pytest.skip(f"ASan linking not supported (nostdlib binary): {p.stderr}")
    else:
        sources = " ".join(str(f) for f in s_files)
        cmd = f"gcc -no-pie -nostdlib {sources} -o {binary}"
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        assert p.returncode == 0 and p.stdout == "" and p.stderr == "", \
            f"Compilation failed:\nstdout: {p.stdout}\nstderr: {p.stderr}"
    return binary


@pytest.fixture(params=[False, True], ids=["normal", "asan"])
def binary(request) -> Path:
    """Fixture that compiles binary in normal or ASan mode."""
    return _compile(asan=request.param)


def _run_binary(binary: Path, args: list, timeout: int = 5):
    """Run binary with arguments and return (stdout, stderr, returncode)."""
    env = os.environ.copy()
    if "asan" in str(binary).lower():
        env["ASAN_OPTIONS"] = "detect_leaks=0:abort_on_error=1"
    p = subprocess.run([str(binary)] + args, capture_output=True, text=True, timeout=timeout, env=env)
    return p.stdout, p.stderr, p.returncode


class MZNEOracle:
    """Oracle that computes expected output for MZ/NE executable analysis."""

    def __init__(self):
        self.data = bytearray()
        self.mz = {}
        self.ne = {}
        self.segments = []
        self.relocs = []
        self.has_ne = False

    def create_mz_only(self, pages=3, num_relocs=2, initial_cs=0, initial_ip=0,
                       reloc_entries=None):
        """Create a pure MZ (DOS) executable."""
        self.mz = {
            'bytes_last_page': 0x50,
            'pages_in_file': pages,
            'num_relocations': num_relocs,
            'header_paragraphs': 4,
            'min_extra': 0,
            'max_extra': 0xFFFF,
            'initial_ss': 0,
            'initial_sp': 0xB8,
            'checksum': 0,
            'initial_ip': initial_ip,
            'initial_cs': initial_cs,
            'reloc_table_offset': 0x40,
            'overlay_number': 0,
            'ne_header_offset': 0,
        }
        self.has_ne = False
        self.segments = []

        if reloc_entries is None:
            reloc_entries = [(random.randint(0, 0xFFFF), random.randint(0, 0xF)) for _ in range(num_relocs)]
        self.relocs = reloc_entries

        header = bytearray(64)
        header[0:2] = b'MZ'
        header[2:4] = struct.pack('<H', self.mz['bytes_last_page'])
        header[4:6] = struct.pack('<H', self.mz['pages_in_file'])
        header[6:8] = struct.pack('<H', self.mz['num_relocations'])
        header[8:10] = struct.pack('<H', self.mz['header_paragraphs'])
        header[10:12] = struct.pack('<H', self.mz['min_extra'])
        header[12:14] = struct.pack('<H', self.mz['max_extra'])
        header[14:16] = struct.pack('<H', self.mz['initial_ss'])
        header[16:18] = struct.pack('<H', self.mz['initial_sp'])
        header[18:20] = struct.pack('<H', self.mz['checksum'])
        header[20:22] = struct.pack('<H', self.mz['initial_ip'])
        header[22:24] = struct.pack('<H', self.mz['initial_cs'])
        header[24:26] = struct.pack('<H', self.mz['reloc_table_offset'])
        header[26:28] = struct.pack('<H', self.mz['overlay_number'])

        reloc_data = bytearray()
        for off, seg in self.relocs:
            reloc_data += struct.pack('<HH', off, seg)

        self.data = header + reloc_data + bytearray(100)
        return self

    def create_mz_ne(self, pages=3, mz_relocs=2, mz_cs=1, mz_ip=0x100,
                     segment_count=3, module_count=2, alignment_shift=9,
                     entry_segment=1, entry_offset=0x200,
                     segment_specs=None):
        """Create an MZ+NE (Windows 1.x-3.x) executable."""
        self.mz = {
            'bytes_last_page': 0x50,
            'pages_in_file': pages,
            'num_relocations': mz_relocs,
            'header_paragraphs': 4,
            'min_extra': 0,
            'max_extra': 0xFFFF,
            'initial_ss': 0,
            'initial_sp': 0xB8,
            'checksum': 0,
            'initial_ip': mz_ip,
            'initial_cs': mz_cs,
            'reloc_table_offset': 0x40,
            'overlay_number': 0,
            'ne_header_offset': 0x80,
        }
        self.has_ne = True

        self.relocs = [(random.randint(0, 0xFFFF), random.randint(0, 0xF)) for _ in range(mz_relocs)]

        self.ne = {
            'linker_version': 0x0500,
            'entry_table_offset': 0x40,
            'entry_table_length': 0x10,
            'crc': 0xDEADBEEF,
            'flags': 0x8000,
            'auto_data_segment': 1,
            'heap_init': 0x1000,
            'stack_init': 0x2000,
            'entry_offset': entry_offset,
            'entry_segment': entry_segment,
            'stack_offset': 0x100,
            'stack_segment': 2,
            'segment_count': segment_count,
            'module_ref_count': module_count,
            'nonres_name_size': 0x10,
            'segment_table_offset': 0x42,
            'resource_table_offset': 0x60,
            'resident_name_offset': 0x60,
            'module_ref_offset': 0x60,
            'import_name_offset': 0x60,
            'nonres_name_offset': 0,
            'moveable_entries': 0,
            'alignment_shift': alignment_shift,
            'resource_segments': 0,
            'target_os': 2,
            'flags2': 0,
            'fastload_area': 0,
            'fastload_size': 0,
            'reserved': 0,
            'expected_windows_version': 0x0300,
        }

        if segment_specs is None:
            segment_specs = []
            sector = 2
            for i in range(segment_count):
                is_code = (i % 2 == 0)
                length = random.randint(0x80, 0x400)
                flags = 0x0001 if is_code else 0x0000
                segment_specs.append((sector, length, flags, length))
                sector += (length >> alignment_shift) + 1

        self.segments = segment_specs

        mz_header = bytearray(64)
        mz_header[0:2] = b'MZ'
        mz_header[2:4] = struct.pack('<H', self.mz['bytes_last_page'])
        mz_header[4:6] = struct.pack('<H', self.mz['pages_in_file'])
        mz_header[6:8] = struct.pack('<H', self.mz['num_relocations'])
        mz_header[8:10] = struct.pack('<H', self.mz['header_paragraphs'])
        mz_header[10:12] = struct.pack('<H', self.mz['min_extra'])
        mz_header[12:14] = struct.pack('<H', self.mz['max_extra'])
        mz_header[14:16] = struct.pack('<H', self.mz['initial_ss'])
        mz_header[16:18] = struct.pack('<H', self.mz['initial_sp'])
        mz_header[18:20] = struct.pack('<H', self.mz['checksum'])
        mz_header[20:22] = struct.pack('<H', self.mz['initial_ip'])
        mz_header[22:24] = struct.pack('<H', self.mz['initial_cs'])
        mz_header[24:26] = struct.pack('<H', self.mz['reloc_table_offset'])
        mz_header[26:28] = struct.pack('<H', self.mz['overlay_number'])
        mz_header[0x3C:0x40] = struct.pack('<I', self.mz['ne_header_offset'])

        reloc_data = bytearray()
        for off, seg in self.relocs:
            reloc_data += struct.pack('<HH', off, seg)

        ne_header = bytearray(66)
        ne_header[0:2] = b'NE'
        ne_header[2:4] = struct.pack('<H', self.ne['linker_version'])
        ne_header[4:6] = struct.pack('<H', self.ne['entry_table_offset'])
        ne_header[6:8] = struct.pack('<H', self.ne['entry_table_length'])
        ne_header[8:12] = struct.pack('<I', self.ne['crc'])
        ne_header[12:14] = struct.pack('<H', self.ne['flags'])
        ne_header[14:16] = struct.pack('<H', self.ne['auto_data_segment'])
        ne_header[16:18] = struct.pack('<H', self.ne['heap_init'])
        ne_header[18:20] = struct.pack('<H', self.ne['stack_init'])
        ne_header[20:22] = struct.pack('<H', self.ne['entry_offset'])
        ne_header[22:24] = struct.pack('<H', self.ne['entry_segment'])
        ne_header[24:26] = struct.pack('<H', self.ne['stack_offset'])
        ne_header[26:28] = struct.pack('<H', self.ne['stack_segment'])
        ne_header[28:30] = struct.pack('<H', self.ne['segment_count'])
        ne_header[30:32] = struct.pack('<H', self.ne['module_ref_count'])
        ne_header[32:34] = struct.pack('<H', self.ne['nonres_name_size'])
        ne_header[34:36] = struct.pack('<H', self.ne['segment_table_offset'])
        ne_header[36:38] = struct.pack('<H', self.ne['resource_table_offset'])
        ne_header[38:40] = struct.pack('<H', self.ne['resident_name_offset'])
        ne_header[40:42] = struct.pack('<H', self.ne['module_ref_offset'])
        ne_header[42:44] = struct.pack('<H', self.ne['import_name_offset'])
        ne_header[44:48] = struct.pack('<I', self.ne['nonres_name_offset'])
        ne_header[48:50] = struct.pack('<H', self.ne['moveable_entries'])
        ne_header[50:52] = struct.pack('<H', self.ne['alignment_shift'])
        ne_header[52:54] = struct.pack('<H', self.ne['resource_segments'])
        ne_header[54] = self.ne['target_os']
        ne_header[55] = self.ne['flags2']
        ne_header[56:60] = struct.pack('<I', self.ne['fastload_area'])
        ne_header[60:62] = struct.pack('<H', self.ne['fastload_size'])
        ne_header[62:64] = struct.pack('<H', self.ne['reserved'])
        ne_header[64:66] = struct.pack('<H', self.ne['expected_windows_version'])

        seg_table = bytearray()
        for sector_off, length, flags, min_alloc in self.segments:
            seg_table += struct.pack('<HHHH', sector_off, length, flags, min_alloc)

        padding1 = bytearray(0x80 - 64 - len(reloc_data))
        padding2 = bytearray(max(0, 0x42 - 66))

        self.data = mz_header + reloc_data + padding1 + ne_header + padding2 + seg_table + bytearray(1024)
        return self

    def expected_headers(self) -> str:
        """Generate expected output for 'headers' mode."""
        lines = []
        mz_line = f"MZ pages={self.mz['pages_in_file']:04d} relocs={self.mz['num_relocations']:04d} " \
                  f"entry={self.mz['initial_cs']:04X}:{self.mz['initial_ip']:04X}"
        lines.append(mz_line)

        if self.has_ne:
            ne_line = f"NE segments={self.ne['segment_count']:02d} modules={self.ne['module_ref_count']:02d} " \
                      f"entry={self.ne['entry_segment']:04X}:{self.ne['entry_offset']:04X}"
            lines.append(ne_line)

        code_bytes, data_bytes = self._compute_code_data_bytes()
        summary = f"EXECUTABLE type={'NE' if self.has_ne else 'MZ'} code={code_bytes:04d} " \
                  f"data={data_bytes:04d} relocs={self.mz['num_relocations']:04d}"
        lines.append(summary)
        return '\n'.join(lines) + '\n'

    def expected_segments(self) -> str:
        """Generate expected output for 'segments' mode."""
        lines = []
        if self.has_ne:
            shift = self.ne['alignment_shift']
            for i, (sector_off, length, flags, _) in enumerate(self.segments):
                computed_off = sector_off << shift
                seg_type = "CODE" if (flags & 1) else "DATA"
                line = f"SEG {i:02d} off={computed_off:08X} len={length:04X} flags={flags:04X} type={seg_type}"
                lines.append(line)

        code_bytes, data_bytes = self._compute_code_data_bytes()
        summary = f"EXECUTABLE type={'NE' if self.has_ne else 'MZ'} code={code_bytes:04d} " \
                  f"data={data_bytes:04d} relocs={self.mz['num_relocations']:04d}"
        lines.append(summary)
        return '\n'.join(lines) + '\n'

    def expected_relocs(self) -> str:
        """Generate expected output for 'relocs' mode."""
        lines = []
        for off, seg in self.relocs:
            line = f"RELOC seg={seg:04X} off={off:04X}"
            lines.append(line)

        code_bytes, data_bytes = self._compute_code_data_bytes()
        summary = f"EXECUTABLE type={'NE' if self.has_ne else 'MZ'} code={code_bytes:04d} " \
                  f"data={data_bytes:04d} relocs={self.mz['num_relocations']:04d}"
        lines.append(summary)
        return '\n'.join(lines) + '\n'

    def _compute_code_data_bytes(self) -> tuple:
        """Compute total code and data bytes from segments."""
        code = 0
        data = 0
        for _, length, flags, _ in self.segments:
            if flags & 1:
                code += length
            else:
                data += length
        return code, data

    def write_to_file(self, path: str):
        """Write executable data to file."""
        with open(path, 'wb') as f:
            f.write(self.data)


def test_compiles_cleanly(binary):
    """Verify assembly compiles with no errors or warnings."""
    assert binary.exists()


def test_mz_headers_basic(binary):
    """Verify correct parsing of basic MZ headers."""
    oracle = MZNEOracle().create_mz_only(pages=5, num_relocs=3, initial_cs=0x1234, initial_ip=0x5678)
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        oracle.write_to_file(f.name)
        stdout, stderr, ret = _run_binary(binary, [f.name, 'headers'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr, f"Memory error: {stderr}"
    assert ret == 0, f"Exit code {ret}"
    assert stderr == "", f"Stderr: {stderr}"
    assert stdout == oracle.expected_headers(), f"Got:\n{stdout}\nExpected:\n{oracle.expected_headers()}"


def test_mz_relocs_basic(binary):
    """Verify correct parsing of MZ relocation entries."""
    relocs = [(0x1234, 0x0001), (0x5678, 0x0002), (0xABCD, 0x0003)]
    oracle = MZNEOracle().create_mz_only(pages=3, num_relocs=3, reloc_entries=relocs)
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        oracle.write_to_file(f.name)
        stdout, stderr, ret = _run_binary(binary, [f.name, 'relocs'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr, f"Memory error: {stderr}"
    assert ret == 0, f"Exit code {ret}"
    assert stdout == oracle.expected_relocs(), f"Got:\n{stdout}\nExpected:\n{oracle.expected_relocs()}"


def test_ne_headers_basic(binary):
    """Verify correct parsing of NE headers with segment and module counts."""
    oracle = MZNEOracle().create_mz_ne(segment_count=4, module_count=3,
                                        entry_segment=2, entry_offset=0x0300)
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        oracle.write_to_file(f.name)
        stdout, stderr, ret = _run_binary(binary, [f.name, 'headers'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr, f"Memory error: {stderr}"
    assert ret == 0, f"Exit code {ret}"
    assert stdout == oracle.expected_headers(), f"Got:\n{stdout}\nExpected:\n{oracle.expected_headers()}"


def test_ne_segments_with_alignment(binary):
    """Verify segment offset calculation using alignment_shift multiplier."""
    segment_specs = [
        (0x02, 0x0100, 0x0001, 0x0100),
        (0x04, 0x0200, 0x0000, 0x0200),
        (0x08, 0x0080, 0x0001, 0x0080),
    ]
    oracle = MZNEOracle().create_mz_ne(segment_count=3, alignment_shift=9,
                                        segment_specs=segment_specs)
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        oracle.write_to_file(f.name)
        stdout, stderr, ret = _run_binary(binary, [f.name, 'segments'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr, f"Memory error: {stderr}"
    assert ret == 0
    assert stdout == oracle.expected_segments(), f"Got:\n{stdout}\nExpected:\n{oracle.expected_segments()}"


def test_ne_segments_different_alignment(binary):
    """Verify segment offsets with different alignment shift values."""
    segment_specs = [
        (0x10, 0x0200, 0x0001, 0x0200),
        (0x20, 0x0100, 0x0000, 0x0100),
    ]
    oracle = MZNEOracle().create_mz_ne(segment_count=2, alignment_shift=4,
                                        segment_specs=segment_specs)
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        oracle.write_to_file(f.name)
        stdout, stderr, ret = _run_binary(binary, [f.name, 'segments'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr
    assert ret == 0
    assert stdout == oracle.expected_segments()


def test_error_file_not_found(binary):
    """Verify correct error message when file does not exist."""
    stdout, stderr, ret = _run_binary(binary, ['/nonexistent/path/file.exe', 'headers'])
    assert "AddressSanitizer" not in stderr
    assert ret == 1
    assert stdout == "ERROR: file_not_found\n"


def test_error_bad_magic(binary):
    """Verify correct error message when MZ magic is invalid."""
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        f.write(b'NOTMZ' + b'\x00' * 60)
        f.flush()
        stdout, stderr, ret = _run_binary(binary, [f.name, 'headers'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr
    assert ret == 1
    assert stdout == "ERROR: bad_magic\n"


def test_error_truncated_file(binary):
    """Verify correct error message when file is too small for MZ header."""
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        f.write(b'MZ' + b'\x00' * 10)
        f.flush()
        stdout, stderr, ret = _run_binary(binary, [f.name, 'headers'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr
    assert ret == 1
    assert stdout == "ERROR: truncated\n"


def test_error_bad_ne_offset(binary):
    """Verify correct error message when NE header has invalid magic."""
    data = bytearray(330)
    data[0:2] = b'MZ'
    data[0x3C:0x40] = struct.pack('<I', 0x80)
    data[0x80:0x82] = b'XX'

    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        f.write(data)
        f.flush()
        stdout, stderr, ret = _run_binary(binary, [f.name, 'headers'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr
    assert ret == 1
    assert stdout == "ERROR: bad_ne_offset\n"


def test_random_mz_executables(binary):
    """Test with multiple randomly generated MZ executables."""
    for _ in range(15):
        pages = random.randint(1, 100)
        num_relocs = random.randint(0, 10)
        cs = random.randint(0, 0xFFFF)
        ip = random.randint(0, 0xFFFF)

        oracle = MZNEOracle().create_mz_only(pages=pages, num_relocs=num_relocs,
                                              initial_cs=cs, initial_ip=ip)
        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
            oracle.write_to_file(f.name)

            stdout, stderr, ret = _run_binary(binary, [f.name, 'headers'])
            assert "AddressSanitizer" not in stderr
            assert ret == 0
            assert stdout == oracle.expected_headers()

            stdout, stderr, ret = _run_binary(binary, [f.name, 'relocs'])
            assert "AddressSanitizer" not in stderr
            assert ret == 0
            assert stdout == oracle.expected_relocs()

            os.unlink(f.name)


def test_random_ne_executables(binary):
    """Test with multiple randomly generated NE executables."""
    for _ in range(15):
        seg_count = random.randint(1, 8)
        mod_count = random.randint(0, 10)
        shift = random.randint(4, 12)
        entry_seg = random.randint(1, max(1, seg_count))
        entry_off = random.randint(0, 0xFFFF)

        segments = []
        sector = 2
        for i in range(seg_count):
            is_code = random.choice([True, False])
            length = random.randint(0x40, 0x400)
            flags = 0x0001 if is_code else 0x0000
            segments.append((sector, length, flags, length))
            sector += random.randint(1, 4)

        oracle = MZNEOracle().create_mz_ne(
            segment_count=seg_count, module_count=mod_count,
            alignment_shift=shift, entry_segment=entry_seg,
            entry_offset=entry_off, segment_specs=segments
        )

        with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
            oracle.write_to_file(f.name)

            stdout, stderr, ret = _run_binary(binary, [f.name, 'headers'])
            assert "AddressSanitizer" not in stderr
            assert ret == 0
            assert stdout == oracle.expected_headers(), f"headers mismatch"

            stdout, stderr, ret = _run_binary(binary, [f.name, 'segments'])
            assert "AddressSanitizer" not in stderr
            assert ret == 0
            assert stdout == oracle.expected_segments(), f"segments mismatch"

            stdout, stderr, ret = _run_binary(binary, [f.name, 'relocs'])
            assert "AddressSanitizer" not in stderr
            assert ret == 0
            assert stdout == oracle.expected_relocs(), f"relocs mismatch"

            os.unlink(f.name)


def test_code_data_byte_totals(binary):
    """Verify correct calculation of total code and data bytes from segments."""
    segments = [
        (0x02, 0x0100, 0x0001, 0x0100),
        (0x04, 0x0200, 0x0000, 0x0200),
        (0x06, 0x0150, 0x0001, 0x0150),
        (0x08, 0x0050, 0x0000, 0x0050),
    ]
    oracle = MZNEOracle().create_mz_ne(segment_count=4, segment_specs=segments)

    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        oracle.write_to_file(f.name)
        stdout, stderr, ret = _run_binary(binary, [f.name, 'headers'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr
    assert ret == 0

    code_expected = 0x0100 + 0x0150
    data_expected = 0x0200 + 0x0050

    assert f"code={code_expected:04d}" in stdout
    assert f"data={data_expected:04d}" in stdout


def test_mz_only_zero_relocs(binary):
    """Verify correct handling of MZ executable with no relocations."""
    oracle = MZNEOracle().create_mz_only(pages=2, num_relocs=0, reloc_entries=[])
    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        oracle.write_to_file(f.name)
        stdout, stderr, ret = _run_binary(binary, [f.name, 'relocs'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr
    assert ret == 0
    assert stdout == oracle.expected_relocs()
    assert "relocs=0000" in stdout


def test_ne_single_segment(binary):
    """Verify correct handling of NE with single segment."""
    segments = [(0x02, 0x1000, 0x0001, 0x1000)]
    oracle = MZNEOracle().create_mz_ne(segment_count=1, segment_specs=segments)

    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        oracle.write_to_file(f.name)
        stdout, stderr, ret = _run_binary(binary, [f.name, 'segments'])
        os.unlink(f.name)

    assert "AddressSanitizer" not in stderr
    assert ret == 0
    assert "SEG 00" in stdout
    assert "SEG 01" not in stdout


def test_repeated_analysis(binary):
    """Run repeated analyses to catch intermittent memory issues."""
    oracle = MZNEOracle().create_mz_ne(segment_count=4, module_count=2)

    with tempfile.NamedTemporaryFile(suffix='.exe', delete=False) as f:
        oracle.write_to_file(f.name)

        for _ in range(20):
            mode = random.choice(['headers', 'segments', 'relocs'])
            stdout, stderr, ret = _run_binary(binary, [f.name, mode])
            assert "AddressSanitizer" not in stderr
            assert ret == 0

            if mode == 'headers':
                assert stdout == oracle.expected_headers()
            elif mode == 'segments':
                assert stdout == oracle.expected_segments()
            else:
                assert stdout == oracle.expected_relocs()

        os.unlink(f.name)
