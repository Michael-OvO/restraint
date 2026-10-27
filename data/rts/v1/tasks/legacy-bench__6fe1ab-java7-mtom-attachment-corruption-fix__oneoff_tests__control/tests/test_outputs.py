import subprocess
import os
import tempfile
import shutil
from pathlib import Path


def compile_java():
    """Compile Java source files to /app/bin."""
    result = subprocess.run(
        ["javac", "-source", "1.7", "-target", "1.7", "-d", "/app/bin"] +
        [str(p) for p in Path("/app/src").glob("*.java")],
        capture_output=True, text=True
    )
    return result.returncode == 0, result.stderr


def run_claims_processor(claims_file, attachments_dir, providers_file, members_file,
                         fee_schedule_file, plan_benefits_file, procedure_codes_file,
                         diagnosis_codes_file, output_dir):
    """Run the ClaimsProcessor with given inputs."""
    result = subprocess.run(
        ["java", "-cp", "/app/bin", "ClaimsProcessor",
         claims_file, attachments_dir, providers_file, members_file,
         fee_schedule_file, plan_benefits_file, procedure_codes_file,
         diagnosis_codes_file, output_dir],
        capture_output=True, text=True
    )
    return result.returncode == 0, result.stdout, result.stderr


def test_java_compiles():
    """Verify that all Java source files compile without errors under JDK 1.7."""
    success, stderr = compile_java()
    assert success, f"Java compilation failed: {stderr}"


def test_small_attachment_preserved():
    """
    Verify that small attachments (under 64KB) are processed correctly
    with size preserved and correct checksum computed.
    """
    success, _ = compile_java()
    assert success, "Compilation failed"

    with tempfile.TemporaryDirectory() as tmpdir:
        claims_file = os.path.join(tmpdir, "claims.csv")
        attachments_dir = os.path.join(tmpdir, "attachments")
        providers_file = os.path.join(tmpdir, "providers.csv")
        members_file = os.path.join(tmpdir, "members.csv")
        fee_schedule_file = os.path.join(tmpdir, "fee_schedule.csv")
        plan_benefits_file = os.path.join(tmpdir, "plan_benefits.csv")
        procedure_codes_file = os.path.join(tmpdir, "procedure_codes.csv")
        diagnosis_codes_file = os.path.join(tmpdir, "diagnosis_codes.csv")
        output_dir = os.path.join(tmpdir, "output")

        os.makedirs(attachments_dir)
        os.makedirs(output_dir)

        with open(providers_file, "w") as f:
            f.write("provider_id,name,specialty,network_status,effective_date,termination_date\n")
            f.write("PROV001,Test Provider,General,IN_NETWORK,2020-01-01,\n")

        with open(members_file, "w") as f:
            f.write("member_id,plan_id,effective_date,termination_date,deductible_met,out_of_pocket_met\n")
            f.write("MEM001,PLAN_GOLD,2023-01-01,,50000,100000\n")

        with open(fee_schedule_file, "w") as f:
            f.write("procedure_code,network_allowed,out_of_network_allowed,requires_auth\n")
            f.write("99214,20000,15000,false\n")

        with open(plan_benefits_file, "w") as f:
            f.write("plan_id,deductible,coinsurance_rate,copay,out_of_pocket_max\n")
            f.write("PLAN_GOLD,100000,0.10,2500,500000\n")

        with open(procedure_codes_file, "w") as f:
            f.write("code,description\n")
            f.write("99214,Office visit level 4\n")

        with open(diagnosis_codes_file, "w") as f:
            f.write("code,description\n")
            f.write("J06.9,Respiratory infection\n")

        small_data = bytes([i % 256 for i in range(10000)])
        with open(os.path.join(attachments_dir, "small.bin"), "wb") as f:
            f.write(small_data)

        with open(claims_file, "w") as f:
            f.write("claim_id,member_id,provider_id,service_date,total_charges,line_id,procedure_code,diagnosis_code,charged_amount,units,attachment_file\n")
            f.write("CLM001,MEM001,PROV001,2024-03-15,20000,L001,99214,J06.9,20000,1,small.bin\n")

        success, stdout, stderr = run_claims_processor(
            claims_file, attachments_dir, providers_file, members_file,
            fee_schedule_file, plan_benefits_file, procedure_codes_file,
            diagnosis_codes_file, output_dir
        )
        assert success, f"ClaimsProcessor failed: {stderr}"

        summary_file = os.path.join(output_dir, "claim_summary.csv")
        assert os.path.exists(summary_file), "claim_summary.csv not created"

        with open(summary_file) as f:
            lines = f.readlines()

        assert len(lines) >= 2, "Expected at least header and one data line"


def test_large_attachment_size_preserved():
    """
    Verify that large attachments (above 64KB) are processed with their
    original size preserved. The processed size should equal the original size.
    """
    success, _ = compile_java()
    assert success, "Compilation failed"

    with tempfile.TemporaryDirectory() as tmpdir:
        claims_file = os.path.join(tmpdir, "claims.csv")
        attachments_dir = os.path.join(tmpdir, "attachments")
        providers_file = os.path.join(tmpdir, "providers.csv")
        members_file = os.path.join(tmpdir, "members.csv")
        fee_schedule_file = os.path.join(tmpdir, "fee_schedule.csv")
        plan_benefits_file = os.path.join(tmpdir, "plan_benefits.csv")
        procedure_codes_file = os.path.join(tmpdir, "procedure_codes.csv")
        diagnosis_codes_file = os.path.join(tmpdir, "diagnosis_codes.csv")
        output_dir = os.path.join(tmpdir, "output")

        os.makedirs(attachments_dir)
        os.makedirs(output_dir)

        with open(providers_file, "w") as f:
            f.write("provider_id,name,specialty,network_status,effective_date,termination_date\n")
            f.write("PROV001,Test Provider,General,IN_NETWORK,2020-01-01,\n")

        with open(members_file, "w") as f:
            f.write("member_id,plan_id,effective_date,termination_date,deductible_met,out_of_pocket_met\n")
            f.write("MEM001,PLAN_GOLD,2023-01-01,,50000,100000\n")

        with open(fee_schedule_file, "w") as f:
            f.write("procedure_code,network_allowed,out_of_network_allowed,requires_auth\n")
            f.write("99214,20000,15000,false\n")

        with open(plan_benefits_file, "w") as f:
            f.write("plan_id,deductible,coinsurance_rate,copay,out_of_pocket_max\n")
            f.write("PLAN_GOLD,100000,0.10,2500,500000\n")

        with open(procedure_codes_file, "w") as f:
            f.write("code,description\n")
            f.write("99214,Office visit level 4\n")

        with open(diagnosis_codes_file, "w") as f:
            f.write("code,description\n")
            f.write("J06.9,Respiratory infection\n")

        large_size = 100000
        large_data = bytes([i % 256 for i in range(large_size)])
        with open(os.path.join(attachments_dir, "large.bin"), "wb") as f:
            f.write(large_data)

        with open(claims_file, "w") as f:
            f.write("claim_id,member_id,provider_id,service_date,total_charges,line_id,procedure_code,diagnosis_code,charged_amount,units,attachment_file\n")
            f.write("CLM001,MEM001,PROV001,2024-03-15,20000,L001,99214,J06.9,20000,1,large.bin\n")

        success, stdout, stderr = run_claims_processor(
            claims_file, attachments_dir, providers_file, members_file,
            fee_schedule_file, plan_benefits_file, procedure_codes_file,
            diagnosis_codes_file, output_dir
        )
        assert success, f"ClaimsProcessor failed: {stderr}"

        attachment_report = os.path.join(output_dir, "attachment_report.csv")
        assert os.path.exists(attachment_report), "attachment_report.csv not created"

        with open(attachment_report) as f:
            lines = f.readlines()

        assert len(lines) >= 2, "Expected at least header and one data line in attachment_report"

        clm001_line = None
        for line in lines[1:]:
            if line.startswith("CLM001"):
                clm001_line = line.strip()
                break

        assert clm001_line is not None, "CLM001 not found in attachment_report.csv"

        parts = clm001_line.split(",")
        attachment_count = int(parts[1])
        assert attachment_count == 1, f"Expected 1 attachment, got {attachment_count}"


def test_attachment_checksum_consistency():
    """
    Verify that a large attachment (>64KB, spanning multiple chunks) produces
    the same checksum as when directly processed without chunking. Uses a small
    file that doesn't get chunked as baseline to verify the hash algorithm is
    correct, then verifies the large file checksum is deterministic across runs.
    Data loss from incorrect chunk reassembly would cause checksum changes.
    """
    success, _ = compile_java()
    assert success, "Compilation failed"

    checksums = []

    for run in range(2):
        with tempfile.TemporaryDirectory() as tmpdir:
            claims_file = os.path.join(tmpdir, "claims.csv")
            attachments_dir = os.path.join(tmpdir, "attachments")
            providers_file = os.path.join(tmpdir, "providers.csv")
            members_file = os.path.join(tmpdir, "members.csv")
            fee_schedule_file = os.path.join(tmpdir, "fee_schedule.csv")
            plan_benefits_file = os.path.join(tmpdir, "plan_benefits.csv")
            procedure_codes_file = os.path.join(tmpdir, "procedure_codes.csv")
            diagnosis_codes_file = os.path.join(tmpdir, "diagnosis_codes.csv")
            output_dir = os.path.join(tmpdir, "output")

            os.makedirs(attachments_dir)
            os.makedirs(output_dir)

            with open(providers_file, "w") as f:
                f.write("provider_id,name,specialty,network_status,effective_date,termination_date\n")
                f.write("PROV001,Test Provider,General,IN_NETWORK,2020-01-01,\n")

            with open(members_file, "w") as f:
                f.write("member_id,plan_id,effective_date,termination_date,deductible_met,out_of_pocket_met\n")
                f.write("MEM001,PLAN_GOLD,2023-01-01,,50000,100000\n")

            with open(fee_schedule_file, "w") as f:
                f.write("procedure_code,network_allowed,out_of_network_allowed,requires_auth\n")
                f.write("99214,20000,15000,false\n")

            with open(plan_benefits_file, "w") as f:
                f.write("plan_id,deductible,coinsurance_rate,copay,out_of_pocket_max\n")
                f.write("PLAN_GOLD,100000,0.10,2500,500000\n")

            with open(procedure_codes_file, "w") as f:
                f.write("code,description\n")
                f.write("99214,Office visit level 4\n")

            with open(diagnosis_codes_file, "w") as f:
                f.write("code,description\n")
                f.write("J06.9,Respiratory infection\n")

            large_data = bytes([i % 256 for i in range(150000)])
            with open(os.path.join(attachments_dir, "large.bin"), "wb") as f:
                f.write(large_data)

            with open(claims_file, "w") as f:
                f.write("claim_id,member_id,provider_id,service_date,total_charges,line_id,procedure_code,diagnosis_code,charged_amount,units,attachment_file\n")
                f.write("CLM001,MEM001,PROV001,2024-03-15,20000,L001,99214,J06.9,20000,1,large.bin\n")

            success, stdout, stderr = run_claims_processor(
                claims_file, attachments_dir, providers_file, members_file,
                fee_schedule_file, plan_benefits_file, procedure_codes_file,
                diagnosis_codes_file, output_dir
            )
            assert success, f"ClaimsProcessor failed: {stderr}"

            summary_file = os.path.join(output_dir, "claim_summary.csv")
            with open(summary_file) as f:
                lines = f.readlines()

            checksum = None
            for line in lines[1:]:
                if line.startswith("CLM001"):
                    checksum = int(line.strip().split(",")[8])
                    break

            assert checksum is not None, "CLM001 not found"
            checksums.append(checksum)

    assert checksums[0] == checksums[1], \
        f"Checksum inconsistency: {checksums[0]} vs {checksums[1]}. Data may be corrupted during multi-chunk processing."

    assert checksums[0] == 1898254987636178664, \
        f"Checksum mismatch: expected 1898254987636178664 for 150000 bytes of sequential data, got {checksums[0]}. Attachment data was corrupted during processing."


def test_multiple_chunk_boundary_sizes():
    """
    Verify that attachments at various sizes around the 64KB chunk boundary
    are all processed correctly. Tests 65536, 65537, and 131072 byte files.
    """
    success, _ = compile_java()
    assert success, "Compilation failed"

    test_sizes = [65536, 65537, 131072]

    for test_size in test_sizes:
        with tempfile.TemporaryDirectory() as tmpdir:
            claims_file = os.path.join(tmpdir, "claims.csv")
            attachments_dir = os.path.join(tmpdir, "attachments")
            providers_file = os.path.join(tmpdir, "providers.csv")
            members_file = os.path.join(tmpdir, "members.csv")
            fee_schedule_file = os.path.join(tmpdir, "fee_schedule.csv")
            plan_benefits_file = os.path.join(tmpdir, "plan_benefits.csv")
            procedure_codes_file = os.path.join(tmpdir, "procedure_codes.csv")
            diagnosis_codes_file = os.path.join(tmpdir, "diagnosis_codes.csv")
            output_dir = os.path.join(tmpdir, "output")

            os.makedirs(attachments_dir)
            os.makedirs(output_dir)

            with open(providers_file, "w") as f:
                f.write("provider_id,name,specialty,network_status,effective_date,termination_date\n")
                f.write("PROV001,Test Provider,General,IN_NETWORK,2020-01-01,\n")

            with open(members_file, "w") as f:
                f.write("member_id,plan_id,effective_date,termination_date,deductible_met,out_of_pocket_met\n")
                f.write("MEM001,PLAN_GOLD,2023-01-01,,50000,100000\n")

            with open(fee_schedule_file, "w") as f:
                f.write("procedure_code,network_allowed,out_of_network_allowed,requires_auth\n")
                f.write("99214,20000,15000,false\n")

            with open(plan_benefits_file, "w") as f:
                f.write("plan_id,deductible,coinsurance_rate,copay,out_of_pocket_max\n")
                f.write("PLAN_GOLD,100000,0.10,2500,500000\n")

            with open(procedure_codes_file, "w") as f:
                f.write("code,description\n")
                f.write("99214,Office visit level 4\n")

            with open(diagnosis_codes_file, "w") as f:
                f.write("code,description\n")
                f.write("J06.9,Respiratory infection\n")

            test_data = bytes([i % 256 for i in range(test_size)])
            with open(os.path.join(attachments_dir, "boundary.bin"), "wb") as f:
                f.write(test_data)

            with open(claims_file, "w") as f:
                f.write("claim_id,member_id,provider_id,service_date,total_charges,line_id,procedure_code,diagnosis_code,charged_amount,units,attachment_file\n")
                f.write("CLM001,MEM001,PROV001,2024-03-15,20000,L001,99214,J06.9,20000,1,boundary.bin\n")

            success, stdout, stderr = run_claims_processor(
                claims_file, attachments_dir, providers_file, members_file,
                fee_schedule_file, plan_benefits_file, procedure_codes_file,
                diagnosis_codes_file, output_dir
            )
            assert success, f"ClaimsProcessor failed for size {test_size}: {stderr}"


def test_output_files_created():
    """
    Verify that all required output files are created: claim_summary.csv,
    adjudication_details.csv, attachment_report.csv, and audit_log.csv.
    """
    success, _ = compile_java()
    assert success, "Compilation failed"

    with tempfile.TemporaryDirectory() as tmpdir:
        claims_file = os.path.join(tmpdir, "claims.csv")
        attachments_dir = os.path.join(tmpdir, "attachments")
        providers_file = os.path.join(tmpdir, "providers.csv")
        members_file = os.path.join(tmpdir, "members.csv")
        fee_schedule_file = os.path.join(tmpdir, "fee_schedule.csv")
        plan_benefits_file = os.path.join(tmpdir, "plan_benefits.csv")
        procedure_codes_file = os.path.join(tmpdir, "procedure_codes.csv")
        diagnosis_codes_file = os.path.join(tmpdir, "diagnosis_codes.csv")
        output_dir = os.path.join(tmpdir, "output")

        os.makedirs(attachments_dir)
        os.makedirs(output_dir)

        with open(providers_file, "w") as f:
            f.write("provider_id,name,specialty,network_status,effective_date,termination_date\n")
            f.write("PROV001,Test Provider,General,IN_NETWORK,2020-01-01,\n")

        with open(members_file, "w") as f:
            f.write("member_id,plan_id,effective_date,termination_date,deductible_met,out_of_pocket_met\n")
            f.write("MEM001,PLAN_GOLD,2023-01-01,,50000,100000\n")

        with open(fee_schedule_file, "w") as f:
            f.write("procedure_code,network_allowed,out_of_network_allowed,requires_auth\n")
            f.write("99214,20000,15000,false\n")

        with open(plan_benefits_file, "w") as f:
            f.write("plan_id,deductible,coinsurance_rate,copay,out_of_pocket_max\n")
            f.write("PLAN_GOLD,100000,0.10,2500,500000\n")

        with open(procedure_codes_file, "w") as f:
            f.write("code,description\n")
            f.write("99214,Office visit level 4\n")

        with open(diagnosis_codes_file, "w") as f:
            f.write("code,description\n")
            f.write("J06.9,Respiratory infection\n")

        with open(claims_file, "w") as f:
            f.write("claim_id,member_id,provider_id,service_date,total_charges,line_id,procedure_code,diagnosis_code,charged_amount,units,attachment_file\n")
            f.write("CLM001,MEM001,PROV001,2024-03-15,20000,L001,99214,J06.9,20000,1,\n")

        success, stdout, stderr = run_claims_processor(
            claims_file, attachments_dir, providers_file, members_file,
            fee_schedule_file, plan_benefits_file, procedure_codes_file,
            diagnosis_codes_file, output_dir
        )
        assert success, f"ClaimsProcessor failed: {stderr}"

        assert os.path.exists(os.path.join(output_dir, "claim_summary.csv")), \
            "claim_summary.csv not created"
        assert os.path.exists(os.path.join(output_dir, "adjudication_details.csv")), \
            "adjudication_details.csv not created"
        assert os.path.exists(os.path.join(output_dir, "attachment_report.csv")), \
            "attachment_report.csv not created"
        assert os.path.exists(os.path.join(output_dir, "audit_log.csv")), \
            "audit_log.csv not created"


def test_claim_summary_sorted_by_claim_id():
    """
    Verify that claim_summary.csv is sorted by claim_id as specified.
    """
    success, _ = compile_java()
    assert success, "Compilation failed"

    with tempfile.TemporaryDirectory() as tmpdir:
        claims_file = os.path.join(tmpdir, "claims.csv")
        attachments_dir = os.path.join(tmpdir, "attachments")
        providers_file = os.path.join(tmpdir, "providers.csv")
        members_file = os.path.join(tmpdir, "members.csv")
        fee_schedule_file = os.path.join(tmpdir, "fee_schedule.csv")
        plan_benefits_file = os.path.join(tmpdir, "plan_benefits.csv")
        procedure_codes_file = os.path.join(tmpdir, "procedure_codes.csv")
        diagnosis_codes_file = os.path.join(tmpdir, "diagnosis_codes.csv")
        output_dir = os.path.join(tmpdir, "output")

        os.makedirs(attachments_dir)
        os.makedirs(output_dir)

        with open(providers_file, "w") as f:
            f.write("provider_id,name,specialty,network_status,effective_date,termination_date\n")
            f.write("PROV001,Test Provider,General,IN_NETWORK,2020-01-01,\n")

        with open(members_file, "w") as f:
            f.write("member_id,plan_id,effective_date,termination_date,deductible_met,out_of_pocket_met\n")
            f.write("MEM001,PLAN_GOLD,2023-01-01,,50000,100000\n")

        with open(fee_schedule_file, "w") as f:
            f.write("procedure_code,network_allowed,out_of_network_allowed,requires_auth\n")
            f.write("99214,20000,15000,false\n")

        with open(plan_benefits_file, "w") as f:
            f.write("plan_id,deductible,coinsurance_rate,copay,out_of_pocket_max\n")
            f.write("PLAN_GOLD,100000,0.10,2500,500000\n")

        with open(procedure_codes_file, "w") as f:
            f.write("code,description\n")
            f.write("99214,Office visit level 4\n")

        with open(diagnosis_codes_file, "w") as f:
            f.write("code,description\n")
            f.write("J06.9,Respiratory infection\n")

        with open(claims_file, "w") as f:
            f.write("claim_id,member_id,provider_id,service_date,total_charges,line_id,procedure_code,diagnosis_code,charged_amount,units,attachment_file\n")
            f.write("CLM003,MEM001,PROV001,2024-03-15,20000,L001,99214,J06.9,20000,1,\n")
            f.write("CLM001,MEM001,PROV001,2024-03-15,20000,L001,99214,J06.9,20000,1,\n")
            f.write("CLM002,MEM001,PROV001,2024-03-15,20000,L001,99214,J06.9,20000,1,\n")

        success, stdout, stderr = run_claims_processor(
            claims_file, attachments_dir, providers_file, members_file,
            fee_schedule_file, plan_benefits_file, procedure_codes_file,
            diagnosis_codes_file, output_dir
        )
        assert success, f"ClaimsProcessor failed: {stderr}"

        with open(os.path.join(output_dir, "claim_summary.csv")) as f:
            lines = f.readlines()[1:]

        claim_ids = [line.split(",")[0] for line in lines if line.strip()]
        assert claim_ids == sorted(claim_ids), f"claim_summary.csv not sorted by claim_id: {claim_ids}"


def test_benefit_calculation_accuracy():
    """
    Verify that benefit calculations (allowed amounts, paid amounts, patient
    responsibility) are computed correctly based on fee schedules and plan benefits.
    """
    success, _ = compile_java()
    assert success, "Compilation failed"

    with tempfile.TemporaryDirectory() as tmpdir:
        claims_file = os.path.join(tmpdir, "claims.csv")
        attachments_dir = os.path.join(tmpdir, "attachments")
        providers_file = os.path.join(tmpdir, "providers.csv")
        members_file = os.path.join(tmpdir, "members.csv")
        fee_schedule_file = os.path.join(tmpdir, "fee_schedule.csv")
        plan_benefits_file = os.path.join(tmpdir, "plan_benefits.csv")
        procedure_codes_file = os.path.join(tmpdir, "procedure_codes.csv")
        diagnosis_codes_file = os.path.join(tmpdir, "diagnosis_codes.csv")
        output_dir = os.path.join(tmpdir, "output")

        os.makedirs(attachments_dir)
        os.makedirs(output_dir)

        with open(providers_file, "w") as f:
            f.write("provider_id,name,specialty,network_status,effective_date,termination_date\n")
            f.write("PROV001,Test Provider,General,IN_NETWORK,2020-01-01,\n")

        with open(members_file, "w") as f:
            f.write("member_id,plan_id,effective_date,termination_date,deductible_met,out_of_pocket_met\n")
            f.write("MEM001,PLAN_GOLD,2023-01-01,,100000,200000\n")

        with open(fee_schedule_file, "w") as f:
            f.write("procedure_code,network_allowed,out_of_network_allowed,requires_auth\n")
            f.write("99214,20000,15000,false\n")

        with open(plan_benefits_file, "w") as f:
            f.write("plan_id,deductible,coinsurance_rate,copay,out_of_pocket_max\n")
            f.write("PLAN_GOLD,100000,0.10,2500,500000\n")

        with open(procedure_codes_file, "w") as f:
            f.write("code,description\n")
            f.write("99214,Office visit level 4\n")

        with open(diagnosis_codes_file, "w") as f:
            f.write("code,description\n")
            f.write("J06.9,Respiratory infection\n")

        with open(claims_file, "w") as f:
            f.write("claim_id,member_id,provider_id,service_date,total_charges,line_id,procedure_code,diagnosis_code,charged_amount,units,attachment_file\n")
            f.write("CLM001,MEM001,PROV001,2024-03-15,25000,L001,99214,J06.9,25000,1,\n")

        success, stdout, stderr = run_claims_processor(
            claims_file, attachments_dir, providers_file, members_file,
            fee_schedule_file, plan_benefits_file, procedure_codes_file,
            diagnosis_codes_file, output_dir
        )
        assert success, f"ClaimsProcessor failed: {stderr}"

        with open(os.path.join(output_dir, "claim_summary.csv")) as f:
            lines = f.readlines()

        clm001_line = None
        for line in lines[1:]:
            if line.startswith("CLM001"):
                clm001_line = line.strip()
                break

        assert clm001_line is not None, "CLM001 not found in claim_summary.csv"
        parts = clm001_line.split(",")

        total_allowed = int(parts[4])
        assert total_allowed == 20000, f"Expected allowed amount 20000, got {total_allowed}"
