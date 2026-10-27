import java.io.*;
import java.util.*;

public class ClaimsProcessor {

    public static void main(String[] args) throws Exception {
        String claimsFile = "/app/data/claims.csv";
        String attachmentsDir = "/app/data/attachments";
        String providersFile = "/app/data/providers.csv";
        String membersFile = "/app/data/members.csv";
        String feeScheduleFile = "/app/data/fee_schedule.csv";
        String planBenefitsFile = "/app/data/plan_benefits.csv";
        String procedureCodesFile = "/app/data/procedure_codes.csv";
        String diagnosisCodesFile = "/app/data/diagnosis_codes.csv";
        String outputDir = "/app/output";

        if (args.length >= 9) {
            claimsFile = args[0];
            attachmentsDir = args[1];
            providersFile = args[2];
            membersFile = args[3];
            feeScheduleFile = args[4];
            planBenefitsFile = args[5];
            procedureCodesFile = args[6];
            diagnosisCodesFile = args[7];
            outputDir = args[8];
        }

        new File(outputDir).mkdirs();

        ClaimValidator validator = new ClaimValidator();
        validator.loadProviders(providersFile);
        validator.loadMembers(membersFile);
        validator.loadProcedureCodes(procedureCodesFile);
        validator.loadDiagnosisCodes(diagnosisCodesFile);

        BenefitCalculator benefitCalc = new BenefitCalculator();
        benefitCalc.loadFeeSchedules(feeScheduleFile);
        benefitCalc.loadPlanBenefits(planBenefitsFile);
        benefitCalc.setValidator(validator);

        AttachmentProcessor attachmentProcessor = new AttachmentProcessor();
        AuditLogger auditLogger = new AuditLogger();

        ClaimsServiceImpl service = new ClaimsServiceImpl(
            validator, attachmentProcessor, benefitCalc, auditLogger
        );

        List<ClaimRequest> claims = loadClaims(claimsFile, attachmentsDir);

        List<ClaimResponse> responses = new ArrayList<ClaimResponse>();
        for (ClaimRequest claim : claims) {
            ClaimResponse response = service.processClaim(claim);
            responses.add(response);
        }

        writeClaimSummary(responses, outputDir + "/claim_summary.csv");
        writeAdjudicationDetails(responses, outputDir + "/adjudication_details.csv");
        writeAttachmentReport(responses, outputDir + "/attachment_report.csv");

        auditLogger.writeLog(outputDir + "/audit_log.csv");
    }

    private static List<ClaimRequest> loadClaims(String claimsFile, String attachmentsDir)
            throws Exception {
        List<ClaimRequest> claims = new ArrayList<ClaimRequest>();
        Map<String, List<ClaimLine>> linesByClaimId = new HashMap<String, List<ClaimLine>>();
        Map<String, ClaimRequest> claimById = new HashMap<String, ClaimRequest>();

        BufferedReader br = new BufferedReader(new FileReader(claimsFile));
        String headerLine = br.readLine();
        String line;

        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            String claimId = parts[0];

            ClaimRequest claim = claimById.get(claimId);
            if (claim == null) {
                claim = new ClaimRequest();
                claim.claimId = claimId;
                claim.memberId = parts[1];
                claim.providerId = parts[2];
                claim.serviceDate = parts[3];
                claim.totalCharges = Integer.parseInt(parts[4]);
                claim.lines = new ArrayList<ClaimLine>();
                claim.attachments = new ArrayList<BinaryAttachment>();
                claimById.put(claimId, claim);
                claims.add(claim);
            }

            ClaimLine claimLine = new ClaimLine();
            claimLine.lineId = parts[5];
            claimLine.procedureCode = parts[6];
            claimLine.diagnosisCode = parts[7];
            claimLine.chargedAmount = Integer.parseInt(parts[8]);
            claimLine.units = Integer.parseInt(parts[9]);
            claim.lines.add(claimLine);

            if (parts.length > 10 && !parts[10].isEmpty()) {
                String attachmentFile = parts[10];
                BinaryAttachment att = loadAttachment(attachmentsDir, attachmentFile);
                if (att != null) {
                    claim.attachments.add(att);
                }
            }
        }
        br.close();

        return claims;
    }

    private static BinaryAttachment loadAttachment(String dir, String filename) {
        try {
            File f = new File(dir, filename);
            if (!f.exists()) {
                return null;
            }

            BinaryAttachment att = new BinaryAttachment();
            att.attachmentId = filename;
            att.filename = filename;

            if (filename.endsWith(".pdf")) {
                att.contentType = "application/pdf";
            } else if (filename.endsWith(".jpg") || filename.endsWith(".jpeg")) {
                att.contentType = "image/jpeg";
            } else if (filename.endsWith(".png")) {
                att.contentType = "image/png";
            } else {
                att.contentType = "application/octet-stream";
            }

            FileInputStream fis = new FileInputStream(f);
            byte[] data = new byte[(int) f.length()];
            fis.read(data);
            fis.close();

            att.data = data;
            return att;

        } catch (Exception e) {
            return null;
        }
    }

    private static void writeClaimSummary(List<ClaimResponse> responses, String path)
            throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(path));
        pw.println("claim_id,status,error_code,total_charges,total_allowed,total_paid,patient_responsibility,attachment_count,attachment_checksum");

        Collections.sort(responses, new Comparator<ClaimResponse>() {
            public int compare(ClaimResponse a, ClaimResponse b) {
                return a.claimId.compareTo(b.claimId);
            }
        });

        for (ClaimResponse r : responses) {
            pw.println(r.claimId + "," +
                       r.status + "," +
                       (r.errorCode != null ? r.errorCode : "") + "," +
                       r.totalCharges + "," +
                       r.totalAllowed + "," +
                       r.totalPaid + "," +
                       r.totalPatientResponsibility + "," +
                       r.attachmentCount + "," +
                       r.attachmentChecksum);
        }
        pw.close();
    }

    private static void writeAdjudicationDetails(List<ClaimResponse> responses, String path)
            throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(path));
        pw.println("claim_id,line_id,procedure_code,charged_amount,allowed_amount,paid_amount,patient_responsibility,adjustment_reason");

        List<String[]> rows = new ArrayList<String[]>();
        for (ClaimResponse r : responses) {
            if (r.adjudicationDetails != null) {
                for (AdjudicationDetail d : r.adjudicationDetails) {
                    String[] row = new String[] {
                        r.claimId,
                        d.lineId,
                        d.procedureCode,
                        String.valueOf(d.chargedAmount),
                        String.valueOf(d.allowedAmount),
                        String.valueOf(d.paidAmount),
                        String.valueOf(d.patientResponsibility),
                        d.adjustmentReason != null ? d.adjustmentReason : ""
                    };
                    rows.add(row);
                }
            }
        }

        Collections.sort(rows, new Comparator<String[]>() {
            public int compare(String[] a, String[] b) {
                int cmp = a[0].compareTo(b[0]);
                if (cmp != 0) return cmp;
                return a[1].compareTo(b[1]);
            }
        });

        for (String[] row : rows) {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < row.length; i++) {
                if (i > 0) sb.append(",");
                sb.append(row[i]);
            }
            pw.println(sb.toString());
        }
        pw.close();
    }

    private static void writeAttachmentReport(List<ClaimResponse> responses, String path)
            throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(path));
        pw.println("claim_id,attachment_count,total_checksum");

        Collections.sort(responses, new Comparator<ClaimResponse>() {
            public int compare(ClaimResponse a, ClaimResponse b) {
                return a.claimId.compareTo(b.claimId);
            }
        });

        for (ClaimResponse r : responses) {
            if (r.attachmentCount > 0) {
                pw.println(r.claimId + "," + r.attachmentCount + "," + r.attachmentChecksum);
            }
        }
        pw.close();
    }
}
