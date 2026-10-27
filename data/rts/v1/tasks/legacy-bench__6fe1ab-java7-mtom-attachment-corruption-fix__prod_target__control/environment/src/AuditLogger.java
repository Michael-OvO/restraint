import java.io.*;
import java.util.*;

public class AuditLogger {

    private List<AuditEntry> entries;
    private String logFilePath;
    private boolean writeImmediately;

    public AuditLogger() {
        this.entries = new ArrayList<AuditEntry>();
        this.writeImmediately = false;
    }

    public AuditLogger(String logFilePath) {
        this();
        this.logFilePath = logFilePath;
        this.writeImmediately = true;
    }

    public void logClaimStatus(String claimId, String status, String errorCode) {
        AuditEntry entry = new AuditEntry();
        entry.timestamp = System.currentTimeMillis();
        entry.claimId = claimId;
        entry.status = status;
        entry.errorCode = errorCode;
        entry.action = "STATUS_CHANGE";

        entries.add(entry);

        if (writeImmediately && logFilePath != null) {
            appendToFile(entry);
        }
    }

    public void logAttachmentProcessing(String claimId, String attachmentId,
                                         int originalSize, int processedSize) {
        AuditEntry entry = new AuditEntry();
        entry.timestamp = System.currentTimeMillis();
        entry.claimId = claimId;
        entry.attachmentId = attachmentId;
        entry.action = "ATTACHMENT_PROCESSED";
        entry.details = "original=" + originalSize + ",processed=" + processedSize;

        entries.add(entry);

        if (writeImmediately && logFilePath != null) {
            appendToFile(entry);
        }
    }

    public void logBatchStart(String batchId, int claimCount) {
        AuditEntry entry = new AuditEntry();
        entry.timestamp = System.currentTimeMillis();
        entry.batchId = batchId;
        entry.action = "BATCH_START";
        entry.details = "claim_count=" + claimCount;

        entries.add(entry);

        if (writeImmediately && logFilePath != null) {
            appendToFile(entry);
        }
    }

    public void logBatchComplete(String batchId, int processed, int rejected, int errors) {
        AuditEntry entry = new AuditEntry();
        entry.timestamp = System.currentTimeMillis();
        entry.batchId = batchId;
        entry.action = "BATCH_COMPLETE";
        entry.details = "processed=" + processed + ",rejected=" + rejected + ",errors=" + errors;

        entries.add(entry);

        if (writeImmediately && logFilePath != null) {
            appendToFile(entry);
        }
    }

    private void appendToFile(AuditEntry entry) {
        try {
            PrintWriter pw = new PrintWriter(new FileWriter(logFilePath, true));
            pw.println(formatEntry(entry));
            pw.close();
        } catch (IOException e) {
            // Silent fail for audit logging
        }
    }

    private String formatEntry(AuditEntry entry) {
        StringBuilder sb = new StringBuilder();
        sb.append(entry.timestamp).append(",");
        sb.append(entry.action).append(",");
        sb.append(entry.claimId != null ? entry.claimId : "").append(",");
        sb.append(entry.batchId != null ? entry.batchId : "").append(",");
        sb.append(entry.attachmentId != null ? entry.attachmentId : "").append(",");
        sb.append(entry.status != null ? entry.status : "").append(",");
        sb.append(entry.errorCode != null ? entry.errorCode : "").append(",");
        sb.append(entry.details != null ? entry.details : "");
        return sb.toString();
    }

    public void writeLog(String path) throws Exception {
        PrintWriter pw = new PrintWriter(new FileWriter(path));
        pw.println("timestamp,action,claim_id,batch_id,attachment_id,status,error_code,details");
        for (AuditEntry entry : entries) {
            pw.println(formatEntry(entry));
        }
        pw.close();
    }

    public List<AuditEntry> getEntries() {
        return new ArrayList<AuditEntry>(entries);
    }

    public int getEntryCount() {
        return entries.size();
    }

    public List<AuditEntry> getEntriesForClaim(String claimId) {
        List<AuditEntry> result = new ArrayList<AuditEntry>();
        for (AuditEntry entry : entries) {
            if (claimId.equals(entry.claimId)) {
                result.add(entry);
            }
        }
        return result;
    }
}

class AuditEntry {
    long timestamp;
    String action;
    String claimId;
    String batchId;
    String attachmentId;
    String status;
    String errorCode;
    String details;
}
