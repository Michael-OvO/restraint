import javax.jws.WebService;
import javax.jws.WebMethod;
import javax.xml.ws.BindingType;
import javax.xml.ws.soap.SOAPBinding;
import java.io.*;
import java.util.*;

@WebService(name = "ClaimsService", serviceName = "ClaimsServicePort")
@BindingType(value = SOAPBinding.SOAP11HTTP_MTOM_BINDING)
public class ClaimsServiceImpl {

    private ClaimValidator validator;
    private AttachmentProcessor attachmentProcessor;
    private BenefitCalculator benefitCalc;
    private AuditLogger auditLogger;

    public ClaimsServiceImpl() {
        this.validator = new ClaimValidator();
        this.attachmentProcessor = new AttachmentProcessor();
        this.benefitCalc = new BenefitCalculator();
        this.auditLogger = new AuditLogger();
    }

    public ClaimsServiceImpl(ClaimValidator v, AttachmentProcessor ap,
                             BenefitCalculator bc, AuditLogger al) {
        this.validator = v;
        this.attachmentProcessor = ap;
        this.benefitCalc = bc;
        this.auditLogger = al;
    }

    @WebMethod
    public ClaimResponse processClaim(ClaimRequest request) {
        ClaimResponse response = new ClaimResponse();
        response.claimId = request.claimId;
        response.status = "PENDING";
        response.adjudicationDetails = new ArrayList<AdjudicationDetail>();

        try {
            ValidationResult validation = validator.validateClaim(request);
            if (!validation.isValid) {
                response.status = "REJECTED";
                response.errorCode = validation.errorCode;
                response.errorMessage = validation.errorMessage;
                auditLogger.logClaimStatus(request.claimId, "REJECTED", validation.errorCode);
                return response;
            }

            List<ProcessedAttachment> processedDocs = new ArrayList<ProcessedAttachment>();
            if (request.attachments != null) {
                for (BinaryAttachment attachment : request.attachments) {
                    ProcessedAttachment processed = attachmentProcessor.processAttachment(attachment);
                    processedDocs.add(processed);
                }
            }

            int totalAllowed = 0;
            int totalPaid = 0;
            int totalPatientResponsibility = 0;

            for (ClaimLine line : request.lines) {
                AdjudicationDetail detail = benefitCalc.adjudicateLine(
                    line, request.providerId, request.memberId, request.serviceDate
                );
                response.adjudicationDetails.add(detail);
                totalAllowed += detail.allowedAmount;
                totalPaid += detail.paidAmount;
                totalPatientResponsibility += detail.patientResponsibility;
            }

            response.totalCharges = request.totalCharges;
            response.totalAllowed = totalAllowed;
            response.totalPaid = totalPaid;
            response.totalPatientResponsibility = totalPatientResponsibility;
            response.attachmentCount = processedDocs.size();
            response.attachmentChecksum = computeAttachmentChecksum(processedDocs);
            response.status = "PROCESSED";

            auditLogger.logClaimStatus(request.claimId, "PROCESSED", null);

        } catch (Exception e) {
            response.status = "ERROR";
            response.errorCode = "SYSTEM_ERROR";
            response.errorMessage = e.getMessage();
            auditLogger.logClaimStatus(request.claimId, "ERROR", "SYSTEM_ERROR");
        }

        return response;
    }

    @WebMethod
    public BatchResponse processBatch(BatchRequest batch) {
        BatchResponse response = new BatchResponse();
        response.batchId = batch.batchId;
        response.submittedCount = batch.claims.size();
        response.results = new ArrayList<ClaimResponse>();

        int processedCount = 0;
        int rejectedCount = 0;
        int errorCount = 0;

        for (ClaimRequest claim : batch.claims) {
            ClaimResponse result = processClaim(claim);
            response.results.add(result);

            if ("PROCESSED".equals(result.status)) {
                processedCount++;
            } else if ("REJECTED".equals(result.status)) {
                rejectedCount++;
            } else {
                errorCount++;
            }
        }

        response.processedCount = processedCount;
        response.rejectedCount = rejectedCount;
        response.errorCount = errorCount;

        return response;
    }

    private long computeAttachmentChecksum(List<ProcessedAttachment> attachments) {
        long checksum = 0;
        for (ProcessedAttachment att : attachments) {
            checksum ^= att.contentHash;
            checksum += att.processedSize;
        }
        return checksum;
    }
}

class ClaimRequest {
    String claimId;
    String memberId;
    String providerId;
    String serviceDate;
    int totalCharges;
    List<ClaimLine> lines;
    List<BinaryAttachment> attachments;
}

class ClaimLine {
    String lineId;
    String procedureCode;
    String diagnosisCode;
    int chargedAmount;
    int units;
}

class BinaryAttachment {
    String attachmentId;
    String contentType;
    String filename;
    byte[] data;
}

class ClaimResponse {
    String claimId;
    String status;
    String errorCode;
    String errorMessage;
    int totalCharges;
    int totalAllowed;
    int totalPaid;
    int totalPatientResponsibility;
    int attachmentCount;
    long attachmentChecksum;
    List<AdjudicationDetail> adjudicationDetails;
}

class AdjudicationDetail {
    String lineId;
    String procedureCode;
    int chargedAmount;
    int allowedAmount;
    int paidAmount;
    int patientResponsibility;
    String adjustmentReason;
}

class BatchRequest {
    String batchId;
    String submitterId;
    List<ClaimRequest> claims;
}

class BatchResponse {
    String batchId;
    int submittedCount;
    int processedCount;
    int rejectedCount;
    int errorCount;
    List<ClaimResponse> results;
}

class ProcessedAttachment {
    String attachmentId;
    int originalSize;
    int processedSize;
    long contentHash;
    boolean valid;
}

class ValidationResult {
    boolean isValid;
    String errorCode;
    String errorMessage;
}
