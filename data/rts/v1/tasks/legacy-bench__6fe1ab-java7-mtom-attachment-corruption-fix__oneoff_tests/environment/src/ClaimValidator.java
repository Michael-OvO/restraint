import java.util.*;
import java.text.*;

public class ClaimValidator {

    private Set<String> validProcedureCodes;
    private Set<String> validDiagnosisCodes;
    private Map<String, ProviderInfo> providers;
    private Map<String, MemberInfo> members;

    public ClaimValidator() {
        this.validProcedureCodes = new HashSet<String>();
        this.validDiagnosisCodes = new HashSet<String>();
        this.providers = new HashMap<String, ProviderInfo>();
        this.members = new HashMap<String, MemberInfo>();
    }

    public void loadProcedureCodes(String path) throws Exception {
        java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(path));
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (!line.isEmpty() && !line.startsWith("#")) {
                validProcedureCodes.add(line.split(",")[0]);
            }
        }
        br.close();
    }

    public void loadDiagnosisCodes(String path) throws Exception {
        java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(path));
        String line;
        while ((line = br.readLine()) != null) {
            line = line.trim();
            if (!line.isEmpty() && !line.startsWith("#")) {
                validDiagnosisCodes.add(line.split(",")[0]);
            }
        }
        br.close();
    }

    public void loadProviders(String path) throws Exception {
        java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(path));
        String line = br.readLine();
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            ProviderInfo info = new ProviderInfo();
            info.providerId = parts[0];
            info.name = parts[1];
            info.specialty = parts[2];
            info.networkStatus = parts[3];
            info.effectiveDate = parts[4];
            info.terminationDate = parts.length > 5 ? parts[5] : "";
            providers.put(info.providerId, info);
        }
        br.close();
    }

    public void loadMembers(String path) throws Exception {
        java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(path));
        String line = br.readLine();
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            MemberInfo info = new MemberInfo();
            info.memberId = parts[0];
            info.planId = parts[1];
            info.effectiveDate = parts[2];
            info.terminationDate = parts.length > 3 ? parts[3] : "";
            info.deductibleMet = Integer.parseInt(parts.length > 4 ? parts[4] : "0");
            info.outOfPocketMet = Integer.parseInt(parts.length > 5 ? parts[5] : "0");
            members.put(info.memberId, info);
        }
        br.close();
    }

    public ValidationResult validateClaim(ClaimRequest claim) {
        ValidationResult result = new ValidationResult();
        result.isValid = true;

        if (claim.claimId == null || claim.claimId.isEmpty()) {
            result.isValid = false;
            result.errorCode = "MISSING_CLAIM_ID";
            result.errorMessage = "Claim ID is required";
            return result;
        }

        if (claim.memberId == null || claim.memberId.isEmpty()) {
            result.isValid = false;
            result.errorCode = "MISSING_MEMBER_ID";
            result.errorMessage = "Member ID is required";
            return result;
        }

        if (claim.providerId == null || claim.providerId.isEmpty()) {
            result.isValid = false;
            result.errorCode = "MISSING_PROVIDER_ID";
            result.errorMessage = "Provider ID is required";
            return result;
        }

        MemberInfo member = members.get(claim.memberId);
        if (member == null) {
            result.isValid = false;
            result.errorCode = "INVALID_MEMBER";
            result.errorMessage = "Member not found: " + claim.memberId;
            return result;
        }

        ProviderInfo provider = providers.get(claim.providerId);
        if (provider == null) {
            result.isValid = false;
            result.errorCode = "INVALID_PROVIDER";
            result.errorMessage = "Provider not found: " + claim.providerId;
            return result;
        }

        if (!isDateWithinRange(claim.serviceDate, member.effectiveDate, member.terminationDate)) {
            result.isValid = false;
            result.errorCode = "MEMBER_NOT_COVERED";
            result.errorMessage = "Member not covered on service date";
            return result;
        }

        if (claim.lines == null || claim.lines.isEmpty()) {
            result.isValid = false;
            result.errorCode = "NO_CLAIM_LINES";
            result.errorMessage = "At least one claim line is required";
            return result;
        }

        for (ClaimLine line : claim.lines) {
            if (!validProcedureCodes.isEmpty() && !validProcedureCodes.contains(line.procedureCode)) {
                result.isValid = false;
                result.errorCode = "INVALID_PROCEDURE";
                result.errorMessage = "Invalid procedure code: " + line.procedureCode;
                return result;
            }

            if (!validDiagnosisCodes.isEmpty() && !validDiagnosisCodes.contains(line.diagnosisCode)) {
                result.isValid = false;
                result.errorCode = "INVALID_DIAGNOSIS";
                result.errorMessage = "Invalid diagnosis code: " + line.diagnosisCode;
                return result;
            }
        }

        return result;
    }

    private boolean isDateWithinRange(String testDate, String startDate, String endDate) {
        if (testDate == null || startDate == null) {
            return false;
        }
        if (testDate.compareTo(startDate) < 0) {
            return false;
        }
        if (endDate != null && !endDate.isEmpty() && testDate.compareTo(endDate) > 0) {
            return false;
        }
        return true;
    }

    public boolean isProcedureValid(String code) {
        return validProcedureCodes.isEmpty() || validProcedureCodes.contains(code);
    }

    public boolean isDiagnosisValid(String code) {
        return validDiagnosisCodes.isEmpty() || validDiagnosisCodes.contains(code);
    }

    public MemberInfo getMember(String memberId) {
        return members.get(memberId);
    }

    public ProviderInfo getProvider(String providerId) {
        return providers.get(providerId);
    }
}

class ProviderInfo {
    String providerId;
    String name;
    String specialty;
    String networkStatus;
    String effectiveDate;
    String terminationDate;
}

class MemberInfo {
    String memberId;
    String planId;
    String effectiveDate;
    String terminationDate;
    int deductibleMet;
    int outOfPocketMet;
}
