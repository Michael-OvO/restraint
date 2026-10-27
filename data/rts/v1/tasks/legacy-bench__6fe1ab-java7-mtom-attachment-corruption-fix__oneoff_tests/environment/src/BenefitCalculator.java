import java.util.*;

public class BenefitCalculator {

    private Map<String, FeeSchedule> feeSchedules;
    private Map<String, PlanBenefits> planBenefits;
    private ClaimValidator validator;

    public BenefitCalculator() {
        this.feeSchedules = new HashMap<String, FeeSchedule>();
        this.planBenefits = new HashMap<String, PlanBenefits>();
    }

    public void setValidator(ClaimValidator validator) {
        this.validator = validator;
    }

    public void loadFeeSchedules(String path) throws Exception {
        java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(path));
        String line = br.readLine();
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            FeeSchedule fs = new FeeSchedule();
            fs.procedureCode = parts[0];
            fs.networkAllowed = Integer.parseInt(parts[1]);
            fs.outOfNetworkAllowed = Integer.parseInt(parts[2]);
            fs.requiresAuth = "true".equalsIgnoreCase(parts.length > 3 ? parts[3] : "false");
            feeSchedules.put(fs.procedureCode, fs);
        }
        br.close();
    }

    public void loadPlanBenefits(String path) throws Exception {
        java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(path));
        String line = br.readLine();
        while ((line = br.readLine()) != null) {
            String[] parts = line.split(",");
            PlanBenefits pb = new PlanBenefits();
            pb.planId = parts[0];
            pb.deductible = Integer.parseInt(parts[1]);
            pb.coinsuranceRate = Double.parseDouble(parts[2]);
            pb.copay = Integer.parseInt(parts[3]);
            pb.outOfPocketMax = Integer.parseInt(parts[4]);
            planBenefits.put(pb.planId, pb);
        }
        br.close();
    }

    public AdjudicationDetail adjudicateLine(ClaimLine line, String providerId,
                                              String memberId, String serviceDate) {
        AdjudicationDetail detail = new AdjudicationDetail();
        detail.lineId = line.lineId;
        detail.procedureCode = line.procedureCode;
        detail.chargedAmount = line.chargedAmount;

        FeeSchedule fs = feeSchedules.get(line.procedureCode);
        if (fs == null) {
            detail.allowedAmount = line.chargedAmount;
            detail.adjustmentReason = "NO_FEE_SCHEDULE";
        } else {
            ProviderInfo provider = validator != null ? validator.getProvider(providerId) : null;
            boolean inNetwork = provider != null && "IN_NETWORK".equals(provider.networkStatus);

            int allowed = inNetwork ? fs.networkAllowed : fs.outOfNetworkAllowed;
            detail.allowedAmount = Math.min(allowed * line.units, line.chargedAmount);

            if (detail.allowedAmount < line.chargedAmount) {
                detail.adjustmentReason = "FEE_SCHEDULE_REDUCTION";
            }
        }

        MemberInfo member = validator != null ? validator.getMember(memberId) : null;
        PlanBenefits plan = member != null ? planBenefits.get(member.planId) : null;

        if (plan != null) {
            int deductibleRemaining = Math.max(0, plan.deductible - member.deductibleMet);
            int appliedToDeductible = Math.min(deductibleRemaining, detail.allowedAmount);

            int afterDeductible = detail.allowedAmount - appliedToDeductible;
            int coinsurance = (int)(afterDeductible * plan.coinsuranceRate);
            int planPays = afterDeductible - coinsurance;

            int oopRemaining = Math.max(0, plan.outOfPocketMax - member.outOfPocketMet);
            int patientTotal = appliedToDeductible + coinsurance;

            if (patientTotal > oopRemaining) {
                patientTotal = oopRemaining;
                planPays = detail.allowedAmount - patientTotal;
            }

            detail.paidAmount = planPays;
            detail.patientResponsibility = patientTotal;
        } else {
            detail.paidAmount = 0;
            detail.patientResponsibility = detail.allowedAmount;
            if (detail.adjustmentReason == null) {
                detail.adjustmentReason = "NO_PLAN_FOUND";
            }
        }

        return detail;
    }

    public int calculateTotalAllowed(List<ClaimLine> lines, String providerId) {
        int total = 0;
        for (ClaimLine line : lines) {
            FeeSchedule fs = feeSchedules.get(line.procedureCode);
            if (fs != null) {
                ProviderInfo provider = validator != null ? validator.getProvider(providerId) : null;
                boolean inNetwork = provider != null && "IN_NETWORK".equals(provider.networkStatus);
                int allowed = inNetwork ? fs.networkAllowed : fs.outOfNetworkAllowed;
                total += Math.min(allowed * line.units, line.chargedAmount);
            } else {
                total += line.chargedAmount;
            }
        }
        return total;
    }

    public FeeSchedule getFeeSchedule(String procedureCode) {
        return feeSchedules.get(procedureCode);
    }

    public PlanBenefits getPlanBenefits(String planId) {
        return planBenefits.get(planId);
    }
}

class FeeSchedule {
    String procedureCode;
    int networkAllowed;
    int outOfNetworkAllowed;
    boolean requiresAuth;
}

class PlanBenefits {
    String planId;
    int deductible;
    double coinsuranceRate;
    int copay;
    int outOfPocketMax;
}
