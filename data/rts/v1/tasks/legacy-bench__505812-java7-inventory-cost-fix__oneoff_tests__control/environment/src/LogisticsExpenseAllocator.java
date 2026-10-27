import java.util.*;

public class LogisticsExpenseAllocator {

    private static final Map<String, Double> SKU_WEIGHTS = new HashMap<String, Double>();
    private static final Map<String, Double> SKU_VOLUMES = new HashMap<String, Double>();

    static {
        SKU_WEIGHTS.put("SKU001", 2.5);
        SKU_WEIGHTS.put("SKU002", 1.2);
        SKU_WEIGHTS.put("SKU003", 5.0);
        SKU_WEIGHTS.put("SKU004", 0.8);
        SKU_WEIGHTS.put("SKU005", 3.3);

        SKU_VOLUMES.put("SKU001", 0.015);
        SKU_VOLUMES.put("SKU002", 0.008);
        SKU_VOLUMES.put("SKU003", 0.030);
        SKU_VOLUMES.put("SKU004", 0.005);
        SKU_VOLUMES.put("SKU005", 0.020);
    }

    public Map<String, Double> allocate(List<String> skus, double totalExpense, String basis) {
        Map<String, Double> allocations = new HashMap<String, Double>();

        Map<String, Double> basisValues;
        if (basis.equals("V")) {
            basisValues = SKU_WEIGHTS;
        } else {
            basisValues = SKU_VOLUMES;
        }

        double totalBasis = 0.0;
        for (String sku : skus) {
            Double val = basisValues.get(sku);
            if (val != null) {
                totalBasis += val;
            }
        }

        if (totalBasis == 0.0) {
            for (String sku : skus) {
                allocations.put(sku, 0.0);
            }
            return allocations;
        }

        for (String sku : skus) {
            Double val = basisValues.get(sku);
            if (val != null) {
                double share = (val / totalBasis) * totalExpense;
                allocations.put(sku, Math.floor(share * 100) / 100.0);
            } else {
                allocations.put(sku, 0.0);
            }
        }

        return allocations;
    }
}
