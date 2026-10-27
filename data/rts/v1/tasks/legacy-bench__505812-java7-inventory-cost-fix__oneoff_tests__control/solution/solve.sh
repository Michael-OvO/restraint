#!/bin/bash

SOURCE="/app/src"
BIN="/app/bin"
mkdir -p $SOURCE
mkdir -p $BIN

cat <<'EOF' >$SOURCE/CostAllocationEngine.java
import java.io.*;
import java.util.*;

public class CostAllocationEngine {

    private WeightedAverageCostCalculator costCalc;
    private LogisticsExpenseAllocator expenseAllocator;
    private CurrencyConverter currencyConverter;
    private WarehouseTransferProcessor transferProcessor;

    public CostAllocationEngine() {
        this.costCalc = new WeightedAverageCostCalculator();
        this.expenseAllocator = new LogisticsExpenseAllocator();
        this.currencyConverter = new CurrencyConverter();
        this.transferProcessor = new WarehouseTransferProcessor();
    }

    public List<InventoryValuation> calculateValuations(
            List<TransactionRecord> transactions,
            Map<String, Map<String, Double>> exchangeRates,
            Map<String, ExpenseRecord> expenses) {

        Collections.sort(transactions, new Comparator<TransactionRecord>() {
            public int compare(TransactionRecord a, TransactionRecord b) {
                int dateCompare = a.transactionDate.compareTo(b.transactionDate);
                if (dateCompare != 0) return dateCompare;
                if (a.transactionType.equals("TRANSFER_OUT") && b.transactionType.equals("TRANSFER_IN")) {
                    return -1;
                }
                if (a.transactionType.equals("TRANSFER_IN") && b.transactionType.equals("TRANSFER_OUT")) {
                    return 1;
                }
                return 0;
            }
        });

        Map<String, Map<String, WarehouseInventory>> inventoryState =
            new HashMap<String, Map<String, WarehouseInventory>>();

        Map<String, Set<String>> skuBatches = new HashMap<String, Set<String>>();

        for (TransactionRecord txn : transactions) {
            if (!inventoryState.containsKey(txn.sku)) {
                inventoryState.put(txn.sku, new HashMap<String, WarehouseInventory>());
            }
            Map<String, WarehouseInventory> warehouseMap = inventoryState.get(txn.sku);

            if (!warehouseMap.containsKey(txn.warehouseId)) {
                WarehouseInventory inv = new WarehouseInventory();
                inv.sku = txn.sku;
                inv.warehouseId = txn.warehouseId;
                inv.quantity = 0;
                inv.avgCost = 0.0;
                warehouseMap.put(txn.warehouseId, inv);
            }

            WarehouseInventory inv = warehouseMap.get(txn.warehouseId);

            if (txn.transactionType.equals("PURCHASE")) {
                double convertedCost = txn.unitCost;
                if (!txn.currencyCode.equals("USD")) {
                    Map<String, Double> dateRates = exchangeRates.get(txn.transactionDate);
                    if (dateRates != null && dateRates.containsKey(txn.currencyCode)) {
                        convertedCost = currencyConverter.convert(txn.unitCost, dateRates.get(txn.currencyCode));
                    }
                }
                inv.avgCost = costCalc.calculateNewAverage(inv.quantity, inv.avgCost, txn.quantity, convertedCost);
                inv.quantity += txn.quantity;
            } else if (txn.transactionType.equals("SALE")) {
                inv.quantity -= txn.quantity;
                if (inv.quantity < 0) inv.quantity = 0;
            } else if (txn.transactionType.equals("TRANSFER_OUT")) {
                double transferCost = inv.avgCost;
                inv.quantity -= txn.quantity;
                if (inv.quantity < 0) inv.quantity = 0;
                transferProcessor.recordOutgoing(txn.sku, txn.warehouseId, txn.transactionDate, txn.quantity, transferCost);
            } else if (txn.transactionType.equals("TRANSFER_IN")) {
                double incomingCost = transferProcessor.getIncomingCost(txn.sku, txn.transactionDate, txn.quantity);
                inv.avgCost = costCalc.calculateNewAverage(inv.quantity, inv.avgCost, txn.quantity, incomingCost);
                inv.quantity += txn.quantity;
            }

            if (txn.allocationBatchId != null && !txn.allocationBatchId.isEmpty()) {
                if (!skuBatches.containsKey(txn.sku)) {
                    skuBatches.put(txn.sku, new HashSet<String>());
                }
                skuBatches.get(txn.sku).add(txn.allocationBatchId);
            }
        }

        Map<String, Map<String, Double>> skuAllocations = new HashMap<String, Map<String, Double>>();
        for (String batchId : expenses.keySet()) {
            ExpenseRecord expense = expenses.get(batchId);
            List<String> batchSkus = new ArrayList<String>();
            for (String sku : skuBatches.keySet()) {
                if (skuBatches.get(sku).contains(batchId)) {
                    batchSkus.add(sku);
                }
            }
            if (!batchSkus.isEmpty()) {
                Map<String, Double> allocations = expenseAllocator.allocate(batchSkus, expense.totalExpense, expense.allocationBasis);
                for (String sku : allocations.keySet()) {
                    if (!skuAllocations.containsKey(sku)) {
                        skuAllocations.put(sku, new HashMap<String, Double>());
                    }
                    Map<String, Double> skuAlloc = skuAllocations.get(sku);
                    if (!skuAlloc.containsKey(batchId)) {
                        skuAlloc.put(batchId, 0.0);
                    }
                    skuAlloc.put(batchId, skuAlloc.get(batchId) + allocations.get(sku));
                }
            }
        }

        List<InventoryValuation> results = new ArrayList<InventoryValuation>();

        for (String sku : inventoryState.keySet()) {
            Map<String, WarehouseInventory> warehouseMap = inventoryState.get(sku);
            for (String warehouseId : warehouseMap.keySet()) {
                WarehouseInventory inv = warehouseMap.get(warehouseId);
                if (inv.quantity > 0) {
                    InventoryValuation val = new InventoryValuation();
                    val.sku = sku;
                    val.warehouseId = warehouseId;
                    val.finalQuantity = inv.quantity;
                    val.finalUnitCost = truncate4(inv.avgCost);

                    double totalAllocated = 0.0;
                    if (skuAllocations.containsKey(sku)) {
                        for (Double alloc : skuAllocations.get(sku).values()) {
                            totalAllocated += alloc;
                        }
                    }
                    val.allocatedExpense = truncate2(totalAllocated);
                    val.totalValue = truncate2(val.finalQuantity * val.finalUnitCost);
                    results.add(val);
                }
            }
        }

        Collections.sort(results, new Comparator<InventoryValuation>() {
            public int compare(InventoryValuation a, InventoryValuation b) {
                int skuCompare = a.sku.compareTo(b.sku);
                if (skuCompare != 0) return skuCompare;
                return a.warehouseId.compareTo(b.warehouseId);
            }
        });

        return results;
    }

    private double truncate4(double value) {
        return Math.floor(value * 10000) / 10000.0;
    }

    private double truncate2(double value) {
        return Math.floor(value * 100) / 100.0;
    }

    public static void main(String[] args) {
        if (args.length < 4) {
            System.exit(1);
        }

        CostAllocationEngine engine = new CostAllocationEngine();

        try {
            List<TransactionRecord> transactions = InventoryDataParser.parseTransactions(args[0]);
            Map<String, Map<String, Double>> rates = InventoryDataParser.parseRates(args[1]);
            Map<String, ExpenseRecord> expenses = InventoryDataParser.parseExpenses(args[2]);
            List<InventoryValuation> results = engine.calculateValuations(transactions, rates, expenses);
            ReportWriter.writeReport(args[3], results);
        } catch (Exception e) {
            System.exit(1);
        }
    }
}

class WarehouseInventory {
    String sku;
    String warehouseId;
    int quantity;
    double avgCost;
}

class InventoryValuation {
    String sku;
    String warehouseId;
    int finalQuantity;
    double finalUnitCost;
    double allocatedExpense;
    double totalValue;
}
EOF

cat <<'EOF' >$SOURCE/WeightedAverageCostCalculator.java
public class WeightedAverageCostCalculator {

    public double calculateNewAverage(int prevQty, double prevAvgCost, int incomingQty, double incomingCost) {
        if (prevQty + incomingQty == 0) {
            return 0.0;
        }

        double totalValue = (prevQty * prevAvgCost) + (incomingQty * incomingCost);
        double newAvg = totalValue / (prevQty + incomingQty);

        return Math.floor(newAvg * 10000) / 10000.0;
    }
}
EOF

cat <<'EOF' >$SOURCE/LogisticsExpenseAllocator.java
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
        if (basis.equals("W")) {
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
EOF

cat <<'EOF' >$SOURCE/CurrencyConverter.java
public class CurrencyConverter {

    public double convert(double foreignAmount, double rateToUsd) {
        double converted = foreignAmount * rateToUsd;
        return Math.floor(converted * 10000) / 10000.0;
    }
}
EOF

cat <<'EOF' >$SOURCE/WarehouseTransferProcessor.java
import java.util.*;

public class WarehouseTransferProcessor {

    private Map<String, List<TransferRecord>> pendingTransfers;

    public WarehouseTransferProcessor() {
        this.pendingTransfers = new HashMap<String, List<TransferRecord>>();
    }

    public void recordOutgoing(String sku, String fromWarehouse, String date, int quantity, double unitCost) {
        String key = sku + "|" + date;
        if (!pendingTransfers.containsKey(key)) {
            pendingTransfers.put(key, new ArrayList<TransferRecord>());
        }

        TransferRecord record = new TransferRecord();
        record.sku = sku;
        record.fromWarehouse = fromWarehouse;
        record.date = date;
        record.quantity = quantity;
        record.unitCost = unitCost;
        pendingTransfers.get(key).add(record);
    }

    public double getIncomingCost(String sku, String date, int quantity) {
        String key = sku + "|" + date;
        List<TransferRecord> transfers = pendingTransfers.get(key);

        if (transfers == null || transfers.isEmpty()) {
            return 0.0;
        }

        for (int i = 0; i < transfers.size(); i++) {
            TransferRecord t = transfers.get(i);
            if (t.quantity == quantity) {
                transfers.remove(i);
                return t.unitCost;
            }
        }

        if (!transfers.isEmpty()) {
            TransferRecord t = transfers.remove(0);
            return t.unitCost;
        }

        return 0.0;
    }
}

class TransferRecord {
    String sku;
    String fromWarehouse;
    String date;
    int quantity;
    double unitCost;
}
EOF

cat <<'EOF' >$SOURCE/InventoryDataParser.java
import java.io.*;
import java.util.*;

public class InventoryDataParser {

    public static List<TransactionRecord> parseTransactions(String filename) throws IOException {
        List<TransactionRecord> records = new ArrayList<TransactionRecord>();
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        boolean header = true;

        while ((line = reader.readLine()) != null) {
            if (header) {
                header = false;
                continue;
            }
            line = line.trim();
            if (line.isEmpty()) continue;

            String[] parts = line.split(",", -1);
            if (parts.length < 8) continue;

            try {
                TransactionRecord record = new TransactionRecord();
                record.sku = parts[0].trim();
                record.transactionDate = parts[1].trim();
                record.transactionType = parts[2].trim();
                record.quantity = Integer.parseInt(parts[3].trim());
                record.unitCost = Double.parseDouble(parts[4].trim());
                record.warehouseId = parts[5].trim();
                record.currencyCode = parts[6].trim();
                record.allocationBatchId = parts[7].trim();
                records.add(record);
            } catch (Exception e) {
                continue;
            }
        }
        reader.close();
        return records;
    }

    public static Map<String, Map<String, Double>> parseRates(String filename) throws IOException {
        Map<String, Map<String, Double>> rates = new HashMap<String, Map<String, Double>>();
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        boolean header = true;

        while ((line = reader.readLine()) != null) {
            if (header) {
                header = false;
                continue;
            }
            line = line.trim();
            if (line.isEmpty()) continue;

            String[] parts = line.split(",");
            if (parts.length < 3) continue;

            try {
                String date = parts[0].trim();
                String currency = parts[1].trim();
                double rate = Double.parseDouble(parts[2].trim());

                if (!rates.containsKey(date)) {
                    rates.put(date, new HashMap<String, Double>());
                }
                rates.get(date).put(currency, rate);
            } catch (Exception e) {
                continue;
            }
        }
        reader.close();
        return rates;
    }

    public static Map<String, ExpenseRecord> parseExpenses(String filename) throws IOException {
        Map<String, ExpenseRecord> expenses = new HashMap<String, ExpenseRecord>();
        BufferedReader reader = new BufferedReader(new FileReader(filename));
        String line;
        boolean header = true;

        while ((line = reader.readLine()) != null) {
            if (header) {
                header = false;
                continue;
            }
            line = line.trim();
            if (line.isEmpty()) continue;

            String[] parts = line.split(",");
            if (parts.length < 3) continue;

            try {
                ExpenseRecord record = new ExpenseRecord();
                record.batchId = parts[0].trim();
                record.totalExpense = Double.parseDouble(parts[1].trim());
                record.allocationBasis = parts[2].trim();
                expenses.put(record.batchId, record);
            } catch (Exception e) {
                continue;
            }
        }
        reader.close();
        return expenses;
    }
}

class TransactionRecord {
    String sku;
    String transactionDate;
    String transactionType;
    int quantity;
    double unitCost;
    String warehouseId;
    String currencyCode;
    String allocationBatchId;
}

class ExpenseRecord {
    String batchId;
    double totalExpense;
    String allocationBasis;
}
EOF

cat <<'EOF' >$SOURCE/ReportWriter.java
import java.io.*;
import java.util.*;

public class ReportWriter {

    public static void writeReport(String filename, List<InventoryValuation> results) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter(filename));

        writer.write("sku,warehouse_id,final_quantity,final_unit_cost,allocated_expense,total_value");
        writer.newLine();

        for (InventoryValuation v : results) {
            StringBuilder sb = new StringBuilder();
            sb.append(v.sku).append(",");
            sb.append(v.warehouseId).append(",");
            sb.append(v.finalQuantity).append(",");
            sb.append(String.format("%.4f", v.finalUnitCost)).append(",");
            sb.append(String.format("%.2f", v.allocatedExpense)).append(",");
            sb.append(String.format("%.2f", v.totalValue));
            writer.write(sb.toString());
            writer.newLine();
        }

        writer.close();
    }
}
EOF

javac -source 1.7 -target 1.7 $SOURCE/*.java -d $BIN
