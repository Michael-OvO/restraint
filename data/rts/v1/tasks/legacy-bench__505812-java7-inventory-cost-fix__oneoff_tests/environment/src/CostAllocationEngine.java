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
