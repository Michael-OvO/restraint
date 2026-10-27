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

            String[] parts = line.split(",");
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
