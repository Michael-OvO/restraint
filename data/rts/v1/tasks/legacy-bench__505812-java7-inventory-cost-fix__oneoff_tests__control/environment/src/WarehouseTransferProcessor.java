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
        record.unitCost = Math.round(unitCost * 10000) / 10000.0;
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
