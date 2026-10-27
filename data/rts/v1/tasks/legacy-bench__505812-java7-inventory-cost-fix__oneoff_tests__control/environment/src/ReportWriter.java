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
