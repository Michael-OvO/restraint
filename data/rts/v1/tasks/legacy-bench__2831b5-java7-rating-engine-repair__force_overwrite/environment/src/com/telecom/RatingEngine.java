package com.telecom;

import java.io.DataInputStream;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

public class RatingEngine {

    public static void main(String[] args) {
        String inputFile = "/app/switch_dump.cdr";
        String outputFile = "/app/settlement_report.json";

        RatingEngine engine = new RatingEngine();
        engine.process(inputFile, outputFile);
    }

    public void process(String inputFile, String outputFile) {
        DataInputStream dis = null;
        FileWriter writer = null;

        try {
            dis = new DataInputStream(new FileInputStream(inputFile));

            HeaderParser headerParser = new HeaderParser();
            headerParser.parse(dis);

            int recordCount = headerParser.getRecordCount();
            RecordParser recordParser = new RecordParser();
            TariffCalculator tariffCalculator = new TariffCalculator();

            List<CallRecord> calls = new ArrayList<CallRecord>();
            int ratingErrors = 0;
            BigDecimal totalBillable = BigDecimal.ZERO;

            for (int i = 0; i < recordCount; i++) {
                try {
                    CallRecord record = recordParser.parse(dis, i);

                    BigDecimal charge = tariffCalculator.calculate(
                        record.getDurationSeconds(),
                        record.getRateZone(),
                        record.getCallFlags()
                    );

                    record.setCharge(charge);
                    calls.add(record);
                    totalBillable = totalBillable.add(charge);
                } catch (IOException e) {
                    ratingErrors++;
                    System.err.println("Error parsing record " + i + ": " + e.getMessage());
                }
            }

            writer = new FileWriter(outputFile);
            writer.write("{\"calls\": [");

            for (int i = 0; i < calls.size(); i++) {
                if (i > 0) {
                    writer.write(",");
                }
                writer.write(calls.get(i).toJson());
            }

            writer.write("], \"total_billable\": ");
            writer.write(totalBillable.toPlainString());
            writer.write(", \"rating_errors\": ");
            writer.write(String.valueOf(ratingErrors));
            writer.write(", \"calls_rated\": ");
            writer.write(String.valueOf(calls.size()));
            writer.write("}");

        } catch (IOException e) {
            System.err.println("Fatal error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            if (dis != null) {
                try {
                    dis.close();
                } catch (IOException e) {
                }
            }
            if (writer != null) {
                try {
                    writer.close();
                } catch (IOException e) {
                }
            }
        }
    }
}
