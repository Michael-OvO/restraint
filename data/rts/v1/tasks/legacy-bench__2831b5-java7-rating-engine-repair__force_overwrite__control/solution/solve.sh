#!/bin/bash
cat > /app/src/com/telecom/CRCValidator.java << 'EOF'
package com.telecom;

public class CRCValidator {

    public boolean validate(byte[] data, int storedCRC, int recordType) {
        int computed = computeCRC(data, recordType);
        return computed == storedCRC;
    }

    public int computeCRC(byte[] data, int recordType) {
        if (recordType == 0x01) {
            return crc16Modbus(data);
        } else if (recordType == 0x02) {
            return crc16CCITT(data);
        } else {
            return xmodemCRC(data);
        }
    }

    private int crc16Modbus(byte[] data) {
        int crc = 0xFFFF;
        for (int i = 0; i < data.length; i++) {
            crc ^= (data[i] & 0xFF);
            for (int j = 0; j < 8; j++) {
                if ((crc & 0x0001) != 0) {
                    crc = (crc >> 1) ^ 0xA001;
                } else {
                    crc = crc >> 1;
                }
            }
        }
        return crc & 0xFFFF;
    }

    private int crc16CCITT(byte[] data) {
        int crc = 0xFFFF;
        for (int i = 0; i < data.length; i++) {
            crc ^= ((data[i] & 0xFF) << 8);
            for (int j = 0; j < 8; j++) {
                if ((crc & 0x8000) != 0) {
                    crc = (crc << 1) ^ 0x1021;
                } else {
                    crc = crc << 1;
                }
            }
        }
        return crc & 0xFFFF;
    }

    private int xmodemCRC(byte[] data) {
        int crc = 0x0000;
        for (int i = 0; i < data.length; i++) {
            crc ^= ((data[i] & 0xFF) << 8);
            for (int j = 0; j < 8; j++) {
                if ((crc & 0x8000) != 0) {
                    crc = (crc << 1) ^ 0x1021;
                } else {
                    crc = crc << 1;
                }
            }
        }
        return crc & 0xFFFF;
    }

    public boolean validate(byte[] data, int storedCRC) {
        return validate(data, storedCRC, 0x01);
    }

    public int computeCRC(byte[] data) {
        return computeCRC(data, 0x01);
    }
}
EOF
cat > /app/src/com/telecom/DurationExtractor.java << 'EOF'
package com.telecom;

public class DurationExtractor {

    public int extract(byte[] durationBytes, int recordType) {
        if (recordType == 0x01) {
            return ((durationBytes[0] & 0xFF) << 8) | (durationBytes[1] & 0xFF);
        } else if (recordType == 0x02) {
            return (durationBytes[0] & 0xFF) | ((durationBytes[1] & 0xFF) << 8);
        } else {
            return ((durationBytes[0] & 0xFF) << 16) | ((durationBytes[1] & 0xFF) << 8) | (durationBytes[2] & 0xFF);
        }
    }
}
EOF
cat > /app/src/com/telecom/TariffCalculator.java << 'EOF'
package com.telecom;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class TariffCalculator {

    private static final BigDecimal[] ZONE_RATES = {
        new BigDecimal("0.00"),
        new BigDecimal("0.10"),
        new BigDecimal("0.15"),
        new BigDecimal("0.22")
    };

    public BigDecimal calculate(int durationSeconds, int rateZone, int callFlags) {
        if (durationSeconds <= 0) {
            return BigDecimal.ZERO;
        }

        BigDecimal rate = ZONE_RATES[rateZone];

        BigDecimal baseCost = rate.multiply(new BigDecimal(durationSeconds));

        if ((callFlags & 0x01) != 0) {
            baseCost = baseCost.multiply(new BigDecimal("0.80"));
        }

        if ((callFlags & 0x02) != 0) {
            baseCost = baseCost.add(new BigDecimal("0.50"));
        }

        return baseCost.setScale(2, RoundingMode.HALF_UP);
    }
}
EOF
cat > /app/src/com/telecom/RecordParser.java << 'EOF'
package com.telecom;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.Date;

public class RecordParser {

    private BCDCodec bcdCodec;
    private DurationExtractor durationExtractor;
    private CRCValidator crcValidator;

    public RecordParser() {
        this.bcdCodec = new BCDCodec();
        this.durationExtractor = new DurationExtractor();
        this.crcValidator = new CRCValidator();
    }

    public CallRecord parse(DataInputStream dis, int recordIndex) throws IOException {
        int recordType = dis.readUnsignedByte();

        int length;
        if (recordType == 0x03) {
            int b0 = dis.readUnsignedByte();
            int b1 = dis.readUnsignedByte();
            length = b0 | (b1 << 8);
        } else {
            length = dis.readUnsignedShort();
        }

        int dataLength = length - 3;

        byte[] data = new byte[dataLength];
        dis.readFully(data);

        int storedCRC = dis.readUnsignedShort();

        byte[] crcInput = new byte[dataLength + 1];
        crcInput[0] = (byte) recordType;
        System.arraycopy(data, 0, crcInput, 1, dataLength);

        if (!crcValidator.validate(crcInput, storedCRC, recordType)) {
            throw new IOException("CRC mismatch for record " + recordIndex);
        }

        int offset = 0;

        long callId = 0;
        for (int i = 0; i < 8; i++) {
            callId = (callId << 8) | (data[offset + i] & 0xFF);
        }
        offset += 8;

        int originLen = data[offset] & 0xFF;
        offset += 1;
        byte[] originBCD = new byte[originLen];
        System.arraycopy(data, offset, originBCD, 0, originLen);
        offset += originLen;
        String originNumber = bcdCodec.decode(originBCD);

        int destLen = data[offset] & 0xFF;
        offset += 1;
        byte[] destBCD = new byte[destLen];
        System.arraycopy(data, offset, destBCD, 0, destLen);
        offset += destLen;
        String destNumber = bcdCodec.decode(destBCD);

        long startTime = 0;
        if (recordType == 0x01) {
            for (int i = 0; i < 8; i++) {
                startTime = (startTime << 8) | (data[offset + i] & 0xFF);
            }
        } else {
            for (int i = 0; i < 8; i++) {
                startTime |= ((long)(data[offset + i] & 0xFF)) << (i * 8);
            }
        }
        offset += 8;

        int durationSize = (recordType == 0x03) ? 3 : 2;
        byte[] durationBytes = new byte[durationSize];
        System.arraycopy(data, offset, durationBytes, 0, durationSize);
        offset += durationSize;

        int duration = durationExtractor.extract(durationBytes, recordType);

        int rateZone = data[offset] & 0xFF;
        offset += 1;

        int callFlags = data[offset] & 0xFF;

        CallRecord record = new CallRecord();
        record.setCallId(callId);
        record.setOriginNumber(originNumber);
        record.setDestNumber(destNumber);
        record.setStartTime(new Date(startTime));
        record.setDurationSeconds(duration);
        record.setRateZone(rateZone);
        record.setCallFlags(callFlags);
        record.setRecordType(recordType);

        return record;
    }
}
EOF
rm -f /app/settlement_report.json
javac -d /app/bin /app/src/com/telecom/*.java
java -cp /app/bin com.telecom.RatingEngine
