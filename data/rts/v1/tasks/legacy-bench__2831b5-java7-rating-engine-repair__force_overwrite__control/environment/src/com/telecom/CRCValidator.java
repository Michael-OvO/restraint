package com.telecom;

public class CRCValidator {

    public boolean validate(byte[] data, int storedCRC, int recordType) {
        int computed = computeCRC(data, recordType);
        return computed == storedCRC;
    }

    public int computeCRC(byte[] data, int recordType) {
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

    public boolean validate(byte[] data, int storedCRC) {
        return validate(data, storedCRC, 0x01);
    }

    public int computeCRC(byte[] data) {
        return computeCRC(data, 0x01);
    }
}
