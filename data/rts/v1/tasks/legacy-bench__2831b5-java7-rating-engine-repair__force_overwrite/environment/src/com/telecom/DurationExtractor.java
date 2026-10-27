package com.telecom;

public class DurationExtractor {

    public int extract(byte[] durationBytes, int recordType) {
        if (recordType == 0x01) {
            return ((durationBytes[0] & 0xFF) << 8) | (durationBytes[1] & 0xFF);
        } else {
            return (durationBytes[0] & 0xFF) | ((durationBytes[1] & 0xFF) << 8);
        }
    }
}
