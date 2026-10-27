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
        for (int i = 0; i < 8; i++) {
            startTime = (startTime << 8) | (data[offset + i] & 0xFF);
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
