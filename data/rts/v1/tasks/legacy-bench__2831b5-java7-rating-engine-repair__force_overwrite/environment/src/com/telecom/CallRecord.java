package com.telecom;

import java.math.BigDecimal;
import java.util.Date;

public class CallRecord {

    private long callId;
    private String originNumber;
    private String destNumber;
    private Date startTime;
    private int durationSeconds;
    private int rateZone;
    private int callFlags;
    private BigDecimal charge;
    private int recordType;

    public long getCallId() {
        return callId;
    }

    public void setCallId(long callId) {
        this.callId = callId;
    }

    public String getOriginNumber() {
        return originNumber;
    }

    public void setOriginNumber(String originNumber) {
        this.originNumber = originNumber;
    }

    public String getDestNumber() {
        return destNumber;
    }

    public void setDestNumber(String destNumber) {
        this.destNumber = destNumber;
    }

    public Date getStartTime() {
        return startTime;
    }

    public void setStartTime(Date startTime) {
        this.startTime = startTime;
    }

    public int getDurationSeconds() {
        return durationSeconds;
    }

    public void setDurationSeconds(int durationSeconds) {
        this.durationSeconds = durationSeconds;
    }

    public int getRateZone() {
        return rateZone;
    }

    public void setRateZone(int rateZone) {
        this.rateZone = rateZone;
    }

    public int getCallFlags() {
        return callFlags;
    }

    public void setCallFlags(int callFlags) {
        this.callFlags = callFlags;
    }

    public BigDecimal getCharge() {
        return charge;
    }

    public void setCharge(BigDecimal charge) {
        this.charge = charge;
    }

    public int getRecordType() {
        return recordType;
    }

    public void setRecordType(int recordType) {
        this.recordType = recordType;
    }

    public String toJson() {
        StringBuilder sb = new StringBuilder();
        sb.append("{");
        sb.append("\"call_id\":").append(callId).append(",");
        sb.append("\"origin\":\"").append(originNumber).append("\",");
        sb.append("\"dest\":\"").append(destNumber).append("\",");
        sb.append("\"start_time\":").append(startTime.getTime()).append(",");
        sb.append("\"duration\":").append(durationSeconds).append(",");
        sb.append("\"zone\":").append(rateZone).append(",");
        sb.append("\"flags\":").append(callFlags).append(",");
        sb.append("\"charge\":").append(charge.toPlainString());
        sb.append("}");
        return sb.toString();
    }
}
