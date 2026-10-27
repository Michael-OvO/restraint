package com.telecom;

public class BCDCodec {

    public String decode(byte[] bcdBytes) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bcdBytes.length; i++) {
            int high = (bcdBytes[i] >> 4) & 0x0F;
            int low = bcdBytes[i] & 0x0F;

            sb.append(high);
            if (low != 0x0F) {
                sb.append(low);
            }
        }
        return sb.toString();
    }
}
