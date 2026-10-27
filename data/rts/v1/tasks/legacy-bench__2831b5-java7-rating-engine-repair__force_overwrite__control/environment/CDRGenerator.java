import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;

public class CDRGenerator {

    public static void main(String[] args) throws IOException {
        DataOutputStream dos = new DataOutputStream(new FileOutputStream("/app/switch_dump.cdr"));

        dos.write(new byte[] {'C', 'D', 'R', 'X'});
        dos.writeShort(0x0002);
        dos.writeShort(0x0001);
        dos.writeInt(30);
        dos.writeInt(0);

        int[] durations = {
            107, 237, 207, 124, 296, 140, 189, 254, 142, 206,
            229, 174, 132, 312, 221, 115, 272, 181, 139, 247,
            198, 288, 157, 128, 221, 264, 180, 144, 212, 213
        };

        int[] rateZones = {
            1, 2, 1, 3, 2, 1, 3, 2, 1, 3,
            2, 1, 3, 2, 1, 2, 3, 1, 2, 3,
            1, 2, 3, 1, 2, 3, 1, 2, 3, 1
        };

        for (int i = 0; i < 30; i++) {
            int recordType;
            if (i < 10) {
                recordType = 0x01;
            } else if (i < 20) {
                recordType = 0x02;
            } else {
                recordType = 0x03;
            }

            ByteArrayOutputStream data = new ByteArrayOutputStream();

            long callId = 1000000L + i;
            for (int b = 7; b >= 0; b--) {
                data.write((int) ((callId >> (b * 8)) & 0xFF));
            }

            byte[] originNum = encodeBCD("155512" + String.format("%05d", 10000 + i));
            data.write(originNum.length);
            data.write(originNum);

            byte[] destNum = encodeBCD("180098" + String.format("%05d", 20000 + i));
            data.write(destNum.length);
            data.write(destNum);

            long startTime = 1609459200000L + (i * 3600000L);
            if (recordType == 0x01) {
                for (int b = 7; b >= 0; b--) {
                    data.write((int) ((startTime >> (b * 8)) & 0xFF));
                }
            } else {
                for (int b = 0; b < 8; b++) {
                    data.write((int) ((startTime >> (b * 8)) & 0xFF));
                }
            }

            int duration = durations[i];
            if (recordType == 0x01) {
                data.write((duration >> 8) & 0xFF);
                data.write(duration & 0xFF);
            } else if (recordType == 0x02) {
                data.write(duration & 0xFF);
                data.write((duration >> 8) & 0xFF);
            } else {
                data.write((duration >> 16) & 0xFF);
                data.write((duration >> 8) & 0xFF);
                data.write(duration & 0xFF);
            }

            int zone = rateZones[i];
            data.write(zone);

            int callFlags = 0x00;
            if (i % 5 == 0) {
                callFlags |= 0x01;
            }
            if (i % 7 == 0) {
                callFlags |= 0x02;
            }
            data.write(callFlags);

            byte[] dataBytes = data.toByteArray();

            byte[] crcInput = new byte[dataBytes.length + 1];
            crcInput[0] = (byte) recordType;
            System.arraycopy(dataBytes, 0, crcInput, 1, dataBytes.length);

            int crc;
            if (recordType == 0x01) {
                crc = crc16Modbus(crcInput);
            } else if (recordType == 0x02) {
                crc = crc16CCITT(crcInput);
            } else {
                crc = xmodemCRC(crcInput);
            }

            int length = 1 + 2 + dataBytes.length;

            dos.writeByte(recordType);

            if (recordType == 0x03) {
                dos.writeByte(length & 0xFF);
                dos.writeByte((length >> 8) & 0xFF);
            } else {
                dos.writeShort(length);
            }

            dos.write(dataBytes);
            dos.writeShort(crc);
        }

        dos.close();
        System.out.println("Generated switch_dump.cdr with 30 records");
    }

    static byte[] encodeBCD(String digits) {
        int len = (digits.length() + 1) / 2;
        byte[] bcd = new byte[len];
        for (int i = 0; i < digits.length(); i += 2) {
            int high = Character.digit(digits.charAt(i), 10);
            int low = (i + 1 < digits.length()) ? Character.digit(digits.charAt(i + 1), 10) : 0x0F;
            bcd[i / 2] = (byte) ((high << 4) | low);
        }
        return bcd;
    }

    static int crc16Modbus(byte[] data) {
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

    static int crc16CCITT(byte[] data) {
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

    static int xmodemCRC(byte[] data) {
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
}
