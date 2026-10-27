package com.telecom;

import java.io.DataInputStream;
import java.io.IOException;

public class HeaderParser {

    private int version;
    private int flags;
    private int recordCount;

    public void parse(DataInputStream dis) throws IOException {
        byte[] magic = new byte[4];
        dis.readFully(magic);

        if (magic[0] != 'C' || magic[1] != 'D' || magic[2] != 'R' || magic[3] != 'X') {
            throw new IOException("Invalid magic bytes in CDR header");
        }

        this.version = dis.readUnsignedShort();
        this.flags = dis.readUnsignedShort();
        this.recordCount = dis.readInt();

        int reserved = dis.readInt();
    }

    public int getVersion() {
        return version;
    }

    public int getFlags() {
        return flags;
    }

    public int getRecordCount() {
        return recordCount;
    }
}
