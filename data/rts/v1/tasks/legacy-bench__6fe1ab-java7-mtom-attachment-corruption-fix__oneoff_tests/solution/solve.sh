#!/bin/bash
cd /app/src
cat > AttachmentProcessor.java << 'JAVAEOF'
import java.util.*;

public class AttachmentProcessor {

    private static final int CHUNK_SIZE = 65536;
    private static final int BOUNDARY_THRESHOLD = 65536;

    public ProcessedAttachment processAttachment(BinaryAttachment attachment) {
        ProcessedAttachment result = new ProcessedAttachment();
        result.attachmentId = attachment.attachmentId;
        result.originalSize = attachment.data != null ? attachment.data.length : 0;
        result.valid = true;

        if (attachment.data == null || attachment.data.length == 0) {
            result.processedSize = 0;
            result.contentHash = 0;
            return result;
        }

        byte[] processed = processBinaryData(attachment.data);
        result.processedSize = processed.length;
        result.contentHash = computeHash(processed);

        return result;
    }

    private byte[] processBinaryData(byte[] data) {
        if (data.length <= BOUNDARY_THRESHOLD) {
            return normalizeContent(data);
        }

        List<byte[]> chunks = new ArrayList<byte[]>();
        int offset = 0;

        while (offset < data.length) {
            int remaining = data.length - offset;
            int chunkLen = Math.min(CHUNK_SIZE, remaining);

            byte[] chunk = new byte[chunkLen];
            System.arraycopy(data, offset, chunk, 0, chunkLen);

            byte[] normalizedChunk = normalizeContent(chunk);
            chunks.add(normalizedChunk);

            offset += chunkLen;
        }

        return reassembleChunks(chunks, data.length);
    }

    private byte[] normalizeContent(byte[] data) {
        byte[] result = new byte[data.length];
        for (int i = 0; i < data.length; i++) {
            result[i] = (byte)(data[i] & 0xFF);
        }
        return result;
    }

    private byte[] reassembleChunks(List<byte[]> chunks, int expectedSize) {
        int totalSize = 0;
        for (byte[] chunk : chunks) {
            totalSize += chunk.length;
        }

        byte[] result = new byte[totalSize];
        int pos = 0;

        for (int i = 0; i < chunks.size(); i++) {
            byte[] chunk = chunks.get(i);
            int copyLen = Math.min(chunk.length, result.length - pos);
            if (copyLen > 0) {
                System.arraycopy(chunk, 0, result, pos, copyLen);
                pos += copyLen;
            }
        }

        return Arrays.copyOf(result, pos);
    }

    private long computeHash(byte[] data) {
        long hash = 0;
        for (int i = 0; i < data.length; i++) {
            hash = hash * 31 + (data[i] & 0xFF);
        }
        return hash;
    }

    public boolean validateAttachmentIntegrity(byte[] original, ProcessedAttachment processed) {
        if (original == null || original.length == 0) {
            return processed.processedSize == 0;
        }

        return processed.originalSize == original.length &&
               processed.processedSize == original.length;
    }

    public int getExpectedProcessedSize(int originalSize) {
        return originalSize;
    }
}
JAVAEOF
javac -source 1.7 -target 1.7 -d /app/bin /app/src/*.java
java -cp /app/bin ClaimsProcessor
