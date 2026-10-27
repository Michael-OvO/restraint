package com.telecom;

import java.math.BigDecimal;
import java.math.RoundingMode;

public class TariffCalculator {

    private static final BigDecimal[] ZONE_RATES = {
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
