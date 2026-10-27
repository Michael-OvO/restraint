public class CurrencyConverter {

    public double convert(double foreignAmount, double rateToUsd) {
        double converted = foreignAmount * rateToUsd;
        return Math.round(converted * 10000) / 10000.0;
    }
}
