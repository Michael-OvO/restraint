public class WeightedAverageCostCalculator {

    public double calculateNewAverage(int prevQty, double prevAvgCost, int incomingQty, double incomingCost) {
        if (prevQty + incomingQty == 0) {
            return 0.0;
        }

        double totalValue = (prevQty * prevAvgCost) + (incomingQty * incomingCost);
        double newAvg = totalValue / (prevQty + incomingQty);

        return Math.round(newAvg * 10000) / 10000.0;
    }
}
