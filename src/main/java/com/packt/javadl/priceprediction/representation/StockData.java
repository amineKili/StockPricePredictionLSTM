package com.packt.javadl.priceprediction.representation;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class StockData {

    private String date;
    private String symbol;

    private double open;
    private double close;
    private double low;
    private double high;
    private double volume;

    private double wap;
    private double count;
    private double minute;

    private double tesla3;
    private double tesla6;
    private double tesla9;

    private double decision; // 0: NO, 1: BUY, -1: SELL
    private double execute; // 0: NO, 1: EXECUTE

}
