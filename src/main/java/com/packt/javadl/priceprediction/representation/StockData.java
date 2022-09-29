package com.packt.javadl.priceprediction.representation;

import lombok.AllArgsConstructor;
import lombok.Data;

@Data
@AllArgsConstructor
public class StockData {

    private String date;
    private String symbol;

    private double open;
    private double high;
    private double low;
    private double close;
    private double volume;

    private double wap;
    private double count;
    private double minute;

    private double tesla3;
    private double tesla6;
    private double tesla9;

    private double buy;
    private double sell;

    private double execute;

}
