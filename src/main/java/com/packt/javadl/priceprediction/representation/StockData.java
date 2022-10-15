package com.packt.javadl.priceprediction.representation;

import lombok.Data;

@Data
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

    public StockData(String symbol, String date, double open, double high, double low, double close, double volume, double wap, double count, double minute, double tesla3, double tesla6, double tesla9, double buy, double sell, double execute) {
        this.date = date;
        this.symbol = symbol;
        this.open = open;
        this.high = high;
        this.low = low;
        this.close = close;
        this.volume = volume;
        this.wap = wap;
        this.count = count;
        this.minute = minute;
        this.tesla3 = tesla3;
        this.tesla6 = tesla6;
        this.tesla9 = tesla9;
        this.buy = buy;
        this.sell = sell;
        this.execute = execute;
    }
}
