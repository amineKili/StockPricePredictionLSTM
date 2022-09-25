package com.packt.JavaDL.PricePrediction.Representation;

import lombok.Data;

@Data
public class StockData {
    private String date;
    private String symbol;

    private double open;
    private double close;
    private double low;
    private double high;
    private double volume;

    public StockData() {
    }

    public StockData(String date, String symbol, double open, double close, double low, double high, double volume) {
        this.date = date;
        this.symbol = symbol;
        this.open = open;
        this.close = close;
        this.low = low;
        this.high = high;
        this.volume = volume;
    }

}
