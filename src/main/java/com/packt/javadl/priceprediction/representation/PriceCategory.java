package com.packt.javadl.priceprediction.representation;

import java.util.NoSuchElementException;

/**
 * Created by zhanghao on 28/9/17.
 *
 * @author ZHANG HAO
 */
public enum PriceCategory {
    OPEN(0, 2),
    HIGH(1, 3),
    LOW(2, 4),
    CLOSE(3, 5),
    VOLUME(4, 6),
    WAP(5, 7),
    COUNT(6, 8),
    MINUTE(7, 9),
    TESLA3(8, 10),
    TESLA6(9, 11),
    TESLA9(10, 12),
    BUY(11, 13),
    SELL(12, 13),
    NO_DECISION(13, 13),
    EXECUTE(14, 14),
    NO_EXECUTE(15, 14),
    ALL(16, -1);

    public final int featureIndex;
    public final int csvIndex;


    PriceCategory(int featureIndex, int csvIndex) {
        this.featureIndex = featureIndex;
        this.csvIndex = csvIndex;
    }

    public static PriceCategory fromFeatureIndex(int index) {
        return switch (index) {
            case 0 -> OPEN;
            case 1 -> HIGH;
            case 2 -> LOW;
            case 3 -> CLOSE;
            case 4 -> VOLUME;
            case 5 -> WAP;
            case 6 -> COUNT;
            case 7 -> MINUTE;
            case 8 -> TESLA3;
            case 9 -> TESLA6;
            case 10 -> TESLA9;
            case 11 -> BUY;
            case 12 -> SELL;
            case 13 -> NO_DECISION;
            case 14 -> EXECUTE;
            case 15 -> NO_EXECUTE;
            case 16 -> ALL;
            default -> throw new NoSuchElementException();
        };
    }
}
