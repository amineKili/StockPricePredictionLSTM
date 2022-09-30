package com.packt.javadl.priceprediction.representation;

import java.util.NoSuchElementException;

/**
 * Created by zhanghao on 28/9/17.
 *
 * @author ZHANG HAO
 */
public enum PriceCategory {
    OPEN(0),
    HIGH(1),
    LOW(2),
    CLOSE(3),
    VOLUME(4),
    WAP(5),
    COUNT(6),
    MINUTE(7),
    TESLA3(8),
    TESLA6(9),
    TESLA9(10),
    BUY(11),
    SELL(12),
    NO_DECISION(13),
    EXECUTE(14),
    NO_EXECUTE(15),
    ALL(16);

    public final int index;

    PriceCategory(int index) {
        this.index = index;
    }

    public static PriceCategory fromIndex(int index) {
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
