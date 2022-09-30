package com.packt.javadl.priceprediction.representation;

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
}
