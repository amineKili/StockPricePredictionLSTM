package com.packt.javadl.priceprediction.utils;

import java.text.MessageFormat;
import java.text.SimpleDateFormat;
import java.util.Date;

public class LoggingUtils {

    public static void print(String message) {
        System.out.println(MessageFormat.format("[{0}] {1}", getCurrentTimeForLogging(), message));
    }

    public static String getCurrentTimeForLogging() {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
    }

}
