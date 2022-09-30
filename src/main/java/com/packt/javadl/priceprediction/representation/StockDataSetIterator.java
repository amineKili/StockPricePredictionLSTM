package com.packt.javadl.priceprediction.representation;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import com.packt.javadl.priceprediction.utils.Pair;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;

import java.io.FileReader;
import java.io.IOException;
import java.text.MessageFormat;
import java.util.*;

@SuppressWarnings("serial")
public class StockDataSetIterator implements DataSetIterator {
    /**
     * category and its index
     */
    private final Map<PriceCategory, Integer> featureMapIndex = new HashMap<>();
    private final Map<Integer, PriceCategory> indexMapFeature = new HashMap<>();

    // New features size is length of [Open,High,Low,Close,Volume,WAP,Count,Minute,Tesla3,Tesla6,Tesla9,BUY, SELL, NO_Decision, EXECUTE, No_Execute] = 16 feature
    public static final int VECTOR_SIZE = 16; // number of features for a stock data

    private final int miniBatchSize; // mini-batch size
    private int exampleLength = 22; // default 22, say, 22 working days per month

    // TODO : make this a variable, predict X next day
    private final int predictLength = 1; // default 1, say, one day ahead prediction

    /**
     * minimal values of each feature in stock dataset
     */
    private final double[] minArray = new double[VECTOR_SIZE];
    /**
     * maximal values of each feature in stock dataset
     */
    private final double[] maxArray = new double[VECTOR_SIZE];

    /**
     * feature to be selected as a training target
     */
    private final PriceCategory category;

    /**
     * mini-batch offset
     */
    private final LinkedList<Integer> exampleStartOffsets = new LinkedList<>();

    /**
     * stock dataset for training
     */
    private final List<StockData> train;

    /**
     * adjusted stock dataset for testing
     */
    private final List<Pair<INDArray, INDArray>> test;

    public StockDataSetIterator(String filename, String symbol, int miniBatchSize, int exampleLength, double splitRatio, PriceCategory category) {

        for (PriceCategory priceCategory : PriceCategory.values()) {
            if (priceCategory != PriceCategory.ALL) {
                featureMapIndex.put(priceCategory, priceCategory.index);
                indexMapFeature.put(priceCategory.index, priceCategory);
            }
        }

        List<StockData> stockDataList = readStockDataFromFile(filename, symbol);
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        this.category = category;
        int split = (int) Math.round(stockDataList.size() * splitRatio);
        train = stockDataList.subList(0, split);
        // TODO : change in case you want to test on different data
        test = generateTestDataSet(stockDataList.subList(split, stockDataList.size()));
        initializeOffsets();
    }

    /**
     * initialize the mini-batch offsets
     */
    private void initializeOffsets() {
        exampleStartOffsets.clear();
        int window = exampleLength + predictLength;
        for (int i = 0; i < train.size() - window; i++) {
            exampleStartOffsets.add(i);
        }
    }


    // TODO: return TestData
    public List<Pair<INDArray, INDArray>> getTestDataSet() {
        return test;
    }

    public double[] getMaxArray() {
        return maxArray;
    }

    public double[] getMinArray() {
        return minArray;
    }

    public double getMaxNum(PriceCategory category) {
        return maxArray[featureMapIndex.get(category)];
    }

    public double getMinNum(PriceCategory category) {
        return minArray[featureMapIndex.get(category)];
    }

    public DataSet next(int num) {
        if (exampleStartOffsets.size() == 0) throw new NoSuchElementException();
        int actualMiniBatchSize = Math.min(num, exampleStartOffsets.size());
        INDArray input = Nd4j.create(new int[]{actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        INDArray label;

        if (category.equals(PriceCategory.ALL))
            label = Nd4j.create(new int[]{actualMiniBatchSize, VECTOR_SIZE, exampleLength}, 'f');
        else
            label = Nd4j.create(new int[]{actualMiniBatchSize, predictLength, exampleLength}, 'f');

        for (int index = 0; index < actualMiniBatchSize; index++) {
            int startIdx = exampleStartOffsets.removeFirst();
            int endIdx = startIdx + exampleLength;
            StockData curData = train.get(startIdx);
            StockData nextData;
            for (int i = startIdx; i < endIdx; i++) {
                int c = i - startIdx;

                // Input Features

                for(PriceCategory priceCategory : PriceCategory.values()) {
                    if (priceCategory != PriceCategory.ALL) {
                        int featureIdx = featureMapIndex.get(priceCategory);
                        double value = getNormalizedValue(curData, priceCategory);
                        input.putScalar(new int[]{index, featureIdx, c}, value);
                    }
                }

                nextData = train.get(i + 1);
                if (category.equals(PriceCategory.ALL)) {
                    // Put all the features as labels

                    for (PriceCategory priceCategory : PriceCategory.values()) {
                        if (priceCategory != PriceCategory.ALL) {
                            int featureIdx = featureMapIndex.get(priceCategory);
                            double value = getNormalizedValue(nextData, priceCategory);
                            label.putScalar(new int[]{index, featureIdx, c}, value);
                        }
                    }

                } else {
                    label.putScalar(new int[]{index, 0, c}, getNormalizedValue(nextData, category));
                }
                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
        return new DataSet(input, label);
    }

    public int totalExamples() {
        return train.size() - exampleLength - predictLength;
    }

    public int inputColumns() {
        return VECTOR_SIZE;
    }

    @Override
    public int totalOutcomes() {
        if (this.category.equals(PriceCategory.ALL)) return predictLength * VECTOR_SIZE;
        else return predictLength;
    }

    public boolean resetSupported() {
        return false;
    }

    public boolean asyncSupported() {
        return false;
    }

    public void reset() {
        initializeOffsets();
    }

    public int batch() {
        return miniBatchSize;
    }

    public int cursor() {
        return totalExamples() - exampleStartOffsets.size();
    }

    public int numExamples() {
        return totalExamples();
    }

    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not Implemented");
    }

    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not Implemented");
    }

    public boolean hasNext() {
        return exampleStartOffsets.size() > 0;
    }

    public DataSet next() {
        return next(miniBatchSize);
    }

    private List<Pair<INDArray, INDArray>> generateTestDataSet(List<StockData> stockDataList) {
        int window = exampleLength + predictLength;
        List<Pair<INDArray, INDArray>> test = new ArrayList<>();
        for (int i = 0; i < stockDataList.size() - window; i++) {
            INDArray input = Nd4j.create(new int[]{exampleLength, VECTOR_SIZE}, 'f');
            for (int j = i; j < i + exampleLength; j++) {
                StockData stock = stockDataList.get(j);

                input.putScalar(new int[]{j - i, 0}, getNormalizedValue(stock, PriceCategory.OPEN));
                input.putScalar(new int[]{j - i, 1}, getNormalizedValue(stock, PriceCategory.HIGH));
                input.putScalar(new int[]{j - i, 2}, getNormalizedValue(stock, PriceCategory.LOW));
                input.putScalar(new int[]{j - i, 3}, getNormalizedValue(stock, PriceCategory.CLOSE));
                input.putScalar(new int[]{j - i, 4}, getNormalizedValue(stock, PriceCategory.VOLUME));

                input.putScalar(new int[]{j - i, 5}, getNormalizedValue(stock, PriceCategory.WAP));
                input.putScalar(new int[]{j - i, 6}, getNormalizedValue(stock, PriceCategory.COUNT));
                input.putScalar(new int[]{j - i, 7}, getNormalizedValue(stock, PriceCategory.MINUTE));

                input.putScalar(new int[]{j - i, 8}, getNormalizedValue(stock, PriceCategory.TESLA3));
                input.putScalar(new int[]{j - i, 9}, getNormalizedValue(stock, PriceCategory.TESLA6));
                input.putScalar(new int[]{j - i, 10}, getNormalizedValue(stock, PriceCategory.TESLA9));

                // Features based on Decision column
                input.putScalar(new int[]{j - i, 11}, getNormalizedValue(stock, PriceCategory.BUY));
                input.putScalar(new int[]{j - i, 12}, getNormalizedValue(stock, PriceCategory.SELL));
                input.putScalar(new int[]{j - i, 13}, getNormalizedValue(stock, PriceCategory.NO_DECISION));

                // Added Features based on execute Column
                input.putScalar(new int[]{j - i, 14}, getNormalizedValue(stock, PriceCategory.EXECUTE));
                input.putScalar(new int[]{j - i, 15}, getNormalizedValue(stock, PriceCategory.NO_EXECUTE));
            }

            StockData stock = stockDataList.get(i + exampleLength);
            INDArray label;
            // TODO: add column, execute, no execute, buy, sell, hold
            if (category.equals(PriceCategory.ALL)) {
                label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f'); // ordering is set as 'f', faster construct
                label.putScalar(new int[]{0}, stock.getOpen());
                label.putScalar(new int[]{1}, stock.getHigh());
                label.putScalar(new int[]{2}, stock.getLow());
                label.putScalar(new int[]{3}, stock.getClose());
                label.putScalar(new int[]{4}, stock.getVolume());
                label.putScalar(new int[]{5}, stock.getWap());
                label.putScalar(new int[]{6}, stock.getCount());
                label.putScalar(new int[]{7}, stock.getMinute());
                label.putScalar(new int[]{8}, stock.getTesla3());
                label.putScalar(new int[]{9}, stock.getTesla6());
                label.putScalar(new int[]{10}, stock.getTesla9());

                // Features based on Decision column
                label.putScalar(new int[]{11}, decisionToFeature(PriceCategory.BUY, stock));
                label.putScalar(new int[]{12}, decisionToFeature(PriceCategory.SELL, stock));
                label.putScalar(new int[]{13}, decisionToFeature(PriceCategory.NO_DECISION, stock));

                // Added Features based on execute Column
                label.putScalar(new int[]{14}, executeToFeature(PriceCategory.EXECUTE, stock));
                label.putScalar(new int[]{15}, executeToFeature(PriceCategory.NO_EXECUTE, stock));

            } else {
                label = Nd4j.create(new int[]{1}, 'f');
                // TODO: add column, execute, no execute, buy, sell, hold
                switch (category) {
                    case OPEN:
                        label.putScalar(new int[]{0}, stock.getOpen());
                        break;
                    case CLOSE:
                        label.putScalar(new int[]{0}, stock.getClose());
                        break;
                    case LOW:
                        label.putScalar(new int[]{0}, stock.getLow());
                        break;
                    case HIGH:
                        label.putScalar(new int[]{0}, stock.getHigh());
                        break;
                    case VOLUME:
                        label.putScalar(new int[]{0}, stock.getVolume());
                        break;
                    case WAP:
                        label.putScalar(new int[]{0}, stock.getWap());
                        break;
                    case COUNT:
                        label.putScalar(new int[]{0}, stock.getCount());
                        break;
                    case MINUTE:
                        label.putScalar(new int[]{0}, stock.getMinute());
                        break;
                    case TESLA3:
                        label.putScalar(new int[]{0}, stock.getTesla3());
                        break;
                    case TESLA6:
                        label.putScalar(new int[]{0}, stock.getTesla6());
                        break;
                    case TESLA9:
                        label.putScalar(new int[]{0}, stock.getTesla9());
                        break;
                    case BUY:
                    case NO_DECISION:
                    case SELL:
                        label.putScalar(new int[]{0}, decisionToFeature(category, stock));
                        break;
                    case EXECUTE:
                    case NO_EXECUTE:
                        label.putScalar(new int[]{0}, executeToFeature(category, stock));
                        break;
                    default:
                        throw new NoSuchElementException();
                }
            }
            test.add(new Pair<>(input, label));
        }
        return test;
    }


    /**
     * @param filename File Content:
     *                 Currency,YYYYMMDD_HHMMSS,Open,High,Low,Close,Volume,WAP,Count,Minute,Tesla3,Tesla6,Tesla9,Decision,EXECUTE
     *                 CHF,20200616  09:30:00,1.0574,1.0583,1.0573,1.0583,269,1.0578,111,30,0.408462,0.81675,1.225584,BUY,NO
     * @param symbol
     * @return List<StockData> parsed content
     */

    // TODO: change features, add column execute, not execute, buy, sell, hold, no decision
    @SuppressWarnings("resource")
    private List<StockData> readStockDataFromFile(String filename, String symbol) {
        List<StockData> stockDataList = new ArrayList<>();
        try {
            for (int i = 0; i < maxArray.length; i++) {
                // initialize max and min arrays, for normalization
                maxArray[i] = Double.MIN_VALUE;
                minArray[i] = Double.MAX_VALUE;
                if (i == 11 || i == 12 || i == 13 || i == 14 || i == 15) {
                    maxArray[i] = 1;
                    minArray[i] = 0;
                }
            }
            // load all elements in a list,
            List<String[]> list = new CSVReader(new FileReader(filename)).readAll();
            // item nb 0 is String "Currency" =>  No relative feature
            // item nb 1 is String "YYYYMMDD_HHMMSS" the Date  =>  No relative feature
            // item nb 2 is Double "Open" [To be parsed] => Feature 0
            // item nb 3 is Double "High" [To be parsed] => Feature 1
            // item nb 4 is Double "Low" [To be parsed] => Feature 2
            // item nb 5 is Double "Close" [To be parsed] => Feature 3
            // item nb 6 is Double "Volume" [To be parsed] => Feature 4
            // item nb 7 is Double "WAP" [To be parsed] => Feature 5
            // item nb 8 is Double "Count" [To be parsed] => Feature 6
            // item nb 9 is Double "Minute" [To be parsed] => Feature 7
            // item nb 10 is Double "Tesla3" [To be parsed] => Feature 8
            // item nb 11 is Double "Tesla6" [To be parsed] => Feature 9
            // item nb 12 is Double "Tesla9" [To be parsed] => Feature 10
            // item nb 13 is Double "Decision" => Create three columns BUY, SELL, NO_DECISION => Feature 11, 12, 13
            // item nb 14 is Double "Execute" => Create two columns EXECUTE, NO_EXECUTE => Feature 14, 15
            System.out.println(MessageFormat.format("FileContent Reading, Lines Count {0}", list.size()));
            boolean headerSkipped = false;
            for (int j = 0; j < list.size(); j++) {
                String[] arr = list.get(j);
                if (!headerSkipped) {
                    headerSkipped = true;
                    continue;
                }
                double[] nums = new double[VECTOR_SIZE];
                // TODO: adjust to new features, add column execute, not execute, buy, sell, hold, no decision
                for (int i = 0; i < arr.length - 2; i++) {
                    var value = arr[i + 2].trim().toLowerCase();
                    // Parsing Decision Column
                    if (i == 11) {
                        if (value.isEmpty() || value.isBlank() || value.equals("sell") || value.equals("buy") || value.equals("no")
                        ) {
                            switch (value) {
                                case "no", "" -> {
                                    nums[i] = 0;
                                    nums[i + 1] = 0;
                                    nums[i + 2] = 1;
                                }
                                case "sell" -> {
                                    nums[i] = 0;
                                    nums[i + 1] = 1;
                                    nums[i + 2] = 0;
                                }
                                case "buy" -> {
                                    nums[i] = 1;
                                    nums[i + 1] = 0;
                                    nums[i + 2] = 0;
                                }
                                default -> throw new RuntimeException(MessageFormat.format("String Value {0} is not parsed Correctly", value));
                            }
                        } else {
                            throw new RuntimeException(MessageFormat.format("String Value {0} is not parsed Correctly", value));
                        }
                        // Parsing Execute Column
                    } else if (i == 12) {
                        if (value.isEmpty() || value.isBlank() || value.equals("execute") || value.equals("no")
                        ) {
                            switch (value) {
                                case "no", "" -> {
                                    nums[i + 2] = 0;
                                    nums[i + 3] = 1;
                                }
                                case "execute" -> {
                                    nums[i + 2] = 1;
                                    nums[i + 3] = 0;
                                }
                                default -> throw new RuntimeException(MessageFormat.format("String Value {0} is not parsed Correctly", value));
                            }
                        } else {
                            throw new RuntimeException(MessageFormat.format("String Value {0} is not parsed Correctly", value));
                        }
                    } else {
                        if (value.isEmpty() || value.isBlank()) {
                            nums[i] = 0;
                        } else {
                            nums[i] = Double.parseDouble(value);
                        }
                        if (nums[i] > maxArray[i]) maxArray[i] = nums[i];
                        if (nums[i] < minArray[i]) minArray[i] = nums[i];
                    }

                }
                stockDataList.add(new StockData(
                                arr[0], arr[1], // Currency, Date
                                nums[0], nums[1], nums[2], nums[3], nums[4], // Open, High, Low, Close, Volume
                                nums[5], nums[6], nums[7], // WAP, Count, Minute
                                nums[8], nums[9], nums[10], // Tesla3, Tesla6, Tesla9
                                nums[11], nums[12], // buy, sell
                                nums[14] // execute
                        )
                );
            }
        } catch (IOException | CsvException e) {
            e.printStackTrace();
        }
        System.out.print("MaxArray: " + Arrays.toString(maxArray));
        System.out.println("MinArray: " + Arrays.toString(minArray));
        System.out.println(MessageFormat.format("Finish Reading CSV, Stock Dataset Size {0}", stockDataList.size()));
        return stockDataList;
    }

    public static int decisionToFeature(PriceCategory priceCategory, StockData stock) {
        if (priceCategory == PriceCategory.BUY && stock.getBuy() == 1) return 1;
        if (priceCategory == PriceCategory.SELL && stock.getSell() == 1) return 1;
        if (priceCategory == PriceCategory.NO_DECISION && stock.getBuy() == 0 && stock.getSell() == 0) return 1;
        return 0;
    }


    public static int executeToFeature(PriceCategory priceCategory, StockData stock) {
        if (priceCategory == PriceCategory.EXECUTE && stock.getExecute() == 1) return 1;
        if (priceCategory == PriceCategory.NO_EXECUTE && stock.getExecute() == 0) return 1;
        return 0;
    }


    public double getNormalizedValue(StockData stock, PriceCategory priceCategory) {
        int index = this.featureMapIndex.get(priceCategory);

        switch (priceCategory) {
            case OPEN -> {
                return (stock.getOpen() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case HIGH -> {
                return (stock.getHigh() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case LOW -> {
                return (stock.getLow() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case CLOSE -> {
                return (stock.getClose() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case VOLUME -> {
                return (stock.getVolume() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case WAP -> {
                return (stock.getWap() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case COUNT -> {
                return (stock.getCount() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case MINUTE -> {
                return (stock.getMinute() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case TESLA3 -> {
                return (stock.getTesla3() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case TESLA6 -> {
                return (stock.getTesla6() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case TESLA9 -> {
                return (stock.getTesla9() - minArray[this.featureMapIndex.get(priceCategory)] / (maxArray[index] - minArray[index]));
            }
            case BUY, NO_DECISION, SELL -> {
                return decisionToFeature(priceCategory, stock);
            }
            case EXECUTE, NO_EXECUTE -> {
                return executeToFeature(priceCategory, stock);
            }
            default -> throw new RuntimeException(MessageFormat.format("Price Category {0} is not defined", priceCategory));
        }
    }
}


