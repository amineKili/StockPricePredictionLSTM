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

        this.featureMapIndex.put(PriceCategory.OPEN, 0);
        this.featureMapIndex.put(PriceCategory.HIGH, 1);
        this.featureMapIndex.put(PriceCategory.LOW, 2);
        this.featureMapIndex.put(PriceCategory.CLOSE, 3);
        this.featureMapIndex.put(PriceCategory.VOLUME, 4);
        this.featureMapIndex.put(PriceCategory.WAP, 5);
        this.featureMapIndex.put(PriceCategory.COUNT, 6);
        this.featureMapIndex.put(PriceCategory.MINUTE, 7);
        this.featureMapIndex.put(PriceCategory.TESLA3, 8);
        this.featureMapIndex.put(PriceCategory.TESLA6, 9);
        this.featureMapIndex.put(PriceCategory.TESLA9, 10);
        this.featureMapIndex.put(PriceCategory.BUY, 11);
        this.featureMapIndex.put(PriceCategory.SELL, 12);
        this.featureMapIndex.put(PriceCategory.NO_DECISION, 13);
        this.featureMapIndex.put(PriceCategory.EXECUTE, 14);
        this.featureMapIndex.put(PriceCategory.NO_EXECUTE, 15);

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
                input.putScalar(new int[]{index, 0, c}, (curData.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]));
                input.putScalar(new int[]{index, 1, c}, (curData.getHigh() - minArray[1]) / (maxArray[1] - minArray[1]));
                input.putScalar(new int[]{index, 2, c}, (curData.getLow() - minArray[2]) / (maxArray[2] - minArray[2]));
                input.putScalar(new int[]{index, 3, c}, (curData.getClose() - minArray[3]) / (maxArray[3] - minArray[3]));
                input.putScalar(new int[]{index, 4, c}, (curData.getVolume() - minArray[4]) / (maxArray[4] - minArray[4]));
                input.putScalar(new int[]{index, 5, c}, (curData.getWap() - minArray[5]) / (maxArray[5] - minArray[5]));
                input.putScalar(new int[]{index, 6, c}, (curData.getCount() - minArray[6]) / (maxArray[6] - minArray[6]));
                input.putScalar(new int[]{index, 7, c}, (curData.getMinute() - minArray[7]) / (maxArray[7] - minArray[7]));
                input.putScalar(new int[]{index, 8, c}, (curData.getTesla3() - minArray[8]) / (maxArray[8] - minArray[8]));
                input.putScalar(new int[]{index, 9, c}, (curData.getTesla6() - minArray[9]) / (maxArray[9] - minArray[9]));
                input.putScalar(new int[]{index, 10, c}, (curData.getTesla9() - minArray[10]) / (maxArray[10] - minArray[10]));

                // Features based on Decision column
                input.putScalar(new int[]{index, 11, c}, (decisionToFeature(PriceCategory.BUY, curData) - minArray[11]) / (maxArray[11] - minArray[11]));
                input.putScalar(new int[]{index, 12, c}, (decisionToFeature(PriceCategory.SELL, curData) - minArray[12]) / (maxArray[12] - minArray[12]));
                input.putScalar(new int[]{index, 13, c}, (decisionToFeature(PriceCategory.NO_DECISION, curData) - minArray[13]) / (maxArray[13] - minArray[13]));

                // Features based on Execute Column
                input.putScalar(new int[]{index, 14, c}, (executeToFeature(PriceCategory.EXECUTE, curData) - minArray[14]) / (maxArray[14] - minArray[14]));
                input.putScalar(new int[]{index, 15, c}, (executeToFeature(PriceCategory.NO_EXECUTE, curData) - minArray[15]) / (maxArray[15] - minArray[15]));
                nextData = train.get(i + 1);
                if (category.equals(PriceCategory.ALL)) {
                    label.putScalar(new int[]{index, 0, c}, (nextData.getOpen() - minArray[1]) / (maxArray[1] - minArray[1]));
                    label.putScalar(new int[]{index, 1, c}, (nextData.getHigh() - minArray[1]) / (maxArray[1] - minArray[1]));
                    label.putScalar(new int[]{index, 2, c}, (nextData.getLow() - minArray[2]) / (maxArray[2] - minArray[2]));
                    label.putScalar(new int[]{index, 3, c}, (nextData.getClose() - minArray[3]) / (maxArray[3] - minArray[3]));
                    label.putScalar(new int[]{index, 4, c}, (nextData.getVolume() - minArray[4]) / (maxArray[4] - minArray[4]));
                    label.putScalar(new int[]{index, 5, c}, (nextData.getWap() - minArray[5]) / (maxArray[5] - minArray[5]));
                    input.putScalar(new int[]{index, 6, c}, (curData.getCount() - minArray[6]) / (maxArray[6] - minArray[6]));
                    input.putScalar(new int[]{index, 7, c}, (curData.getMinute() - minArray[7]) / (maxArray[7] - minArray[7]));
                    input.putScalar(new int[]{index, 8, c}, (curData.getTesla3() - minArray[8]) / (maxArray[8] - minArray[8]));
                    input.putScalar(new int[]{index, 9, c}, (curData.getTesla6() - minArray[9]) / (maxArray[9] - minArray[9]));
                    input.putScalar(new int[]{index, 10, c}, (curData.getTesla9() - minArray[10]) / (maxArray[10] - minArray[10]));

                    // Features based on Decision column
                    input.putScalar(new int[]{index, 11, c}, (decisionToFeature(PriceCategory.BUY, curData) - minArray[11]) / (maxArray[11] - minArray[11]));
                    input.putScalar(new int[]{index, 12, c}, (decisionToFeature(PriceCategory.SELL, curData) - minArray[12]) / (maxArray[12] - minArray[12]));
                    input.putScalar(new int[]{index, 13, c}, (decisionToFeature(PriceCategory.NO_DECISION, curData) - minArray[13]) / (maxArray[13] - minArray[13]));

                    // Features based on Execute Column
                    input.putScalar(new int[]{index, 14, c}, (executeToFeature(PriceCategory.EXECUTE, curData) - minArray[14]) / (maxArray[14] - minArray[14]));
                    input.putScalar(new int[]{index, 15, c}, (executeToFeature(PriceCategory.NO_EXECUTE, curData) - minArray[15]) / (maxArray[15] - minArray[15]));
                } else {
                    label.putScalar(new int[]{index, 0, c}, feedLabel(nextData));
                }
                curData = nextData;
            }
            if (exampleStartOffsets.size() == 0) break;
        }
        return new DataSet(input, label);
    }

    private double feedLabel(StockData data) {
        double value;
        switch (category) {
            case OPEN:
                value = (data.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]);
                break;
            case HIGH:
                value = (data.getClose() - minArray[1]) / (maxArray[1] - minArray[1]);
                break;
            case LOW:
                value = (data.getLow() - minArray[2]) / (maxArray[2] - minArray[2]);
                break;
            case CLOSE:
                value = (data.getHigh() - minArray[3]) / (maxArray[3] - minArray[3]);
                break;
            case VOLUME:
                value = (data.getVolume() - minArray[4]) / (maxArray[4] - minArray[4]);
                break;
            case WAP:
                value = (data.getWap() - minArray[5]) / (maxArray[5] - minArray[5]);
                break;
            case COUNT:
                value = (data.getCount() - minArray[6]) / (maxArray[6] - minArray[6]);
                break;
            case MINUTE:
                value = (data.getMinute() - minArray[7]) / (maxArray[7] - minArray[7]);
                break;

            case TESLA3:
                value = (data.getTesla3() - minArray[8]) / (maxArray[8] - minArray[8]);
                break;
            case TESLA6:
                value = (data.getTesla6() - minArray[9]) / (maxArray[9] - minArray[9]);
                break;
            case TESLA9:
                value = (data.getTesla9() - minArray[10]) / (maxArray[10] - minArray[10]);
                break;

            // Added Features based on decision Column
            case BUY:
                value = (decisionToFeature(category, data) - minArray[11]) / (maxArray[12] - minArray[12]);
                break;
            case SELL:
                value = (decisionToFeature(category, data) - minArray[12]) / (maxArray[12] - minArray[12]);
                break;
            case NO_DECISION:
                value = (decisionToFeature(category, data) - minArray[13]) / (maxArray[13] - minArray[13]);
                break;

            // Added Features based on execute Column
            case EXECUTE:
                value = (executeToFeature(category, data) - minArray[14]) / (maxArray[14] - minArray[14]);
                break;
            case NO_EXECUTE:
                value = (executeToFeature(category, data) - minArray[15]) / (maxArray[15] - minArray[15]);
                break;

            // TODO: add category to index to avoid errors missing an index
            default:
                throw new NoSuchElementException();
        }
        return value;
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

                input.putScalar(new int[]{j - i, 0}, (stock.getOpen() - minArray[0]) / (maxArray[0] - minArray[0]));
                input.putScalar(new int[]{j - i, 1}, (stock.getHigh() - minArray[1]) / (maxArray[1] - minArray[1]));
                input.putScalar(new int[]{j - i, 2}, (stock.getLow() - minArray[2]) / (maxArray[2] - minArray[2]));
                input.putScalar(new int[]{j - i, 3}, (stock.getClose() - minArray[3]) / (maxArray[3] - minArray[3]));
                input.putScalar(new int[]{j - i, 4}, (stock.getVolume() - minArray[4]) / (maxArray[4] - minArray[4]));
                input.putScalar(new int[]{j - i, 5}, (stock.getWap() - minArray[5]) / (maxArray[5] - minArray[5]));
                input.putScalar(new int[]{j - i, 6}, (stock.getCount() - minArray[6]) / (maxArray[6] - minArray[6]));
                input.putScalar(new int[]{j - i, 7}, (stock.getMinute() - minArray[7]) / (maxArray[7] - minArray[7]));
                input.putScalar(new int[]{j - i, 8}, (stock.getTesla3() - minArray[8]) / (maxArray[8] - minArray[8]));
                input.putScalar(new int[]{j - i, 9}, (stock.getTesla6() - minArray[9]) / (maxArray[9] - minArray[9]));
                input.putScalar(new int[]{j - i, 10}, (stock.getTesla9() - minArray[10]) / (maxArray[10] - minArray[10]));
                input.putScalar(new int[]{j - i, 11}, (decisionToFeature(PriceCategory.BUY, stock) - minArray[11]) / (maxArray[11] - minArray[11]));
                input.putScalar(new int[]{j - i, 12}, (decisionToFeature(PriceCategory.SELL, stock) - minArray[12]) / (maxArray[12] - minArray[12]));
                input.putScalar(new int[]{j - i, 13}, (decisionToFeature(PriceCategory.NO_DECISION, stock) - minArray[13]) / (maxArray[13] - minArray[13]));
                input.putScalar(new int[]{j - i, 14}, (executeToFeature(PriceCategory.EXECUTE, stock) - minArray[14]) / (maxArray[14] - minArray[14]));
                input.putScalar(new int[]{j - i, 15}, (executeToFeature(PriceCategory.NO_EXECUTE, stock) - minArray[15]) / (maxArray[15] - minArray[15]));
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
                label.putScalar(new int[]{11}, decisionToFeature(PriceCategory.BUY, stock));
                label.putScalar(new int[]{12}, decisionToFeature(PriceCategory.SELL, stock));
                label.putScalar(new int[]{13}, decisionToFeature(PriceCategory.NO_DECISION, stock));
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
                        label.putScalar(new int[]{0}, stock.getHigh());
                        break;
                    case LOW:
                        label.putScalar(new int[]{0}, stock.getLow());
                        break;
                    case HIGH:
                        label.putScalar(new int[]{0}, stock.getClose());
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
                    }
                    if (nums[i] > maxArray[i]) maxArray[i] = nums[i];
                    if (nums[i] < minArray[i]) minArray[i] = nums[i];
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
}


