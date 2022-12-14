package com.packt.javadl.priceprediction.representation;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import com.packt.javadl.priceprediction.utils.LoggingUtils;
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

    // New features size is length of [Open,High,Low,Close,Volume,WAP,Count,Minute,Tesla3,Tesla6,Tesla9,Buy,Sell,NoDecision,Execute,NoExecute] = 16 features

    public static final int VECTOR_SIZE = 16; // number of features for a stock data

    private final int miniBatchSize; // mini-batch size
    private final int exampleLength; // default 22, say, 22 working days per month

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
        List<StockData> stockDataList = readStockDataFromFile(filename, symbol);
        this.miniBatchSize = miniBatchSize;
        this.exampleLength = exampleLength;
        this.category = category;
        int split = (int) Math.round(stockDataList.size() * splitRatio);
        LoggingUtils.print(MessageFormat.format("Splitting data at index {0} for training and testing", split));

        train = stockDataList.subList(0, split);
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
        return maxArray[category.featureIndex];
    }

    public double getMinNum(PriceCategory category) {
        return minArray[category.featureIndex];
    }


    // Everything should be normalized
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
                for (int k = 0; k < VECTOR_SIZE; k++) {
                    PriceCategory priceCategory = PriceCategory.fromFeatureIndex(k);
                    if (priceCategory != PriceCategory.ALL) {
                        input.putScalar(new int[]{index, k, c}, getNormalizedValue(curData, priceCategory));
                    }
                }

                // Label features : 1 day ahead prediction
                nextData = train.get(i + 1);
                if (category == PriceCategory.ALL) {
                    // Put all the features as labels
                    for (int k = 0; k < VECTOR_SIZE; k++) {
                        PriceCategory priceCategory = PriceCategory.fromFeatureIndex(k);
                        if (priceCategory != PriceCategory.ALL) {
                            label.putScalar(new int[]{index, k, c}, getNormalizedValue(nextData, priceCategory));
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
        if (this.category == PriceCategory.ALL) return VECTOR_SIZE;
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

    // Input should be normalized
    // Label should not be normalized => Used directly for comparison
    private List<Pair<INDArray, INDArray>> generateTestDataSet(List<StockData> stockDataList) {
        int window = exampleLength + predictLength;
        List<Pair<INDArray, INDArray>> test = new ArrayList<>();
        for (int i = 0; i < stockDataList.size() - window; i++) {
            INDArray input = Nd4j.create(new int[]{exampleLength, VECTOR_SIZE}, 'f');
            for (int j = i; j < i + exampleLength; j++) {
                StockData stock = stockDataList.get(j);
                for (int k = 0; k < VECTOR_SIZE; k++) {
                    PriceCategory priceCategory = PriceCategory.fromFeatureIndex(k);
                    if (priceCategory != PriceCategory.ALL) {
                        input.putScalar(new int[]{j - i, k}, getNormalizedValue(stock, priceCategory));
                    }
                }
            }

            // Label features : 1 day ahead prediction
            StockData stock = stockDataList.get(i + exampleLength);
            INDArray label;
            if (category == PriceCategory.ALL) {
                label = Nd4j.create(new int[]{VECTOR_SIZE}, 'f'); // ordering is set as 'f', faster construct
                for (int k = 0; k < VECTOR_SIZE; k++) {
                    PriceCategory priceCategory = PriceCategory.fromFeatureIndex(k);
                    if (priceCategory != PriceCategory.ALL) {
                        label.putScalar(new int[]{k}, getOrdinaryValue(stock, priceCategory));
                    }
                }
            } else {
                label = Nd4j.create(new int[]{1}, 'f');
                label.putScalar(new int[]{0}, getOrdinaryValue(stock, category));
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
                if (i == PriceCategory.BUY.featureIndex ||
                        i == PriceCategory.SELL.featureIndex ||
                        i == PriceCategory.NO_DECISION.featureIndex ||
                        i == PriceCategory.EXECUTE.featureIndex ||
                        i == PriceCategory.NO_EXECUTE.featureIndex
                ) {
                    maxArray[i] = 1;
                    minArray[i] = 0;
                }
            }
            // load all elements in a list,
            List<String[]> list = new CSVReader(new FileReader(filename)).readAll();
            System.out.println(MessageFormat.format("FileContent Reading, Lines Count {0}", list.size()));
            boolean headerSkipped = false;
            boolean firstLinePrinted = false;
            int skippedColumns = 2;
            for (int j = 0; j < list.size(); j++) {
                String[] arr = list.get(j);
                if (!headerSkipped) {
                    headerSkipped = true;
                    continue;
                }
                double[] nums = new double[VECTOR_SIZE];

                for (int i = 0; i < arr.length - skippedColumns; i++) {
                    var value = arr[i + skippedColumns].trim().toLowerCase();

                    // Parsing Decision Column
                    if (i + skippedColumns == PriceCategory.BUY.csvIndex) {
                        if (value.isEmpty() || value.isBlank() || value.equals("sell") || value.equals("buy") || value.equals("no")
                        ) {
                            switch (value) {
                                case "no", "" -> {
                                    nums[PriceCategory.BUY.featureIndex] = 0;
                                    nums[PriceCategory.SELL.featureIndex] = 0;
                                    nums[PriceCategory.NO_DECISION.featureIndex] = 1;
                                }
                                case "sell" -> {
                                    nums[PriceCategory.BUY.featureIndex] = 0;
                                    nums[PriceCategory.SELL.featureIndex] = 1;
                                    nums[PriceCategory.NO_DECISION.featureIndex] = 0;
                                }
                                case "buy" -> {
                                    nums[PriceCategory.BUY.featureIndex] = 1;
                                    nums[PriceCategory.SELL.featureIndex] = 0;
                                    nums[PriceCategory.NO_DECISION.featureIndex] = 0;
                                }
                                default -> throw new RuntimeException(MessageFormat.format("String Value {0} is not parsed Correctly", value));
                            }
                        } else {
                            throw new RuntimeException(MessageFormat.format("String Value {0} is not parsed Correctly", value));
                        }

                        // Parsing Execute Column
                    } else if (i + skippedColumns == PriceCategory.EXECUTE.csvIndex) {
                        if (value.isEmpty() || value.isBlank() || value.equals("execute") || value.equals("no")
                        ) {
                            switch (value) {
                                case "no", "" -> {
                                    nums[PriceCategory.EXECUTE.featureIndex] = 0;
                                    nums[PriceCategory.NO_EXECUTE.featureIndex] = 1;
                                }
                                case "execute" -> {
                                    nums[PriceCategory.EXECUTE.featureIndex] = 1;
                                    nums[PriceCategory.NO_EXECUTE.featureIndex] = 0;
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

                var stockData = new StockData(
                        arr[0], arr[1], // Currency, Date
                        nums[0], nums[1], nums[2], nums[3], nums[4], // Open, High, Low, Close, Volume
                        nums[5], nums[6], nums[7], // WAP, Count, Minute
                        nums[8], nums[9], nums[10], // Tesla3, Tesla6, Tesla9
                        nums[11], nums[12], // buy, sell
                        nums[14] // execute
                );
                if (!firstLinePrinted) {
                    LoggingUtils.print(MessageFormat.format("For verification: Features arrays {0}", Arrays.toString(nums)));
                    LoggingUtils.print(MessageFormat.format("For verification: Stock {0}", stockData));
                    firstLinePrinted = true;
                }
                stockDataList.add(stockData);
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


    public double getOrdinaryValue(StockData stock, PriceCategory priceCategory) {

        return switch (priceCategory) {
            case OPEN -> stock.getOpen();
            case HIGH -> stock.getHigh();
            case LOW -> stock.getLow();
            case CLOSE -> stock.getClose();
            case VOLUME -> stock.getVolume();
            case WAP -> stock.getWap();
            case COUNT -> stock.getCount();
            case MINUTE -> stock.getMinute();
            case TESLA3 -> stock.getTesla3();
            case TESLA6 -> stock.getTesla6();
            case TESLA9 -> stock.getTesla9();
            case BUY, SELL, NO_DECISION -> decisionToFeature(priceCategory, stock);
            case EXECUTE, NO_EXECUTE -> executeToFeature(priceCategory, stock);
            default -> throw new RuntimeException(MessageFormat.format("Price Category {0} is not parsed Correctly", priceCategory));
        };
    }

    public double getNormalizedValue(StockData stock, PriceCategory priceCategory) {

        int index = priceCategory.featureIndex;
        return switch (priceCategory) {
            case OPEN -> (stock.getOpen() - minArray[index]) / (maxArray[index] - minArray[index]);
            case HIGH -> (stock.getHigh() - minArray[index]) / (maxArray[index] - minArray[index]);
            case LOW -> (stock.getLow() - minArray[index]) / (maxArray[index] - minArray[index]);
            case CLOSE -> (stock.getClose() - minArray[index]) / (maxArray[index] - minArray[index]);
            case VOLUME -> (stock.getVolume() - minArray[index]) / (maxArray[index] - minArray[index]);
            case WAP -> (stock.getWap() - minArray[index]) / (maxArray[index] - minArray[index]);
            case COUNT -> (stock.getCount() - minArray[index]) / (maxArray[index] - minArray[index]);
            case MINUTE -> (stock.getMinute() - minArray[index]) / (maxArray[index] - minArray[index]);
            case TESLA3 -> (stock.getTesla3() - minArray[index]) / (maxArray[index] - minArray[index]);
            case TESLA6 -> (stock.getTesla6() - minArray[index]) / (maxArray[index] - minArray[index]);
            case TESLA9 -> (stock.getTesla9() - minArray[index]) / (maxArray[index] - minArray[index]);
            case BUY, NO_DECISION, SELL -> decisionToFeature(priceCategory, stock);
            case EXECUTE, NO_EXECUTE -> executeToFeature(priceCategory, stock);
            default -> throw new RuntimeException(MessageFormat.format("Price Category {0} is not defined", priceCategory));
        };
    }
}


