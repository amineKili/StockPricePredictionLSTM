package com.packt.javadl.priceprediction;

import com.packt.javadl.priceprediction.neuralnetwork.RecurrentNets;
import com.packt.javadl.priceprediction.representation.PriceCategory;
import com.packt.javadl.priceprediction.representation.StockDataSetIterator;
import com.packt.javadl.priceprediction.utils.Pair;
import com.packt.javadl.priceprediction.utils.PlotUtil;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.text.MessageFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.NoSuchElementException;

public class StockPricePrediction {
    private static final int exampleLength = 30; // time series length, assume 22 working days per month
    private static StockDataSetIterator iterator;

    public static void main(String[] args) throws IOException {

        String file = "/Users/amine/Downloads/B010335_07_Codes/StockPricePrediction/data/AUD.csv";

        String symbol = "AUD"; // stock name

        int batchSize = 100; // mini-batch size

        double splitRatio = 0.8; // 80% for training, 20% for testing

        // TODO : Increase to 100 in production
        int epochs = 100; // training epochs

        print("Creating dataSet iterator...");

        //Change to ALL for LSTM to generate All fields or Use a specific Field
        PriceCategory category = PriceCategory.CLOSE;

        iterator = new StockDataSetIterator(file, symbol, batchSize, exampleLength, splitRatio, category);
        print("Loading test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        //TODO : change from fullLstmNetwork to lightLstmNetwork for dev
        print("Building LSTM networks...");
        MultiLayerNetwork net = RecurrentNets.fullLstmNetwork(iterator.inputColumns(), iterator.totalOutcomes());

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage(); //Alternative: new FileStatsStorage(File) - see UIStorageExample

        int listenerFrequency = 1;
        net.setListeners(new StatsListener(statsStorage, listenerFrequency));

        //Print Training Score
        net.addListeners(new ScoreIterationListener(100));

        print("Training LSTM network...");
        for (int i = 0; i < epochs; i++) {
            print("Epoch " + i);
            while (iterator.hasNext()) net.fit(iterator.next()); // fit model using mini-batch data
            iterator.reset(); // reset iterator
            net.rnnClearPreviousState(); // clear previous state
        }

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers_before_saving = net.getLayers();
        int totalNumParams_before_saving = 0;
        for (int i = 0; i < layers_before_saving.length; i++) {
            int nParams = layers_before_saving[i].numParams();
            print("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams_before_saving += nParams;
        }
        print("Total number of network parameters: " + totalNumParams_before_saving);

        print("Saving model...");
        File locationToSave = new File("data/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
        // saveUpdater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, true);

        print("Restoring model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        //print the score with every 1 iteration
        net.setListeners(new ScoreIterationListener(1));

        //Print the  number of parameters in the network (and for each layer)
        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            print("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        print("Total number of network parameters: " + totalNumParams);

        print("Evaluating...");
        if (category.equals(PriceCategory.ALL)) {
            INDArray max = Nd4j.create(iterator.getMaxArray());
            INDArray min = Nd4j.create(iterator.getMinArray());
            predictAllCategories(net, test, max, min);
        } else {
            double max = iterator.getMaxNum(category);
            double min = iterator.getMinNum(category);
            predictPriceOneAhead(net, test, max, min, category);
        }
        print("Done...");
    }

    /**
     * Predict one feature of a stock one-day ahead
     */
    private static void predictPriceOneAhead(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min, PriceCategory category) {
        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];

        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
            actuals[i] = testData.get(i).getValue().getDouble(0);
        }

        RegressionEvaluation eval = net.evaluateRegression(iterator);
        print(eval.stats());

        print("Printing predicted and actual values...");
        print("Predict, Actual");

        // TODO: Uncomment this section if needs to read Actual/Predicted Values
        for (int i = 0; i < predicts.length; i++)
            print(predicts[i] + "," + actuals[i]);

        print("Plottig...");
        PlotUtil.plot(predicts, actuals, String.valueOf(category));
    }

    /**
     * Predict all the features (open, close, low, high prices and volume) of a stock one-day ahead
     */
    private static void predictAllCategories(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min) {
        INDArray[] predicts = new INDArray[testData.size()];
        INDArray[] actuals = new INDArray[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = testData.get(i).getValue();
        }

        RegressionEvaluation eval = net.evaluateRegression(iterator);
        print(eval.stats());

        for (int n = 0; n < StockDataSetIterator.VECTOR_SIZE; n++) {
            double[] pred = new double[predicts.length];
            double[] actu = new double[actuals.length];
            for (int i = 0; i < predicts.length; i++) {
                pred[i] = predicts[i].getDouble(n);
                actu[i] = actuals[i].getDouble(n);
            }
            String name;
            switch (n) {
                case 0:
                    name = "Stock OPEN";
                    break;
                case 1:
                    name = "Stock High Price";
                    break;
                case 2:
                    name = "Stock LOW Price";
                    break;
                case 3:
                    name = "Stock Close Price";
                    break;
                case 4:
                    name = "Stock VOLUME Amount";
                    break;
                case 5:
                    name = "Stock WAP";
                    break;
                case 6:
                    name = "Stock Count";
                    break;
                // Use minute to verify if model predict next minutes, if not the model need to be adjusted
                case 7:
                    name = "Stock Minutes";
                    break;
                case 8:
                    name = "Stock Tesla 3";
                    break;
                case 9:
                    name = "Stock Tesla 6";
                    break;
                case 10:
                    name = "Stock Tesla 9";
                    break;
                case 11:
                    name = "Stock BUY Decision";
                    break;
                case 12:
                    name = "Stock SELL Decision";
                    break;
                case 13:
                    name = "Stock NO Decision";
                    break;
                case 14:
                    name = "Stock Execute";
                    break;
                case 15:
                    name = "Stock NO Execute";
                    break;

                default:
                    throw new NoSuchElementException();
            }
            PlotUtil.plot(pred, actu, name);
        }
    }


    public static String getCurrentTimeForLogging() {
        return new SimpleDateFormat("yyyy-MM-dd HH:mm:ss").format(new Date());
    }

    public static void print(String message) {
        System.out.println(MessageFormat.format("[{0}] {1}", getCurrentTimeForLogging(), message));
    }
}
