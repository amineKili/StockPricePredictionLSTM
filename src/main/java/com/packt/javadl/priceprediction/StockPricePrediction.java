package com.packt.javadl.priceprediction;

import com.packt.javadl.priceprediction.neuralnetwork.RecurrentNets;
import com.packt.javadl.priceprediction.representation.PriceCategory;
import com.packt.javadl.priceprediction.representation.StockDataSetIterator;
import com.packt.javadl.priceprediction.utils.LoggingUtils;
import com.packt.javadl.priceprediction.utils.Pair;
import com.packt.javadl.priceprediction.utils.PlotUtil;
import org.deeplearning4j.eval.RegressionEvaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.PerformanceListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.text.MessageFormat;
import java.util.List;

public class StockPricePrediction {
    private static final int exampleLength = 30; // time series length, assume 22 working days per month
    private static StockDataSetIterator iterator;

    public static void main(String[] args) throws IOException {

        String file = "/Users/amine/Downloads/B010335_07_Codes/StockPricePrediction/data/AUD.csv";

        String symbol = "AUD"; // stock name

        int batchSize = 128; // mini-batch size

        double splitRatio = 0.8; // 80% for training, 20% for testing

        // TODO : Increase to 100 in production
        int epochs = 3; // training epochs

        LoggingUtils.print("Creating dataSet iterator...");

        //Change to ALL for LSTM to generate All fields or Use a specific Field
        PriceCategory category = PriceCategory.EXECUTE;

        iterator = new StockDataSetIterator(file, symbol, batchSize, exampleLength, splitRatio, category);
        LoggingUtils.print("Loading test dataset...");
        List<Pair<INDArray, INDArray>> test = iterator.getTestDataSet();

        //TODO : change from fullLstmNetwork to lightLstmNetwork for dev
        LoggingUtils.print("Building LSTM networks...");
        MultiLayerNetwork net = RecurrentNets.fullLstmNetwork(iterator.inputColumns(), iterator.totalOutcomes());

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains

        int listenerFrequency = 1;
        net.setListeners(new ScoreIterationListener(listenerFrequency), new PerformanceListener(listenerFrequency));


        LoggingUtils.print("Training LSTM network...");
        for (int i = 0; i < epochs; i++) {
            LoggingUtils.print("Epoch " + i);
            while (iterator.hasNext()) net.fit(iterator.next()); // fit model using mini-batch data
            iterator.reset(); // reset iterator
            net.rnnClearPreviousState(); // clear previous state
        }

        //LoggingUtils.print the  number of parameters in the network (and for each layer)
        Layer[] layers_before_saving = net.getLayers();
        int totalNumParams_before_saving = 0;
        for (int i = 0; i < layers_before_saving.length; i++) {
            int nParams = layers_before_saving[i].numParams();
            LoggingUtils.print("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams_before_saving += nParams;
        }
        LoggingUtils.print("Total number of network parameters: " + totalNumParams_before_saving);

        LoggingUtils.print("Saving model...");
        File locationToSave = new File("data/StockPriceLSTM_".concat(String.valueOf(category)).concat(".zip"));
        ModelSerializer.writeModel(net, locationToSave, true);

        LoggingUtils.print("Restoring model...");
        net = ModelSerializer.restoreMultiLayerNetwork(locationToSave);

        net.setListeners(new ScoreIterationListener(listenerFrequency));

        Layer[] layers = net.getLayers();
        int totalNumParams = 0;
        for (int i = 0; i < layers.length; i++) {
            int nParams = layers[i].numParams();
            LoggingUtils.print("Number of parameters in layer " + i + ": " + nParams);
            totalNumParams += nParams;
        }
        LoggingUtils.print("Total number of network parameters: " + totalNumParams);

        LoggingUtils.print("Evaluating...");
        if (category.equals(PriceCategory.ALL)) {
            INDArray max = Nd4j.create(iterator.getMaxArray());
            INDArray min = Nd4j.create(iterator.getMinArray());
            predictAllCategories(net, test, max, min);
        } else {
            double max = iterator.getMaxNum(category);
            double min = iterator.getMinNum(category);
            predictPriceOneAhead(net, test, max, min, category);
        }
        LoggingUtils.print("Done...");
    }

    /**
     * Predict one feature of a stock one-day ahead
     */
    private static void predictPriceOneAhead(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, double max, double min, PriceCategory category) {

        LoggingUtils.print(MessageFormat.format("Predicting one day ahead... category {0}, min {1}, max {2}", category, min, max));

        double[] predicts = new double[testData.size()];
        double[] actuals = new double[testData.size()];

        for (int i = 0; i < testData.size(); i++) {
            // Change this to get more than one step ahead
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getDouble(exampleLength - 1) * (max - min) + min;
            actuals[i] = testData.get(i).getValue().getDouble(0);

            // Fit model using last step of the sequence only
            net.fit(testData.get(i).getKey(), testData.get(i).getValue());
        }

        RegressionEvaluation eval = net.evaluateRegression(iterator);
        LoggingUtils.print(eval.stats());

        LoggingUtils.print("Predicted and actual values...");
        LoggingUtils.print("Predict, Actual");

        for (int i = 0; i < predicts.length; i++)
            LoggingUtils.print(predicts[i] + "," + actuals[i]);

        LoggingUtils.print("Plotting...");
        PlotUtil.plot(predicts, actuals, String.valueOf(category));
    }

    /**
     * Predict all the features (open, close, low, high prices and volume) of a stock one-day ahead
     */
    private static void predictAllCategories(MultiLayerNetwork net, List<Pair<INDArray, INDArray>> testData, INDArray max, INDArray min) {

        LoggingUtils.print(MessageFormat.format("Predicting one day ahead... for all categories, min {1}, max {2}", min, max));

        INDArray[] predicts = new INDArray[testData.size()];
        INDArray[] actuals = new INDArray[testData.size()];
        for (int i = 0; i < testData.size(); i++) {
            predicts[i] = net.rnnTimeStep(testData.get(i).getKey()).getRow(exampleLength - 1).mul(max.sub(min)).add(min);
            actuals[i] = testData.get(i).getValue();
        }

        RegressionEvaluation eval = net.evaluateRegression(iterator);
        LoggingUtils.print(eval.stats());

        for (int n = 0; n < StockDataSetIterator.VECTOR_SIZE; n++) {
            double[] pred = new double[predicts.length];
            double[] actu = new double[actuals.length];
            for (int i = 0; i < predicts.length; i++) {
                pred[i] = predicts[i].getDouble(n);
                actu[i] = actuals[i].getDouble(n);

            }
            String name;
            name = PriceCategory.fromFeatureIndex(n).name();
            LoggingUtils.print("Predicted and actual values...");
            LoggingUtils.print("Predict, Actual");
            for (int i = 0; i < pred.length; i++)
                LoggingUtils.print(pred[i] + "," + actu[i]);
            LoggingUtils.print("Plottig...");
            PlotUtil.plot(pred, actu, name);
        }
    }


}
