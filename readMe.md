# LSTM Price Signal Predictor
## Inspired by LSTM Price prediction by Rezaul Karim

> Price Prediction Uses LSTM neural network build using Deeplearning4j.


![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)

Signal Price Predictor, predict execution signal given a set of inputs

- Prices ['open','close', 'high', 'low', 'wap']
- Volume
- Count: trades count during the time period
- Minute
- Tesla magic numbers ['tesla3', 'tesla6', 'tesla9']
- Decision: flag can be NO, BUY or SELL
- Execute: flag execute the decision or ignore it, can be EXECUTE or No.

## Features

- Save the trained model for quick usage
- Adjustable epoch, training and test training batch sizes.
- Any of the feature can be predicted. eg the systeme can be used to predict the close price or the execute flag.
- Neural network inside RecurrentNets is fully costumizable. Two implementations are provided as examples.

## Usage

Price Prediction LSTM

> Any changes in DataSet input will need adjustement in PriceCategory and StockDataIterator.

1. Put data in data/XXX.csv using the following format.

| Currency | YYYYMMDD_HHMMSS | Open | High | Low | Close | Volume | WAP | Count | Minute | Tesla3 | Tesla6 | Tesla9 | Decision | EXECUTE |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |


2. Change **filePath** variable in RunStockPricePrediction's main function.
3. Put the wanted feature in **outputCategory** variable in RunStockPricePrediction's main function.
4. Run __RunStockPricePrediction.java__


## Tech

LSTM Price Prediciton uses a number of open source projects to work properly:

- [LSTM](https://github.com/PacktPublishing/Java-Deep-Learning-Projects/blob/master/Chapter07/B010335_07_Codes.zip) - Rezaul Karim Implementation
- [DeepLearning4j](https://deeplearning4j.konduit.ai/) - Deeplearning4j for LSTM implementation
- [JFreeChart](https://www.jfree.org/) - Visualization solution.

## Installation

LSTM Price Predictor requires **Java 17** and **Maven** to run.

To build the project

```sh
mvn clean install
```
