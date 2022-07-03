# MachineLearningTradingBot

The purpose of this challenge is to improve the Returns of the trading bot by using Machine learning. To do this 2 classification models were developed SVM model and Linear Regression Model. For each model I tried different itteration were run with different values for the short rolling winodw, long rolling window and duration of (slice) of the training set. The accuracy stayed with 0.53 to 0.56. The ratio of the Stratgic (machine learning based) returns to actual returns, veried quite a bit. The best results were for using the SVM model with parameters short_window=4, long_window=100, duration= 6 months. 

I have documented the details below. I have also put the plots showing the Stratgic Returns compared to Actual returns for each itteration in the Resources folder. The title to each flow does indicate the values of short window, long window and training duration. Please review the png files in the Resources folders




# Training Begin 2015-04-02 15:00:00   Training End 2015-07-02 15:00:00   Train with 3 Months data

precision    recall  f1-score   support

        -1.0       0.43      0.04      0.07      1804
         1.0       0.56      0.96      0.71      2288

    accuracy                           0.55      4092
   macro avg       0.49      0.50      0.39      4092
weighted avg       0.50      0.55      0.43      4092

Stratgic Returns = 1.520   Actual Return = 1.389

# Training Begin 2015-04-02 15:00:00   Training End 2015-08-02 15:00:00   Train with 4 Months data

precision    recall  f1-score   support

        -1.0       0.42      0.04      0.07      1779
         1.0       0.56      0.96      0.71      2270

    accuracy                           0.55      4049
   macro avg       0.49      0.50      0.39      4049
weighted avg       0.50      0.55      0.43      4049

Stratgic Returns = 1.427   Actual Return = 1.456

# Training Begin 2015-04-02 15:00:00   Training End 2015-10-02 15:00:00   Train with 6 Months data

precision    recall  f1-score   support

        -1.0       0.44      0.02      0.04      1732
         1.0       0.56      0.98      0.71      2211

    accuracy                           0.56      3943
   macro avg       0.50      0.50      0.38      3943
weighted avg       0.51      0.56      0.42      3943

Stratgic Returns = 1.845   Actual Return = 1.538

# Training Begin 2015-04-02 15:00:00   Training End 2016-01-02 15:00:00   Train with 9 Months data

precision    recall  f1-score   support

        -1.0       0.45      0.31      0.37      1616
         1.0       0.57      0.70      0.63      2088

    accuracy                           0.53      3704
   macro avg       0.51      0.51      0.50      3704
weighted avg       0.52      0.53      0.51      3704

Stratgic Returns = 1.603   Actual Return = 1.649

# Training Begin 2015-04-02 15:00:00   Training End 2015-05-02 15:00:00   Train with 1 Month data

precision    recall  f1-score   support

        -1.0       0.37      0.03      0.06      1828
         1.0       0.56      0.96      0.70      2324

    accuracy                           0.55      4152
   macro avg       0.47      0.49      0.38      4152
weighted avg       0.48      0.55      0.42      4152

Stratgic Returns = 1.370   Actual Return = 1.293

## Impact of different Fast and Slow Rolling Windows

Fast 20 Slow 120

precision    recall  f1-score   support

        -1.0       0.43      0.13      0.20      1791
         1.0       0.56      0.86      0.68      2278

    accuracy                           0.54      4069
   macro avg       0.50      0.50      0.44      4069
weighted avg       0.50      0.54      0.47      4069


## Regression Model, Short Window=4, Long Window=100, Training Duration 3 Months

precision    recall  f1-score   support

        -1.0       0.44      0.33      0.38      1804
         1.0       0.56      0.66      0.61      2288

    accuracy                           0.52      4092
   macro avg       0.50      0.50      0.49      4092
weighted avg       0.51      0.52      0.51      4092

Stratgic Returns = 1.15 Actual Return = 1.389

## Regression Model, Short Window=4, Long Window=100, Training Duration 4 Months

precision    recall  f1-score   support

        -1.0       0.44      0.26      0.32      1779
         1.0       0.56      0.75      0.64      2270

    accuracy                           0.53      4049
   macro avg       0.50      0.50      0.48      4049
weighted avg       0.51      0.53      0.50      4049

Stratgic Returns = 1.2 Actual Return = 1.458

## Regression Model, Short Window=4, Long Window=100, Training Duration 6 Months

precision    recall  f1-score   support

        -1.0       0.52      0.03      0.06      1732
         1.0       0.56      0.98      0.71      2211

    accuracy                           0.56      3943
   macro avg       0.54      0.50      0.39      3943
weighted avg       0.54      0.56      0.43      3943

Stratgic Returns = 1.575 Actual Return = 1.560

## Regression Model, Short Window=4, Long Window=100, Training Duration 9 Months

precision    recall  f1-score   support

        -1.0       0.51      0.05      0.09      1616
         1.0       0.57      0.96      0.71      2088

    accuracy                           0.56      3704
   macro avg       0.54      0.51      0.40      3704
weighted avg       0.54      0.56      0.44      3704

Stratgic Returns = 1.452 Actual Return = 1.648

## SVC Model, Short Window=20, Long Window=120, Training Duration 3 Months

 precision    recall  f1-score   support

        -1.0       0.48      0.03      0.05      1793
         1.0       0.56      0.98      0.71      2284

    accuracy                           0.56      4077
   macro avg       0.52      0.50      0.38      4077
weighted avg       0.53      0.56      0.42      4077

Stratgic Returns = 1.338 Actual Return = 1.494

## Regression Model, Short Window=20, Long Window=120, Training Duration 3 Months

precision    recall  f1-score   support

        -1.0       0.44      0.54      0.48      1793
         1.0       0.55      0.45      0.50      2284

    accuracy                           0.49      4077
   macro avg       0.49      0.49      0.49      4077
weighted avg       0.50      0.49      0.49      4077

Stratgic Returns = 0.476 Actual Return = 1.494

## SVC Model, Short Window=20, Long Window=120, Training Duration 6 Months
 precision    recall  f1-score   support

        -1.0       0.44      0.02      0.05      1722
         1.0       0.56      0.98      0.71      2196

    accuracy                           0.56      3918
   macro avg       0.50      0.50      0.38      3918
weighted avg       0.51      0.56      0.42      3918

Stratgic Returns = 1.316 Actual Return = 1.489

## Regression Model, Short Window=20, Long Window=120, Training Duration 3 Months

precision    recall  f1-score   support

        -1.0       0.49      0.10      0.16      1722
         1.0       0.57      0.92      0.70      2196

    accuracy                           0.56      3918
   macro avg       0.53      0.51      0.43      3918
weighted avg       0.53      0.56      0.46      3918

Stratgic Returns = 1.75 Actual Return = 1.489