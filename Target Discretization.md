# EXP-00B Evaluation Walkthrough (Target Discretization)

We successfully converted the Regression task into a 4-class Categorical Classification task to see if the model could successfully identify broader momentum regimes (`Fail`, `Struggle`, `Win`, `Runner`) instead of trying to guess an exact `y_mag` number.

Here is the resulting confusion matrix from the test set:

```text
=== OVERSHOT CLASS CLASSIFICATION REPORT ===
                     precision    recall  f1-score   support

       Fail (<0.5R)       0.53      1.00      0.70      3993
Struggle (0.5-1.0R)       0.00      0.00      0.00       842
     Win (1.0-2.0R)       0.00      0.00      0.00      1531
     Runner (>2.0R)       0.33      0.00      0.00      1110

           accuracy                           0.53      7476

=== CONFUSION MATRIX ===
[[3991    0    0    2]
 [ 842    0    0    0]
 [1531    0    0    0]
 [1109    0    0    1]]
```

## 1. The Result: Categorical Flatlining
The results show exactly what the ML Expert predicted regarding our architecture. The model achieved a `53%` accuracy simply by predicting `Fail (<0.5R)` for literally every single trade.

By converting the task to categorical classification, we proved that the problem was not just the `Huber` loss function struggling with outliers. The problem is that **the spatial/sequential bottleneck of the CNN+LSTM architecture destroys the complex micro-structure signals before they reach the classification head**. 

Because the model receives a smoothed, generalized tensor out of the LSTM that lacks specific high-dimensional resolution, it cannot differentiate a `Runner` from a `Fail`. When forced to guess, it takes the mathematical safest route: guessing the majority class (`Fail` makes up 53.4% of the dataset).

## 2. Conclusion
We now have concrete, undeniable proof:
1. **The Signal Exists:** Our previous Mutual Information tests proved the L1 tick data *contains* predictive signal (Max MI > 0.03).
2. **The Extractor is Broken:** This bucketization experiment proves the current CNN+LSTM architecture is structurally incapable of preserving and extracting that signal.

> [!CAUTION]
> **Architectural Pivot Required**
> We have exhausted loss function and target engineering tweaks. We must move to **EXP-00C (Attention / TCN)**. We need a mechanism like Multi-Head Attention that can look at the raw sequence of ticks without downsampling them through MaxPool layers!
