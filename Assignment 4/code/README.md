# Comp551-A4

## Packages used

* `ucimlrepo` (fetch_ucirepo): to get UCI datasets

* `numpy`

* `pandas`

* `sklearn.tree` (DecisionTreeClassifier): for decision tree structure

* `Tensorflow.keras` (Sequential and Dense): for neural network structure

* `category_encoders`: to encode categorical data to ordinal

* `sklearn.preprocessing` (LabelEncoder and LabelBinarizer): to encode labels to ordinal

* `sklearn.model_selection` (train_test_split): for training/testing

* `sklearn.metrics` (accuracy_score): to get accuracy

* `matplotlib.pyplot`

## Datasets

* [Iris](http://archive.ics.uci.edu/dataset/53/iris)

* [Haberman's Survival](http://archive.ics.uci.edu/dataset/43/haberman+s+survival)

* [Car Evaluation](http://archive.ics.uci.edu/dataset/19/car+evaluation)

* [Breast Cancer Wisconsin](http://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original)

* [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

* [Poker Hand](http://archive.ics.uci.edu/dataset/158/poker+hand)

* [German Credit](http://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)

* [Connect-4](http://archive.ics.uci.edu/dataset/26/connect+4)

* [Image Segmentation](http://archive.ics.uci.edu/dataset/50/image+segmentation)

* [Covertype](http://archive.ics.uci.edu/dataset/31/covertype)

## Instructions to replicate our results

### DNDT

Although results are already displayed on the notebook, it is possible to replicate results by running `DNDT.ipynb`. `train_neural_decision_tree` creates a decision tree classifier from the intermediate layers of a Keras model. Datasets are either retrived through `fetch_ucirepo` or through a `.csv` file in 'datasets'.

### DT

Although results are already displayed on the notebook, it is possible to replicate results by running `DT.ipynb`. `DecisionTreeClassifier` is used with two of the key hyper-parameters criterion as ‘gini’ and splitter as ‘best’, as per the paper. Datasets are either retrived through `fetch_ucirepo` or through a `.csv` file in 'datasets'.

### NN 

Although results are already displayed on the notebook, it is possible to replicate results by running `NN.ipynb`. `Sequential` is used to create a neural network of 2 hidden layers and 50 neurons each, as per the paper. 
