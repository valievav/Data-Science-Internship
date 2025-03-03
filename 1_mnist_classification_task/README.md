**TASK 1. Image classification + OOP**

In this task, you need to use a publicly available simple MNIST dataset and build 3 classification
models around it. It should be the following models:
1) Random Forest;
2) Feed-Forward Neural Network;
3) Convolutional Neural Network;

Each model should be a separate class that implements MnistClassifierInterface with 2
abstract methods - train and predict. Finally, each of your three models should be hidden under
another MnistClassifier class. MnistClassifer takes an algorithm as an input parameter.
Possible values for the algorithm are: cnn, rf, and nn for the three models described above.

The solution should contain:
* Interface for models called MnistClassifierInterface.
* 3 classes (1 for each model) that implement MnistClassifierInterface.
* MnistClassifier, which takes as an input parameter the name of the algorithm and provides predictions 
with exactly the same structure (inputs and outputs) not depending
on the selected algorithm.

##########################

**Solution**:

To check results, please run **main.py** to execute **run_models()**, which prints accuracy 
from highest to lowest for 3 models.

Solution uses Factory method, where MnistClassifier retrieves data and instantiates requested model.
Used sklearn for Random Forest and keras for Feed-Forward Neural Network (FNN) and Convolutional Neural Network (CNN).
