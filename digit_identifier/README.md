This algorithm attempts to solve the Digit Recognizer problem in https://www.kaggle.com/c/digit-recognizer/

It uses PCA to 40 dimensions, followed by k nearest neighbors (standard k=5).

According to the Kaggle submission, it obtained an accuracy of 97.3%.

It should be noted that the PCA was done including both the training and test sets.

In the meantime I got a chance to read the book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen and learn about more accurate ways of recognizing handwritten digits. Namely, by using an ensemble of 5 neural networks composed of convolutional layers, max-pooling layers and dense layers, he was able to obtain an accuracy of 99.67% on the test set, with a good amount of the misclassifications attributable to very confusing handwritting (that a human eye would probably also misclassify). Given that it is a very extensively studied dataset, it seems more useful to concentrate my efforts on other machine learning tasks.
