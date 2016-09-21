This algorithm attempts to solve the Digit Recognizer problem in https://www.kaggle.com/c/digit-recognizer/

It uses PCA to 40 dimensions, followed by k nearest neighbors (standard k=5).

According to the Kaggle submission, it obtained an accuracy of 97.3%.

It should be noted that the PCA was done including both the training and test sets.