This subproject aims to learn https://www.kaggle.com/c/facial-keypoints-detection

Most of the code was taken from this nice tutorial by Daniel Nouri http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

Main changes were:
- Using 3/4 of the filters for the convolutional layers,
- Expanded filters (1st layer 5x5, remaining layers 3x3),
- 3/4 of the nodes on last dense layer,
- applying rotations for data augmentation.

Part of the rotations code was inspired by https://github.com/rakeshvar/theanet/blob/master/theanet/layer/inlayers.py

Possible improvements going forward:
- Improve performance by running all the data augmentation on the GPU
- Find a way of allowing training with missing labels (so that a single network would suffice - having multiple networks is usually very redundant). Possible approaches are:
  - Have an automatic dropout layer at the end that hides missing labels
  - Start by training the network only for samples without missing labels, and then keep assigning the network prediction to the missing labels. If done well (at the batch level), it can have the same effect as the previous point.
- Increase number of filters in convolutional layers (assuming we're not using multiple networks)