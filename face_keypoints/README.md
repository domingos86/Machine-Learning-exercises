This subproject aims to learn https://www.kaggle.com/c/facial-keypoints-detection

Most of the code was taken from this nice tutorial by Daniel Nouri http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/

Main changes were:
- Using 3/4 of the filters for the convolutional layers,
- Expanded filters (1st layer 5x5, remaining layers 3x3),
- 3/4 of the nodes on last dense layer,
- applying rotations for data augmentation.

Part of the rotations code was inspired by https://github.com/rakeshvar/theanet/blob/master/theanet/layer/inlayers.py
