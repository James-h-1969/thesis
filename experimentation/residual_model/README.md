# Residual Neural Networks
The idea of a residual neural network is that instead of trying to solve for the function H(x), you 
try solve for the residual function F(x) = H(x) - x instead. This allows gradients to not explode or 
dissapear in large networks (https://www.geeksforgeeks.org/deep-learning/vanishing-and-exploding-gradients-problems-in-deep-learning/).

This repository is a implementation of a residual neural network that is found here:
https://www.geeksforgeeks.org/deep-learning/resnet18-from-scratch-using-pytorch/