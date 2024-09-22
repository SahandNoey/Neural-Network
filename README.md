## Neural Network implementation
### Task 1 & 2: Recognize numbers using Fully Connected Neural Networks and CNNs in Pytorch
- **ReLU** activation function
- Stochastic Gradient Descent(**SGD**) optimizer
- **Cross Entropy** Loss Function
- Dataset: **MNIST**
- Validation data accuracy 97%

![](https://github.com/SahandNoey/Neural-Network/blob/master/MNIST-Samples.jpg)
![](https://github.com/SahandNoey/Neural-Network/blob/master/Fully%20Connected%20Linear%20ReLU.png)

Validation Accuracy of Fully Connected NN<br><br><br><br>
![](https://github.com/SahandNoey/Neural-Network/blob/master/CNN_MNIST.png)

Validation Accuracy of CNN<br><br><br><br>

### Task 3: Recognize brain tumor in PyTorch
- Model: **ResNet50**
- Optimizer: **Adam**
- Loss Function: **Cross Entropy** 

![](https://github.com/SahandNoey/Neural-Network/blob/master/brain_sample_data.png)
Dataset Samples<br><br><br><br>

![](https://github.com/SahandNoey/Neural-Network/blob/master/resnet50_loss_function.png)

Learning Rate is higher than expected but eventually, loss decreased enough<br><br><br><br>


## Task 4 to 6: Implementing my own library for NNs
- **Tensor Class**: Included mathematical operations and ability to save gradients
- Implemented layer: Linear Layer
- Implemented Optimizers: **SGD**, **Momentum**, **Adam**, and **RMSprop**
- Implemented Loss Functions: **Mean Squared Error**, **Categorical Cross Entropy**
- Implemented Activation Functions: **ReLU**, **LeakyReLU**, **Tanh**, and **Sigmoid**
