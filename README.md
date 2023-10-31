<h1 align="center">
Convolutional Neural Network (CNN) for MNIST Image Classification
</h1>

This Python code implements a Convolutional Neural Network (CNN) for classifying hand-written digit images from the MNIST database. The goal is to accurately identify the digits from 0 to 9 present in the images.

## Overview

The neural network implementation comprises the following components:

***Data Loading and Preprocessing***
- Loads the MNIST dataset, preprocesses images, and prepares them for training.

***Convolutional Neural Network***
- Defines the architecture of the CNN, including convolutional layers, pooling layers, fully connected layers, and output layers.
- Trains the network to recognize and classify digit images.

***Training and Evaluation***
- Manages the training process by optimizing the network's weights using backpropagation and gradient descent.
- Evaluates the model's performance on a test set, calculating accuracy and loss metrics.

## Usage

The code can be utilized by setting parameters such as the network architecture, hyperparameters like **learning rate**, **batch size**, **number of epochs**, etc.

The CNN iteratively learns from the MNIST dataset, adjusting its internal parameters through forward and backward propagation to accurately classify hand-written digits.

You can try it out in COLAB:

[COLAB LINK](https://colab.research.google.com/drive/15nO-fPEy0w4TqM-P68pySgSVTexn-0zF?usp=sharing)

## CODE
Importation des bibliothèques :
```python
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix,classification_report
import scikitplot as skplt
from keras.models import Sequential,Model
from tensorflow.keras.layers import Input, InputLayer,Reshape,Conv2D,  MaxPooling2D,Dense, Flatten
from keras.layers.core import Dense,Dropout, Activation
```

Load the "MNIST" image dataset:
```python
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Taille de:")
# Importation des bibliothèques print("- Ensemble d'apprentissage:",format(X_train.shape))
print("- Ensemble de test:",format(X_test.shape))
```

Displaying some images with their labels:
```python
for i in range(9):
    plt.subplot(3,3,i+1)# Affichage de quelque image avec leurs labels
    plt.imshow(X_train[i], cmap='gray', 	             			interpolation='none')
    plt.subplots_adjust(hspace=1,      				 wspace=0.5)
    plt.title("Class {}".format(y_train[i]))
```

Prepare our data for training:
```python
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
nb_classes =10
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)
print(y_test[1])
print(Y_test[1])
```

Creation of the neural network (Sequential):
```python
img_shape_full = (28,28,1)
nb_classes =10
#commencer la construction du modéle séquentiel de keras
model = Sequential()
# Ajouter une couche d'entrée qui doit étre un tuple contenant la taille de l'image
model.add(InputLayer(input_shape=(784,)))
# l'entree est un vecteur de 784 mais les couche convolutionnelle attent des image avec forme (28,28,1)
model.add(Reshape(img_shape_full))
#Premiere couche convolutionnelle avec activation Relu et max-pooling
model.add(Conv2D(kernel_size=5 ,strides = 1, filters=16,padding='same',activation='relu',name='layer_conv1'))
model.add(MaxPooling2D(pool_size=2 ,strides=2))
#Deuxieme couche convolutionnelle avec activation Relu et max-pooling
model.add(Conv2D(kernel_size=5 ,strides = 1, filters=36,padding='same',activation='relu',name='layer_conv2'))
model.add(MaxPooling2D(pool_size=2 ,strides=2))
model.add(Flatten())
#fully-connected/dense layer avec Relu
model.add(Dense(128,activation='relu'))
#derniere fully connected
model.add(Dense(nb_classes, activation='softmax'))
# Création du réseau de neurones
```

Compilation of the model:
```python
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],     optimizer='adam')
```

Training:
```python
 model.fit(X_train, Y_train,batch_size=128, epochs=10,verbose=1)
```

Evaluation:
```python
score = m# Apprentissage# Création du réseau de neuronesodel.evaluate(X_test, Y_test, verbose=1)
for name, value in zip(model.metrics_names, score):
    			print(name, value)

```

Creation of the neural network (Model):
```python
inputs = Input(shape=(784,))
#variable utilisee pour la construction du reseau de neurones.
net = inputs
# l'entree est un vecteur de 784 mais les couche convolutionnelle attent des image avec forme (28,28,1)
img_shape_full = (28,28,1)
net = Reshape(img_shape_full)(net)
#Premiere couche convolutionnelle avec activation Relu et max-pooling
net = Conv2D(kernel_size=5 ,strides = 1, filters=16,
             padding='same',activation='relu',name='layer_conv1')(net)
net = Conv2D(kernel_size=5 ,strides = 1, filters=36,
                 padding='same',activation='relu',
                 name='layer_conv2')(net)
net = MaxPooling2D(pool_size=2 ,strides=2)(net)

net = Flatten()(net)

#Dense 1
net = Dense(128,activation='relu')(net)
#Dense 2
net = Dense(nb_classes, activation='softmax')(net)

#sortie
outputs = net
model2 = Model(inputs = inputs, outputs=outputs)
```

Compilation of Model 2:
```python
model2.compile(loss='categorical_crossentropy',metrics=['accuracy'],     optimizer='adam')
```

Training Model 2:
```python
 model2.fit(X_train, Y_train,batch_size=128, epochs=10,verbose=1)
```

Evaluation Model 2:
```python
score = model2# Évaluation Model2.evaluate(X_test, Y_test, verbose=1)
for name, value in zip(model2.metrics_names, score):
    			print(name, value)
```

Displaying some images with their labels and predictions:
```python
# La fonction « predict_classes » produit la classe de probabilité la plus élevée selon le classificateur formé pour chaque exemple d'entrée.

predicted_classes = np.argmax(model.predict(X_test),axis=1)
correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray',                			interpolation='none')
    plt.subplots_adjust(hspace=1, wspace=1.5)
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect],  			y_test[incorrect]))
```

Confusion matrix & classification report:
```python
import scikitplot as skplt
skplt.metrics.plot_confusion_matrix(y_test, predicted_classes,               normalize=False)
plt.show()
print (confusion_matrix(y_test, predicted_classes))
print (classification_report(y_test, predicted_classes, target_names=None))
```

## License

This project is created and owned by Aziz Tarous.

[LinkedIn](https://www.linkedin.com/in/aziz-tarous/)

## 🔗 Links

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://eportfolio-host.web.app) &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp;   [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/aziz-tarous/) &nbsp;   &nbsp;   &nbsp;   &nbsp;   &nbsp;  [![frontend](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/aziztarous1999/Tunisair_SpringBoot_Frontend)
