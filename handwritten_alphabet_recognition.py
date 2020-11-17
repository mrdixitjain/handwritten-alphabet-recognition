import pandas as pd 
import numpy as np 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# data used here is taken from kaggle.com(path of which is as below)
# either download the data from kaggle in your local directory or 
# run this code on kaggle otherwise it'll throw an error.
data = pd.read_csv("/kaggle/input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv")

# seperate labels from features
# labels are from 0-25 for a-z
labels = data['0']
data.drop('0', inplace=True, axis = 1)

# split the data into three sets train, test, validation
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.40, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.4, random_state=33)

# un comment below line if memory became full and re-run
# data = None

# reshape the data as per model
X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_val = np.array(X_val)
X_val = X_val.reshape(X_val.shape[0], 28, 28, 1)

y_test = np.array(y_test)
y_train = np.array(y_train)
y_val = np.array(y_val)

# normalize the features
# as only pixel intensities are there, dividing by 255 will work.
X_train = X_train/255
X_test = X_test/255
X_val = X_val/255

# design your models architecture
# you may change it as per your convenience
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Dropout(.5))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(26))

print(model.summary())

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# change the #epochs as per your convenience
history = model.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_val, y_val))


# Retrieve a list of accuracy results on training and validation data
# sets for each training epoch
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Retrieve a list of list results on training and validation data
# sets for each training epoch
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and validation accuracy')

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and validation loss')

# check accuracy and loss over test set
print(model.evaluate(X_test, y_test))

# plotting first 10 alphabets and their predictions
for i in range(10) :
	img = X_test[i]*255
	img = np.reshape(img, (28, 28))

	img1 = np.reshape(img, (1, 28, 28, 1))
	predictions = model.predict(img1)
	predict = np.argmax(predictions, axis=1)
	fig = plt.figure()
	plt.imshow(img)
	fig.suptitle("Predicted : " + chr(ord('a') + predict), fontsize=10)
	plt.show()
