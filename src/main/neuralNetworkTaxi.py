import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

# nazwa pliku moze byc rozna za każdym tworzeniem go w Sparku
file_name = "/Users/big0ne/Downloads/ZaliczenieESI/src/main/resources/hourAndPlace.csv/part-00000-793e65f4-04a2-4030-bdf4-42a9d1d3b2ea-c000.csv"
column_to_remove = 'trip_distance'

zbior_treningowy = pd.read_csv(file_name, low_memory=False)
print(zbior_treningowy.head())

features = zbior_treningowy.copy()
labels = zbior_treningowy.pop('trip_distance')

features = np.array(features)
print(features)

model_podrozy = tf.keras.Sequential([
    layers.Dense(64),
    layers.Dense(1)
])

model_podrozy.compile(loss=tf.keras.losses.MeanSquaredError(),
                      optimizer=tf.keras.optimizers.Adam())

# trenowanie modelu
model_podrozy.fit(features, labels, epochs=10)

normalizacja = layers.Normalization()
normalizacja.adapt(features)

model_znormalizowany = tf.keras.Sequential([
    normalizacja,
    layers.Dense(64),
    layers.Dense(1)
])

model_znormalizowany.compile(loss=tf.keras.losses.MeanSquaredError(),
                             optimizer=tf.keras.optimizers.Adam())

model_znormalizowany.fit(features, labels, epochs=10)
print(model_znormalizowany)

raw_dataset = pd.read_csv(file_name)
dataset = raw_dataset.copy()
print(dataset.tail())

dataset = dataset.dropna()

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

# usuniecie kolumny z dystansem
train_labels = train_features.pop(column_to_remove)
test_labels = test_features.pop(column_to_remove)

train_dataset.describe().transpose()[['mean', 'std']]

normalizator = tf.keras.layers.Normalization(axis=-1)
normalizator.adapt(np.array(train_features))
print(normalizator.mean.numpy())

trip_distance = np.array(train_dataset[column_to_remove])
trip_distance_normal = layers.Normalization(input_shape=[1, ], axis=None)
trip_distance_normal.adapt(trip_distance)

model = tf.keras.Sequential([
    trip_distance_normal,
    layers.Dense(units=1)
])

print(model.summary())

# model.compile(
#     optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
#     loss='mean_absolute_error')
#
# history = model.fit(
#     train_dataset[column_to_remove],
#     test_features,
#     epochs=100,
#     # Suppress logging.
#     verbose=0,
#     # Calculate validation results on 20% of the training data.
#     validation_split=0.2)
#
# hist = pd.DataFrame(history.history)
# hist['epoch'] = history.epoch
# print(hist.tail())
#
# test_results = {}
#
# test_results['distance_model'] = model.evaluate(
#     test_features[column_to_remove],
#     test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = model.predict(x)


def plot_distance_prediction(x, y):
    plt.scatter(train_dataset[column_to_remove], train_dataset['hour_of_day'], label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Distance')
    plt.ylabel('Hour')
    plt.legend()
    plt.show()


plot_distance_prediction(x, y)
