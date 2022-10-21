import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional, Dropout

# Reading Powerball.csv
filename = "Powerball.csv"
with open(filename) as f:
    reader = csv.reader(f)
    header_row = next(reader)
    numbers = np.array([])
    dp = True

    for row in reader:
        if row[1] == "August, 21, 2021":
            dp = False

        if dp == False:
            number = int(row[2])
            numbers = np.append(numbers, number)
            number = int(row[3])
            numbers = np.append(numbers, number)
            number = int(row[4])
            numbers = np.append(numbers, number)
            number = int(row[5])
            numbers = np.append(numbers, number)
            number = int(row[6])
            numbers = np.append(numbers, number)
            number = int(row[7])
            numbers = np.append(numbers, number)
        else:
            number = int(row[2])
            numbers = np.append(numbers, number)
            number = int(row[3])
            numbers = np.append(numbers, number)
            number = int(row[4])
            numbers = np.append(numbers, number)
            number = int(row[5])
            numbers = np.append(numbers, number)
            number = int(row[6])
            numbers = np.append(numbers, number)
            number = int(row[7])
            numbers = np.append(numbers, number)
            number = int(row[8])
            numbers = np.append(numbers, number)
            number = int(row[9])
            numbers = np.append(numbers, number)
            number = int(row[10])
            numbers = np.append(numbers, number)
            number = int(row[11])
            numbers = np.append(numbers, number)
            number = int(row[12])
            numbers = np.append(numbers, number)
            number = int(row[13])
            numbers = np.append(numbers, number)

# Prepare/Generate data set
arr = numbers.reshape((math.floor(len(numbers) / 6)), 6)
df = pd.DataFrame(arr, columns=list('ABCDEF'))

# Normalizing data
scaler = StandardScaler().fit(df.values)
transformed_dataset = scaler.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)

# Defining hyper parameters of model
number_of_rows = df.values.shape[0]
window_length = math.floor(len(arr) * .9)
number_of_features = df.values.shape[1]

# Creating train dataset and labels for each row.
train = np.empty([number_of_rows - window_length, window_length, number_of_features], dtype=float)
label = np.empty([number_of_rows - window_length, number_of_features], dtype=float)
window_length = math.floor(len(arr) * .9)

for i in range(0, number_of_rows - window_length):
    train[i] = transformed_df.iloc[i:i+window_length, 0: number_of_features]
    label[i] = transformed_df.iloc[i+window_length: i+window_length+1, 0: number_of_features]

# LSTM model using TensorFlow backend
batch_size = 100
model = Sequential()
model.add(Bidirectional(LSTM(240, input_shape=(window_length, number_of_features), return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(240, input_shape=(window_length, number_of_features), return_sequences=True)))
model.add(Dropout(0.2))
#model.add(Bidirectional(LSTM(240, input_shape=(window_length, number_of_features), return_sequences=True)))
#model.add(Bidirectional(LSTM(240, input_shape=(window_length, number_of_features), return_sequences=True)))
#model.add(Bidirectional(LSTM(240, input_shape=(window_length, number_of_features), return_sequences=False)))
model.add(Dense(69))
model.add(Dense(number_of_features))
model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

# Training
model.fit(train, label, batch_size=100, epochs=5000)

# Prediction
to_predict = arr
scaled_to_predict = scaler.transform(to_predict)
scaled_prediction_output_1 = model.predict(np.array([scaled_to_predict]))
print(scaler.inverse_transform(scaled_prediction_output_1).astype(int)[0])
