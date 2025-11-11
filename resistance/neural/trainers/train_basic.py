import pandas as pd
import numpy as np 
import pickle
import sys,os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# This assumes we've copied the log file ("LoggerBot.log") into our current working folder...
df0 = pd.read_csv("neural/trainers/LoggerBot.log",names=["PlayerID","PlayerName","Tries", "RatioMissionsBeenOnThatFail","VotedMajorityCount", "LossesLeft", "WinsLeft", "OnFirstFailedMission", "GameTurn", "MissionsBeenOn","FailedMissionsBeenOn","UpvotedMissionsTheyAreOn", "UpvotedMissionsTheyAreOff", "AvgSuspicionOfVotedUpMissions", "AvgSuspicionOfVotedDownMissions", "Spy"])

df = df0[["RatioMissionsBeenOnThatFail","VotedMajorityCount", "LossesLeft", "WinsLeft", "OnFirstFailedMission", "GameTurn", "MissionsBeenOn","FailedMissionsBeenOn","UpvotedMissionsTheyAreOn", "UpvotedMissionsTheyAreOff", "AvgSuspicionOfVotedUpMissions", "AvgSuspicionOfVotedDownMissions","Spy"]]

resistance_count = (df["Spy"] == 0).sum()
spy_count = (df["Spy"] == 1).sum()

x_train = df.values[:,0:12].astype(np.float32) # This filters out only the columns we want to use as input vector for our NN.
# with open("neural/trainers/scaler.pkl", "rb") as file:
#   scaler = pickle.load(file)
# x_train = scaler.transform(x_train)

# scaler = MinMaxScaler()
# x_train = scaler.fit_transform(x_train)
# with open("single_scaler.pkl", "wb") as f:
#   pickle.dump(scaler, f)

y_train=df.values[:,12].astype(np.int32) # This is our target column.
num_inputs=x_train.shape[1] # this works out how many columns there are in x, i.e. how many inputs our network needs.
num_outputs=2 # Two outputs needed - for "spy" or "not spy".


dataset_size=len(x_train)
train_set_size=int(dataset_size*0.7) # choose 70% of the data for training and 30% for validation
x_val,y_val=x_train[train_set_size:],y_train[train_set_size:]
x_train,y_train=x_train[:train_set_size],y_train[:train_set_size]


# Define Sequential model with 3 layers
# model = keras.Sequential([

#     layers.Dense(64, activation="relu", input_shape=(num_inputs,)),
#     layers.Dense(32, activation="relu"),
#     layers.Dropout(0.2),
#     layers.Dense(16, activation="relu"),
#     layers.Dense(num_outputs, activation="softmax")

# ], name="my_neural_network")

model = keras.Sequential([

    layers.Dense(10, activation="relu", input_shape=(num_inputs,)),
    layers.Dense(10, activation="relu"),
    layers.Dense(num_outputs, activation="softmax")

], name="my_neural_network")


# Do the usual business for keras training
# It's a classification problem , so we need cross entropy here.
model.compile(
    optimizer=keras.optimizers.Adam(0.001),  # Optimizer
    # Loss function to minimize
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

# Do the usual business for keras training
history = model.fit(
    x_train,
    y_train,
    batch_size=128,
    epochs=40,
    shuffle=False,
    validation_data=(x_val, y_val), verbose=1,
)



# Plot our training curves. This is always important to see if we've started to overfit or whether 
# we could benefit from more training cycles....
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"],label="Validation Set Accuracy")
plt.legend()
plt.grid()
plt.show()
# The following graph should show about 77% accuracy.



model.save('neuralbot_classifier.keras')