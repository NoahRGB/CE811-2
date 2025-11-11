import pandas as pd
import pickle
import numpy as np 
import sys,os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# This assumes we've copied the log file ("LoggerBot.log") into our current working folder...
df0 = pd.read_csv("neural/trainers/LoggerBot.log",names=["PlayerID","PlayerName","Tries", "VotingBias", "RatioMissionsBeenOnThatFail","VotedMajorityCount", "LastChance", "LossesLeft", "WinsLeft", "OnFirstFailedMission", "GameTurn", "MissionsBeenOn","FailedMissionsBeenOn","UpvotedMissionsTheyAreOn", "UpvotedMissionsTheyAreOff", "AvgSuspicionOfVotedUpMissions", "AvgSuspicionOfVotedDownMissions", "VotedUp0","VotedUp1","VotedUp2","VotedUp3","VotedUp4","VotedUp5","VotedDown0","VotedDown1","VotedDown2","VotedDown3","VotedDown4","VotedDown5","Spy"])
df0 = df0[["PlayerName", "RatioMissionsBeenOnThatFail","VotedMajorityCount", "LossesLeft", "WinsLeft", "OnFirstFailedMission", "GameTurn", "MissionsBeenOn","FailedMissionsBeenOn","UpvotedMissionsTheyAreOn", "UpvotedMissionsTheyAreOff", "AvgSuspicionOfVotedUpMissions", "AvgSuspicionOfVotedDownMissions","Spy"]]

print(df0)

# bot_names = ["Trickerton", "Logicalton", "Simpleton", "NeuralBot", "LoggerBot", "Bounder"]
bot_names = ["Trickerton"]
bot_dfs = {}
for bot in bot_names:
  bot_dfs[bot] = df0.query(f"PlayerName=='{bot}'")

for bot, df in bot_dfs.items():
    x_train = df.values[:,1:13].astype(np.float32) # get rid of turn / try / playerid / playername
    
    # scaler = MinMaxScaler()
    # x_train = scaler.fit_transform(x_train)

    # with open(f"{bot}_scaler.pkl", "wb") as f:
    #     pickle.dump(scaler, f)

    y_train=df.values[:,13].astype(np.int32) # target column (spy or not)
    num_inputs = x_train.shape[1]
    num_outputs = 2 # spy or not?

    dataset_size = len(x_train)
    train_set_size = int(dataset_size * 0.7)
    x_val, y_val = x_train[train_set_size:], y_train[train_set_size:]
    x_train, y_train = x_train[:train_set_size], y_train[:train_set_size]

    model = keras.Sequential(name=f"neural_network_{bot}")
    l = [
       layers.Dense(10, activation="relu", input_shape=(num_inputs,)),
       layers.Dense(10, activation="relu"),
       layers.Dense(num_outputs, activation="softmax")
    ]

    for layer in l:
       model.add(layer)

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )

    history = model.fit(
        x_train,
        y_train,
        batch_size=50,
        epochs=50,
        shuffle=True,
        validation_data=(x_val, y_val), verbose=1
    )

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"],label="Validation Set Accuracy")
    plt.legend()
    plt.grid()
    plt.show()

    # accuracy_by_turn=[]
    # maximum_turn=df['Turn'].max()
    # accuracy_metric=tf.keras.metrics.Accuracy()
    # print("maximum_turn",maximum_turn)
    # for turn in range(1,maximum_turn+1):
    #     df_restricted=df.query('Turn>='+str(turn)) # Pull out just those rows of the training data corresponding to later turns in the game
        
    #     x=df_restricted.values[:,4:18].astype(np.float32)
    #     y=df_restricted.values[:,18].astype(np.int32)
    #     y_guess=model(x)
    #     y_guess=tf.argmax(y_guess,axis=1)
    #     #accuracy=tf.reduce_mean(tf.cast(tf.equal(y,y_guess),tf.float32)) # This formula owuld also give us the accuracy but this is hand-evaluated.
    #     accuracy=accuracy_metric(y_guess,y) # This function calculates accuracy using an in-built keras function.
    #     accuracy_by_turn.append(accuracy.numpy()) # record the results so we can plot them.
    # print(tf.range(maximum_turn),accuracy_by_turn)
    # plt.plot(tf.range(1,1+len(accuracy_by_turn)),accuracy_by_turn)
    # plt.title('Accuracy at identifying whether "Bounder" is a spy as the game progresses')
    # plt.xlabel('Turn')
    # plt.ylabel('Accuracy')
    # plt.grid()
    # plt.show()

    # model.save(f"{bot}_classifier.keras")