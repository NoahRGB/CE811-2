from player import Bot 
from game import State
import random
import pickle
import sys,os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from loggerbot_new2 import LoggerBot # this assumes our loggerbot was in a file called loggerbot.py

# model_filename='loggerbot_classifier2.keras'
# if os.path.exists(".."+os.sep+"trainers"+os.sep+model_filename):
#     model_filename=".."+os.sep+"trainers"+os.sep+model_filename
model = keras.models.load_model("neural/models/loggerbot_new_classifier8.keras")
trw=[w.numpy() for w in model.trainable_weights] # capture all of the weights and biases in the saved model.

with open("neural/trainers/single_scaler.pkl", "rb") as file:
  scaler = pickle.load(file)

class NeuralBot(LoggerBot):
    
    def calc_player_probabilities_of_being_spy(self):
        probabilities = {}
        vectors = []
        for p in self.game.players:
            avg_suspicion_of_voted_down_missions = 0
            total_missions_voted_up = sum(self.num_missions_voted_up_with_total_suspect_count[p])
            total_missions_voted_down = sum(self.num_missions_voted_down_with_total_suspect_count[p])

            for i in range(0, len(self.num_missions_voted_up_with_total_suspect_count[p])):
                avg_suspicion_of_voted_up_missions += self.num_missions_voted_up_with_total_suspect_count[p][i] * i
            
            for i in range(0, len(self.num_missions_voted_down_with_total_suspect_count[p])):
                avg_suspicion_of_voted_down_missions += self.num_missions_voted_down_with_total_suspect_count[p][i] * i

            if total_missions_voted_up > 0:                     
                avg_suspicion_of_voted_up_missions /= total_missions_voted_up

            if total_missions_voted_down > 0:                     
                avg_suspicion_of_voted_down_missions /= total_missions_voted_down

    
            ratio_missions_on_that_fail = self.failed_missions_been_on[p] / (self.missions_been_on[p] if self.missions_been_on[p] > 0 else 1)

            input_vector = [ratio_missions_on_that_fail,
                            self.voted_majority_count[p], 
                            3 - self.game.losses, 
                            3 - self.game.wins,
                            self.was_on_first_failed_mission[p], 
                            self.game.turn,
                            self.missions_been_on[p],
                            self.failed_missions_been_on[p],
                            self.upvoted_missions_they_are_on[p], 
                            self.upvoted_missions_they_are_off[p],
                            avg_suspicion_of_voted_up_missions, 
                            avg_suspicion_of_voted_down_missions]
            
            vectors.append(input_vector)

        vectors = np.stack(vectors, axis=0)
        vectors = scaler.transform(vectors)
        output_probabilities = model(vectors).numpy()  # run the neural network.  Its output layer was using softmax (and it was trained with cross-entropy 
        
        for i in range(len(self.game.players)):
            probabilities[self.game.players[i]] = output_probabilities[i, 1]  # this [i,1] pulls off the row for player i, and the second column (which corresponds to probability of being a spy; the first column is the probability of being not-spy)
        return probabilities  # This returns a dictionary of {player: spyProbability}
        
    def select(self, players, count):
        # here I'm replicating logic we used in the CountingBot exercise of lab1-challenge3.
        # But instead of using the count as an estimation of how spy-like a player is, instead
        # we'll use the neural network's estimation of the probability.
        spy_probs=self.calc_player_probabilities_of_being_spy()
        sorted_players_by_trustworthiness=[k for k, v in sorted(spy_probs.items(), key=lambda item: item[1])]
        if self in sorted_players_by_trustworthiness[:count]:
            result= sorted_players_by_trustworthiness[:count]
        else:
            result= [self] + sorted_players_by_trustworthiness[:count-1]
        return result


    def vote(self, team): 
        spy_probs=self.calc_player_probabilities_of_being_spy()
        sorted_players_by_trustworthiness=[k for k, v in sorted(spy_probs.items(), key=lambda item: item[1])]
        if not self.spy:
            for x in team:
                if x in sorted_players_by_trustworthiness[-2:]:
                    return False
            return True
        else:
            return True

    def sabotage(self):
        # the logic here is a bit boring and maybe could be improved.
        return True 

    ''' The 3 methods onVoteComplete, onGameRevealed, onMissionComplete
    will inherit their functionality from ancestor.  We want them to do exactly 
    the same as they did when we captured the training data, so that the variables 
    for input to the NN are set correctly.  Hence we don't override these methods
    '''
    
    # This function used to output log data to the log file. 
    # We don't need to log any data any more so let's override that function
    # and make it do nothing...
    def onGameComplete(self, win, spies):
        pass


