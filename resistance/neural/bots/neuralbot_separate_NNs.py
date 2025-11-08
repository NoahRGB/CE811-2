from player import Bot 
from game import State
import random
import sys,os
os.environ["KERAS_BACKEND"] = "tensorflow"
import tensorflow as tf
from tensorflow import keras
import numpy as np
from resistance.neural.bots.loggerbot_original import LoggerBot

bot_names = ["Trickerton", "Logicalton", "Simpleton", "NeuralBot", "LoggerBot", "Bounder"]
models = {}
for bot in bot_names:
    model_filename = f"./second_test/{bot}_classifier.keras"
    if os.path.exists("bots" + os.sep + model_filename):
        model_filename="bots" + os.sep + model_filename
        model = keras.models.load_model(model_filename)
        models[bot] = model

# trw=[w.numpy() for w in model.trainable_weights] # capture all of the weights and biases in the saved model.

class NeuralBot(LoggerBot):
    
    def calc_player_probabilities_of_being_spy(self):
        probabilities = {}
        for p in self.game.players:
            # This list comprising the input vector must build in **exactly** the same way as
            # we built data to train our neural network - otherwise the neural network
            # is not bieng used to approximate the same function it's been trained to model.
            # That's why this class inherits from the class LoggerBot- so we can ensure that logic is replicated exactly.
            input_vector = [self.game.turn, self.game.tries, p.index, p.name, self.missions_been_on[p],
                            self.failed_missions_been_on[p]] + self.num_missions_voted_up_with_total_suspect_count[p] + \
                           self.num_missions_voted_down_with_total_suspect_count[p]
            input_vector = input_vector[4:] # remove the first 4 cosmetic details, as we did when training the neural network

            input_vector = np.stack(input_vector, axis=0)
            if p.name in models:
                model = models[p.name]
            else:
                model = models["Bounder"]
            
            output = model(input_vector.reshape(1, -1)).numpy()[0][1]
            # print(output)
            probabilities[p] = output

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
                if x in sorted_players_by_trustworthiness[-4:]:
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


