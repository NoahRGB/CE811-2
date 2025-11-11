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
from loggerbot import LoggerBot

model = keras.models.load_model("loggerbot_classifier.keras")
trw=[w.numpy() for w in model.trainable_weights]

class NeuralBot(LoggerBot):
    
    def calc_player_probabilities_of_being_spy(self):
        probabilities = {}
        vectors = []
        for p in self.game.players:
            # calculating avg_suspicion_of_voted_down_missions, avg_suspicion_of_voted_up_missions
            avg_suspicion_of_voted_up_missions = 0
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
        output_probabilities = model(vectors).numpy()  
        
        for i in range(len(self.game.players)):
            probabilities[self.game.players[i]] = output_probabilities[i, 1] 
        return probabilities 
        
    def select(self, players, count):
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
        return True 

    def onGameComplete(self, win, spies):
        pass


