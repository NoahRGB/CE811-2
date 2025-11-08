from player import Bot
from game import State
import random

# run this with python competition.py 10000 bots/intermediates.py bots/loggerbot.py  
# Then check logs/loggerbot.log   Delete that file before running though

class LoggerBot(Bot):

    # Loggerbot makes very simple playing strategy.
    # We're not really trying to win here, but just to observer the other players
    # without disturbing them too much....
    def select(self, players, count):
        return [self] + random.sample(self.others(), count - 1)

    def vote(self, team):
        return True

    def sabotage(self):
        return True

    def mission_total_suspect_count(self, team):
        return 0 # TODO complete this function
        
    def onVoteComplete(self, votes):
        """Callback once the whole team has voted.
        @param votes        Boolean votes for each player (ordered).
        """
        total_suspect_count = self.mission_total_suspect_count(self.game.team)
        total_suspect_count = min(total_suspect_count, 5)

        for i in range(0, len(self.game.players)):
            if votes[i] == True:
                self.num_missions_voted_up_with_total_suspect_count[self.game.players[i]][total_suspect_count] += 1
            else:
                self.num_missions_voted_down_with_total_suspect_count[self.game.players[i]][total_suspect_count] += 1

        for p in self.game.players:
            self.training_feature_vectors[p].append([self.game.turn, self.game.tries, p.index, p.name, self.missions_been_on[p], self.failed_missions_been_on[p]]+self.num_missions_voted_up_with_total_suspect_count[p]+self.num_missions_voted_down_with_total_suspect_count[p])

    def onGameRevealed(self, players, spies):
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        @param players  List of all players in the game including you.
        @param spies    List of players that are spies, or an empty list.
        """
        self.failed_missions_been_on = {}
        self.missions_been_on = {}
        self.num_missions_voted_up_with_total_suspect_count = {}
        self.num_missions_voted_down_with_total_suspect_count = {}
        for player in players:
            self.failed_missions_been_on[player] = 0
            self.missions_been_on[player] = 0
            self.num_missions_voted_up_with_total_suspect_count[player] = [0, 0, 0, 0, 0, 0]
            self.num_missions_voted_down_with_total_suspect_count[player] = [0, 0, 0, 0, 0, 0]

        self.training_feature_vectors={}
        for p in players:
            self.training_feature_vectors[p]=[] # This is going to be a list of length-14 feature vectors for each player.

    def onMissionComplete(self, num_sabotages):
        """Callback once the players have been chosen.
        @param num_sabotages    Integer how many times the mission was sabotaged.
        """
        for player in self.game.team:
            if num_sabotages > 0:
                self.failed_missions_been_on[player] += 1
            self.missions_been_on[player] += 1

    def onGameComplete(self, win, spies):
        for player_number in range(len(self.game.players)):
            player=self.game.players[player_number]
            spy=player in spies # This will be a boolean
            feature_vectors=self.training_feature_vectors[player]  # These are our input features
            for v in feature_vectors:
                v.append(1 if spy else 0)  # append a 1 or 0 onto the end of our feature vector (for the label, i.e. spy or not spy)
                self.log.debug(','.join(map(str, v)) ) # converts all of elements of v into a csv list, and writes the full csv list to the log file

    def mission_total_suspect_count(self, team):
        total = 0
        for player in team:
            total += self.failed_missions_been_on[player]
        return total