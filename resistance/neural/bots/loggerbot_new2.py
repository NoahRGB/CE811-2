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

        majority_vote = sum(vote == True for vote in votes) > sum(vote == False for vote in votes)

        for i in range(0, len(self.game.players)):
            if votes[i] == True:
                if majority_vote == True:
                    self.voted_majority_count[self.game.players[i]] += 1
                self.num_missions_voted_up_with_total_suspect_count[self.game.players[i]][total_suspect_count] += 1
                if self.game.players[i] in self.game.team:
                    self.missions_they_are_on_count[self.game.players[i]] += 1
                    self.upvoted_missions_they_are_on[self.game.players[i]] += 1
                else:
                    self.missions_they_are_off_count[self.game.players[i]] += 1
                    self.upvoted_missions_they_are_off[self.game.players[i]] += 1
            else:
                if majority_vote == False:
                    self.voted_majority_count[self.game.players[i]] += 1
                self.num_missions_voted_down_with_total_suspect_count[self.game.players[i]][total_suspect_count] += 1

        for p in self.game.players:
            avg_suspicion_of_voted_up_missions = 0
            avg_suspicion_of_voted_down_missions = 0
            total_missions_voted_up = sum(self.num_missions_voted_up_with_total_suspect_count[p])
            total_missions_voted_down = sum(self.num_missions_voted_down_with_total_suspect_count[p])

            for i in range(0, len(self.num_missions_voted_up_with_total_suspect_count[p])):
                avg_suspicion_of_voted_up_missions += self.num_missions_voted_up_with_total_suspect_count[p][i] * i
            
            for i in range(0, len(self.num_missions_voted_down_with_total_suspect_count[p])):
                avg_suspicion_of_voted_down_missions += self.num_missions_voted_down_with_total_suspect_count[p][i] * i

            if total_missions_voted_up > 0:                     
                avg_suspicion_of_voted_up_missions = float(avg_suspicion_of_voted_up_missions) / float(total_missions_voted_up)

            if total_missions_voted_down > 0:                     
                avg_suspicion_of_voted_down_missions = float(avg_suspicion_of_voted_down_missions) / float(total_missions_voted_down)

            on = float(self.upvoted_missions_they_are_on[p]) / float(self.missions_been_on[p]) if self.missions_been_on[p] > 0 else 0.5
            off = float(self.upvoted_missions_they_are_off[p]) / float(self.missions_been_off[p]) if self.missions_been_off[p] > 0 else 0.5
            voting_bias = on - off
            
            ratio_missions_on_that_fail = float(self.failed_missions_been_on[p]) / (float(self.missions_been_on[p]) if self.missions_been_on[p] > 0 else 1)

            self.training_feature_vectors[p].append([p.index, p.name, self.game.tries, voting_bias, ratio_missions_on_that_fail, self.voted_majority_count[p], self.game.wins==2, 3 - self.game.losses, 3 - self.game.wins, self.was_on_first_failed_mission[p], self.game.turn, self.missions_been_on[p], self.failed_missions_been_on[p], self.upvoted_missions_they_are_on[p], self.upvoted_missions_they_are_off[p], avg_suspicion_of_voted_up_missions, avg_suspicion_of_voted_down_missions]+self.num_missions_voted_up_with_total_suspect_count[p]+self.num_missions_voted_down_with_total_suspect_count[p])

    def onGameRevealed(self, players, spies):
        """This function will be called to list all the players, and if you're
        a spy, the spies too -- including others and yourself.
        @param players  List of all players in the game including you.
        @param spies    List of players that are spies, or an empty list.
        """
        self.failed_missions_been_on = {}
        self.missions_been_on = {}
        self.missions_been_off = {}
        self.num_missions_voted_up_with_total_suspect_count = {}
        self.num_missions_voted_down_with_total_suspect_count = {}
        self.upvoted_missions_they_are_on = {}
        self.upvoted_missions_they_are_off = {}
        self.was_on_first_failed_mission = {}
        self.voted_majority_count = {}
        self.missions_they_are_on_count = {}
        self.missions_they_are_off_count = {}
        for player in players:
            self.was_on_first_failed_mission[player] = 0
            self.upvoted_missions_they_are_on[player] = 0
            self.missions_they_are_on_count[player] = 0
            self.missions_they_are_off_count[player] = 0
            self.upvoted_missions_they_are_off[player] = 0
            self.voted_majority_count[player] = 0
            self.failed_missions_been_on[player] = 0
            self.missions_been_on[player] = 0
            self.missions_been_off[player] = 0
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
                if self.game.losses == 1:
                    self.was_on_first_failed_mission[player] = 1

            self.missions_been_on[player] += 1
          
        for player in self.game.players:
            if player not in self.game.team:
                self.missions_been_off[player] += 1

    def onGameComplete(self, win, spies):
        for player_number in range(len(self.game.players)):
            player = self.game.players[player_number]
            spy = player in spies # This will be a boolean
            feature_vectors = self.training_feature_vectors[player]  # These are our input features
            for v in feature_vectors:
                v.append(1 if spy else 0)  # append a 1 or 0 onto the end of our feature vector (for the label, i.e. spy or not spy)
                self.log.debug(','.join(map(str, v)) ) # converts all of elements of v into a csv list, and writes the full csv list to the log file

    def mission_total_suspect_count(self, team):
        total = 0
        for player in team:
            total += self.failed_missions_been_on[player]
        return total