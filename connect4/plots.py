import matplotlib.pyplot as plt

# x = [1, 2, 3, 5, 6]
# y1 = [0.005595588684082031, 0.03026576042175293, 0.16427366733551024, 10.519350051879883, 67.19741632938386]
# y2 = [0.005113744735717773, 0.029826664924621583, 0.17411549091339112, 8.17406461238861, 68.64441194534302]
# y3 = [0.004529142379760742, 0.025357627868652345, 0.08530137538909913, 1.3406848907470703, 6.127353715896606]

# plt.plot(x, y1, label="Minimax")
# plt.plot(x, y2, label="Negamax")
# plt.plot(x, y3, label="Negamax + AB pruning")
# plt.legend()
# plt.xlabel("max_depth values")
# plt.ylabel("Average time taken to play a game in seconds (over 10 games)")
# plt.savefig("minimax")
# plt.show()




# for possible_expansion_times = [1, 5, 100, 200]
# and possible_max_depths = [1, 2, 3, 4]
#data = [[(35.0, 65.0, 99.0, 1.0), (0.0, 100.0, 99.0, 1.0), (2.0, 98.0, 100.0, 0.0), (0.0, 100.0, 100.0, 0.0)], 
#        [(28.000000000000004, 72.0, 92.0, 8.0), (9.0, 91.0, 95.0, 5.0), (2.0, 98.0, 100.0, 0.0), (0.0, 100.0, 100.0, 0.0)], 
#        [(83.0, 17.0, 42.0, 57.99999999999999), (57.99999999999999, 42.0, 49.0, 51.0), (43.0, 56.99999999999999, 77.0, 23.0), (6.0, 94.0, 86.0, 14.000000000000002)], 
#        [(94.0, 6.0, 31.0, 69.0), (73.0, 27.0, 24.0, 76.0), (48.0, 52.0, 67.0, 33.0), (21.0, 79.0, 71.0, 28.999999999999996)]]


expansion_times = [1, 10, 100, 200, 500]
max_depths = [1, 2, 3, 4, 5, 6]

# (mcts_player1_winrate, negamax_player2_winrate, negamax_player1_winrate, mcts_player2_winrate)
data = [[(31.0, 69.0, 99.0, 1.0), (1.0, 99.0, 100.0, 0.0), (0.0, 100.0, 100.0, 0.0), (1.0, 99.0, 100.0, 0.0), (0.0, 100.0, 100.0, 0.0), (0.0, 100.0, 100.0, 0.0)], [(47.0, 53.0, 82.0, 18.0), (12.0, 88.0, 87.0, 13.0), (5.0, 95.0, 98.0, 2.0), (5.0, 95.0, 99.0, 1.0), (5.0, 95.0, 98.0, 2.0), (3.0, 97.0, 100.0, 0.0)], [(86.0, 14.000000000000002, 40.0, 60.0), (56.99999999999999, 43.0, 47.0, 53.0), (40.0, 60.0, 77.0, 23.0), (10.0, 90.0, 83.0, 17.0), (12.0, 88.0, 85.0, 15.0), (3.0, 97.0, 93.0, 7.000000000000001)], [(93.0, 7.000000000000001, 30.0, 70.0), (64.0, 36.0, 32.0, 68.0), (44.0, 56.00000000000001, 70.0, 30.0), (14.000000000000002, 86.0, 77.0, 23.0), (13.0, 87.0, 86.0, 14.000000000000002), (21.0, 79.0, 77.0, 23.0)], [(100.0, 0.0, 17.0, 83.0), (87.0, 13.0, 15.0, 85.0), (62.0, 38.0, 55.00000000000001, 45.0), (28.000000000000004, 72.0, 63.0, 37.0), (30.0, 70.0, 68.0, 32.0), (25.0, 75.0, 75.0, 25.0)]]

# winrates for expansion_time = 1, max_depth = [1, 2, 3, 4, 5, 6]
# [(31.0, 69.0, 99.0, 1.0), (1.0, 99.0, 100.0, 0.0), (0.0, 100.0, 100.0, 0.0), (1.0, 99.0, 100.0, 0.0), (0.0, 100.0, 100.0, 0.0), (0.0, 100.0, 100.0, 0.0)]

# winrates for expansion_time = 10, max_depth = [1, 2, 3, 4, 5, 6]
# [(47.0, 53.0, 82.0, 18.0), (12.0, 88.0, 87.0, 13.0), (5.0, 95.0, 98.0, 2.0), (5.0, 95.0, 99.0, 1.0), (5.0, 95.0, 98.0, 2.0), (3.0, 97.0, 100.0, 0.0)]

# winrates for expansion_time = 100, max_depth = [1, 2, 3, 4, 5, 6]
# [(86.0, 14.000000000000002, 40.0, 60.0), (56.99999999999999, 43.0, 47.0, 53.0), (40.0, 60.0, 77.0, 23.0), (10.0, 90.0, 83.0, 17.0), (12.0, 88.0, 85.0, 15.0), (3.0, 97.0, 93.0, 7.000000000000001)]

# winrates for expansion_time = 200, max_depth = [1, 2, 3, 4, 5, 6]
# [(93.0, 7.000000000000001, 30.0, 70.0), (64.0, 36.0, 32.0, 68.0), (44.0, 56.00000000000001, 70.0, 30.0), (14.000000000000002, 86.0, 77.0, 23.0), (13.0, 87.0, 86.0, 14.000000000000002), (21.0, 79.0, 77.0, 23.0)]

# winrates for expansion_time = 500, max_depth = [1, 2, 3, 4, 5, 6]
# [(100.0, 0.0, 17.0, 83.0), (87.0, 13.0, 15.0, 85.0), (62.0, 38.0, 55.00000000000001, 45.0), (28.000000000000004, 72.0, 63.0, 37.0), (30.0, 70.0, 68.0, 32.0), (25.0, 75.0, 75.0, 25.0)]

mcts_winrates_player_1 = [[winrates[0] for winrates in row] for row in data]
mcts_winrates_player_2 = [[winrates[3] for winrates in row] for row in data]

heatmap = plt.imshow(mcts_winrates_player_1, cmap='viridis', origin="lower", vmin=0, vmax=100)
plt.colorbar(heatmap, label="MCTS winrate")
plt.yticks(range(len(expansion_times)), expansion_times)
plt.xticks(range(len(max_depths)), max_depths)

for i in range(len(mcts_winrates_player_1)):
    for j in range(len(mcts_winrates_player_1[0])):
        plt.text(j, i, f"{mcts_winrates_player_1[i][j]:.1f}", ha='center', va='center', color='white')

plt.ylabel("MCTS expansion times")
plt.xlabel("Negamax + AB pruning max_depth")
plt.title("MCTS winrate when MCTS is player 1, negamax is player 2", pad=30)
plt.savefig("MCTS_winrates_player1")
plt.show()


heatmap = plt.imshow(mcts_winrates_player_2, cmap='viridis', origin="lower", vmin=0, vmax=100)
plt.colorbar(heatmap, label="MCTS winrate")
plt.yticks(range(len(expansion_times)), expansion_times)
plt.xticks(range(len(max_depths)), max_depths)

for i in range(len(mcts_winrates_player_2)):
    for j in range(len(mcts_winrates_player_2[0])):
        plt.text(j, i, f"{mcts_winrates_player_2[i][j]:.1f}", ha='center', va='center', color='white')

plt.ylabel("MCTS expansion times")
plt.xlabel("Negamax + AB pruning max_depth")
plt.title("MCTS winrate when MCTS is player 2, negamax is player 1", pad=30)
plt.savefig("MCTS_winrates_player2")
plt.show()