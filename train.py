import sys
sys.path.insert(0,'src')

from helper_functions import train, play

print("The code was tested on Breakout-v0, KungFuMaster-v0, VideoPinball-v0, Centipede-v0")
game = input("Input the game to train the PPO algorithm on (caps-sensitive): ")
episodes = int(input("Input number of episodes: "))
scores,avg = play(game,episodes)
print(f"Last 5 scores were: {scores[-5:]}")
print(f"Average score of the last (max 10) scores: {avg}")

        
