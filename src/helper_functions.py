from frame_process import stack_frames, process
from ppo import PPO
from actor_critic import Actor, Critic
import torch
import gym
import ale_py
import numpy as np
import matplotlib.pyplot as plt

def train(n_episodes,env,agent):
    scores = []
    average = 0
    
    for e in range(n_episodes):
        state = stack_frames(None, env.reset(), True)
        score = 0
        done = False
        while done != True:
            action, log_prob, value = agent.act(state)
            next_state, reward, done, info = env.step(action)
            score += reward   
            next_state = stack_frames(state, next_state, False)
            agent.remember(state, action, value, log_prob, reward, done, next_state)
            if done:
                break
            else:
                state = next_state

        scores.append(score)
        print(f"episode {e}: score:{score}")


    if len(scores) < 10:
        average = np.mean(scores)
    else:
        average = np.mean(scores[:-10])
        
    return scores,average

def play(game,train_eps):
    env = gym.make(game,render_mode='human')
    action_space = env.action_space.n
    agent = PPO(action_space, Actor, Critic)
    scores,avg = train(train_eps,env,agent)
    
    x=[x for x in range(1,len(scores)+1)]
    plt.plot(scores)
    plt.title(game)
    plt.xlabel('Frames')
    plt.ylabel('Score')
    plt.savefig(game+'_'+ str(train_eps) + '.png', bbox_inches="tight") 
    plt.show()

    return scores, avg
