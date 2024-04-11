"""
Author: Joachim Vanneste
Date: 10 Apr 2024
Description: Main method to launch MSA solver
"""

from agents.single_agent import Agent
from agents.white_agent import WhiteAgent
from agents.black_agent import BlackAgent
import seq_generator
import environment
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import time
import os 
import tqdm
import timeit
import argparse


def main(sequences, n, l, training, marl, voting):
    dqn_agent = Agent(n, l)
    scores, average_returns = [], []
    
    #for MARL solution 
    #-----------------------------------------------------------------
    if marl:
        print("LOADING MARL SOLUTION.....")
        white_agent = WhiteAgent(n,l)
        black_agent = BlackAgent(n,l)
        for i in tqdm.tqdm(range(10)):
            timesteps = white_agent.total_timesteps
            timee = time.time()
            ep_return = environment.play_marl_episode(white_agent, black_agent, sequences)
            scores.append(ep_return)
            average_returns.append(ep_return/white_agent.memory_threshold)

            if i%100==0:  
                print('\nEpisode: ' + str(i))
                print('Steps: ' + str(white_agent.total_timesteps - timesteps))
                print('Duration: ' + str(time.time() - timee))
                print('Score: ' + str(ep_return))
                print('Average return of action: ' + str(ep_return/white_agent.memory_threshold))
                print('Epsilon: ' + str(white_agent.epsilon))
                
        plt.plot(scores)
        plt.xlabel('Episode')
        plt.ylabel('Average reward')
        plt.show()
        return scores
    #-----------------------------------------------------------------

    #for voting solution 
    #-----------------------------------------------------------------
    if voting:
        print("LOADiNG VOTING MODEL....")
        agents = [Agent(n,l), Agent(n,l,1e-5), Agent(n,l,1e-5), Agent(n,l,1e-2), Agent(n,l)]
        for i in tqdm.tqdm(range(1000)):
            timesteps = agents[0].total_timesteps
            timee = time.time()
            ep_return = environment.play_voting_episode(agents, sequences)
            scores.append(ep_return)
            average_returns.append(ep_return/agents[0].memory_threshold)

            if i%100==0:  
                print('\nEpisode: ' + str(i))
                print('Steps: ' + str(agents[0].total_timesteps - timesteps))
                print('Duration: ' + str(time.time() - timee))
                print('Score: ' + str(ep_return))
                print('Average return of action: ' + str(ep_return/agents[0].memory_threshold))
                print('Epsilon: ' + str(agents[0].epsilon))
                
        plt.plot(scores)
        plt.xlabel('Episode')
        plt.ylabel('Average reward')
        plt.show()
        return scores
    #-----------------------------------------------------------------
    
    print("LOADING SARL SOLUTION.....")
    if training:
        try:
            dqn_agent.model.load_weights('agents/recent_weights.hdf5')
            dqn_agent.model_target.load_weights('agents/recent_weights.hdf5')
            print("Previous weights loaded...")
        except:
            print("No model weights loaded...")
        
        for i in tqdm.tqdm(range(1000)):
            timesteps = dqn_agent.total_timesteps
            timee = time.time()
            ep_return = environment.play_single_episode(dqn_agent, sequences)
            scores.append(ep_return)
            average_returns.append(ep_return/dqn_agent.memory_threshold)

            if i%100==0:  
                print('\nEpisode: ' + str(i))
                print('Steps: ' + str(dqn_agent.total_timesteps - timesteps))
                print('Duration: ' + str(time.time() - timee))
                print('Score: ' + str(ep_return))
                print('Average return of action: ' + str(ep_return/dqn_agent.memory_threshold))
                print('Epsilon: ' + str(dqn_agent.epsilon))

        plt.plot(scores)
        plt.xlabel('Episode')
        plt.ylabel('Average reward')
        plt.show()
        
        for i in [10, 15, 30, 50]:
            window_size = i
            smoothed_scores = np.convolve(scores, np.ones(window_size)/window_size, mode='valid')

            episode_numbers = np.arange(window_size, len(scores) + 1)

            plt.plot(episode_numbers, smoothed_scores)
            plt.xlabel('Episode')
            plt.ylabel('Smoothed Average Reward')
            plt.show()
        return scores
        
    else: 
        print("Loading previous model weights...")
        try:
            dqn_agent.model.load_weights('agents/recent_weights.hdf5')
            dqn_agent.model_target.load_weights('agents/recent_weights.hdf5')
            dqn_agent.epsilon = 0.0
        except:
            print("No model weights found... exiting")
            return
        
        state = sequences
        print("Starting alignment - ")
        print(state)
        for i in range(2000):
            action = dqn_agent.get_action(state)
            new_state, score, done = dqn_agent.step(state, action)       

            state = new_state
            if i%500==0:
                print(state)
            
        return 0
        


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Run SARL or MARL solution with the --multi flag.')
    parser.add_argument('--multi', dest='multi', action='store_true', help='Run multi-agent (MARL) solution.')
    parser.add_argument('--vote', dest='vote', action='store_true', help='Use ensemble voting model.')
    parser.add_argument('--train', dest='train', action='store_true', help='Train RL model.')
    args = parser.parse_args()

    n = 5
    l = 5
    a = 4
    
    encoder = OneHotEncoder()
    sequences = seq_generator.generate(n,l,a,0.2,0.2)
    sequences = encoder.fit_transform(sequences.reshape(-1, 1)).toarray()
    execution_time = timeit.timeit(lambda: main(sequences, n, l, args.train, args.multi, args.vote), number=1)
    print("Execution time:", execution_time, "seconds")

