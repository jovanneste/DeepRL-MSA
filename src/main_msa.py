from agents.single_agent import Agent
import seq_generator
import environment
import numpy as np
import matplotlib.pyplot as plt
import time
import os 
import tqdm


def main(sequences, n, l, training):
    dqn_agent = Agent(n, l)
    scores, average_returns = [], []
    
    
    marl = False
    #for MARL solution 
    #-----------------------------------------------------------------
    if marl:
        print("LOADING MARL SOLUTION.....")
        white_agent = 
        black_agent = 
        for in tqdm.tqdm(range(10)):
            timesteps = white_agent.total_timesteps
            timee = time.time()
            ep_return = environment.play_marl_episode(white_agent, black_agent, sequences)
            scores.append(ep_return)
            average_returns.append(ep_return/white_agent.memory_threshold)

            if i%1==0:  
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
    
    
    
    print("LOADING SARL SOLUTION.....")
    if training:
        try:
            dqn_agent.model.load_weights('agents/recent_weights.hdf5')
            dqn_agent.model_target.load_weights('agents/recent_weights.hdf5')
            print("Previous weights loaded...")
        except:
            print("No model weights loaded...")
        
        for i in tqdm.tqdm(range(10)):
            timesteps = dqn_agent.total_timesteps
            timee = time.time()
            ep_return = environment.play_single_episode(dqn_agent, sequences)
            scores.append(ep_return)
            average_returns.append(ep_return/dqn_agent.memory_threshold)

            if i%1==0:  
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
        for i in range(100):
            action = dqn_agent.get_action(state)
            new_state, score, done = dqn_agent.step(state, action)       

            state = new_state
            if i %10==0:
                print(state)
            
        return 0
        

if __name__ == '__main__':
#    (n,l,a) tuples to represent no. sequences, length and amino acids 
    n = 5
    l = 5
    a = 4
    scores = []
    for i in range(1):
        sequences = seq_generator.generate(n,l,a,0.2,0.2)
        print("Training on sequence " +str(i))
        print(sequences)
        scores.append(main(sequences, n, l, True))
    print(scores)
