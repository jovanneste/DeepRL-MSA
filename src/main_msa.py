from agent import Agent
import seq_generator
import environment
import numpy as np
import matplotlib.pyplot as plt
import time
import os 
import tqdm


def main(sequences, training):
    dqn_agent = Agent()
    scores, average_returns = [], []
    if training:
        try:
            dqn_agent.model.load_weights('recent_weights.hdf5')
            dqn_agent.model_target.load_weights('recent_weights.hdf5')
        except:
            print("No model weights loaded...")
        
        for i in tqdm.tqdm(range(1100)):
            timesteps = dqn_agent.total_timesteps
            timee = time.time()
            ep_return = environment.play_episode(dqn_agent, sequences)
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
        plt.show()
        return scores
        
    else: 
        print("Loading previous model weights...")
        try:
            dqn_agent.model.load_weights('recent_weights.hdf5')
            dqn_agent.model_target.load_weights('recent_weights.hdf5')
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
    n = 10
    l = 10
    a = 6
    scores = []
    for i in range(1):
        sequences = seq_generator.generate(n,l,a,0.2,0.4)
        print("Training on sequence " +str(i))
        print(sequences)
        scores.append(main(sequences, True))
    print(scores)