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
        dqn_agent.model.load_weights('recent_weights.hdf5')
        dqn_agent.model_target.load_weights('recent_weights.hdf5')
        
        for i in tqdm.tqdm(range(500)):
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

        print(scores)
        plt.plot(scores)
        plt.show()
        
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
        for i in range(10):
            action = dqn_agent.get_action(state)
            new_state, score, done = dqn_agent.step(state, action)       
            print(new_state, score)
            state = new_state
        

if __name__ == '__main__':
#    (n,l,a) tuples to represent no. sequences, length and amino acids 
    n = 5
    l = 5
    a = 4
    for i in range(10):
        sequences = seq_generator.generate(n,l,0.1,0.4)
        print("Training on sequence " +str(i))
        print(sequences)
        main(sequences, True)