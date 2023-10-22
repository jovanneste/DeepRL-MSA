from agent import Agent
import seq_generator
import environment
import numpy as np
import matplotlib.pyplot as plt
import time
import os 


def main(sequence, testing=False):
    dqn_agent = Agent()
    scores = []
    if testing:
        dqn_agent.model.load_weights('recent_weights.hdf5')
        dqn_agent.model_target.load_weights('recent_weights.hdf5')
        dqn_agent.epsilon = 0.0
    else: 
        os.remove('recent_weights.hdf5')
        for i in range(1000):
            timesteps = dqn_agent.total_timesteps
            timee = time.time()
            score = environment.play_episode(dqn_agent, sequences)
            scores.append(score)

            if i%200==0:  
                print('\nEpisode: ' + str(i))
                print('Steps: ' + str(dqn_agent.total_timesteps - timesteps))
                print('Duration: ' + str(time.time() - timee))
                print('Score: ' + str(score))
                print('Epsilon: ' + str(dqn_agent.epsilon))

        print(scores)
        plt.plot(scores)
        plt.show() 


if __name__ == '__main__':
    sequences = seq_generator.generate(5,5,0.2,0.4)
    main(sequences)