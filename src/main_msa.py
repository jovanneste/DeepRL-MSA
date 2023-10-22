from agent import Agent
import seq_generator
import environment
import numpy as np
import matplotlib.pyplot as plt
import time

dqn_agent = Agent()
sequences = seq_generator.generate(5,5,0.2,0.4)
scores = []
average = []

print(sequences)

for i in range(500):
    timesteps = dqn_agent.total_timesteps
    timee = time.time()
    score = environment.play_episode(dqn_agent, sequences)
    scores.append(score)
    
    if i%100==0:  
        print('\nEpisode: ' + str(i))
        print('Steps: ' + str(dqn_agent.total_timesteps - timesteps))
        print('Duration: ' + str(time.time() - timee))
        print('Score: ' + str(score))
        print('Epsilon: ' + str(dqn_agent.epsilon))

print(scores)
plt.plot(scores)
plt.show() 

