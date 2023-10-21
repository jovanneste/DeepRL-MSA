from agent import Agent
import seq_generator
import environment
import matplotlib.pyplot as plt
import time

dqn_agent = Agent()
sequences = seq_generator.generate(5,5,0.2,0.4)
scores = []
average = []

for i in range(1):
    timesteps = dqn_agent.total_timesteps
    timee = time.time()
    score = environment.play_episode(dqn_agent, sequences)
    scores.append(score)
    
    print('\nEpisode: ' + str(i))
    print('Steps: ' + str(dqn_agent.total_timesteps - timesteps))
    print('Duration: ' + str(time.time() - timee))
    print('Score: ' + str(score))
    print('Epsilon: ' + str(dqn_agent.epsilon))
    
    if i%100==0 and i!=0:
        average.append(sum(scores)/len(scores))
        plt.plot(np.arange(0,i+1,100),average)
        plt.show()