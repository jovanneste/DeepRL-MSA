import numpy as np

def take_step(agent):
    agent.total_timesteps += 1
    if agent.total_timesteps % 1000 == 0:
        agent.model.save_weights('recent_weights.hdf5')
        print("Weights saved")
    
    
    
    
    
    
def play_episode(agent):
    done = False
    score = 0
    while True:
        score, done = take_step(agent)
        if done:
            break
    return score