import numpy as np


def initialise_new_game(agent, state):
#    process first state maybe
    agent.memory.store_experience(state, (0,0), 0)
    



def take_step(agent):
    agent.total_timesteps += 1
    if agent.total_timesteps % 1000 == 0:
        agent.model.save_weights('recent_weights.hdf5')
        print("Weights saved")
        
        new_state, reward, done = agent.step(agent.memory.actions[-1])
        next_action = agent.get_action(new_state)
        
        agent.memory.store_experience(next_state, next_action, reward)
    
        if done:
            return (score+reward), True
        
#        update model here
        
        return (score+reward), False
        
    
    
    
    
    
def play_episode(agent, sequences):
    initialise_new_game(agent, sequences)
    done = False
    score = 0
    while True:
        score, done = take_step(agent)
        if done:
            break
    return score