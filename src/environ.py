import numpy as np
from agent import Agent
from generator import *

def initialise_new_game(agent, state):
#    process first state maybe
    agent.memory.clear()
    agent.memory.store_experience(state, (0,0), 0)
    


def take_step(agent):
    agent.total_timesteps += 1
    if agent.total_timesteps % 1000 == 0:
        agent.model.save_weights('recent_weights.hdf5')
        print("Weights saved")
    
    print("Action: " + str(agent.memory.actions[-1]))
        
    new_state, reward, done = agent.step(agent.memory.actions[-1])
    print(new_state)
    print(reward)
    next_action = agent.get_action(new_state)

    agent.memory.store_experience(next_state, next_action, reward)

    if done:
        return (score+reward), True

#        update model here

    return (score+reward), False

    
    
def play_episode(agent, sequences):
    initialise_new_game(agent, sequences)
    print("Environement ready")
    done = False
    score = 0
    while True:
        score, done = take_step(agent)
        print(score)
        if done:
            print("DONE")
            break
    return score



a = Agent()
print(a.total_timesteps)
s = generate_sequence(5, 5, 0.2, 0.4)


print(s)

play_episode(a, s)