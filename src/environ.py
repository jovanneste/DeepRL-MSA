import numpy as np
from agent import Agent
from generator import *

def initialise_new_game(agent, state):
    agent.memory.clear()
    agent.memory.store_experience(state, (0,0), 0)
    

def take_step(agent):
    agent.total_timesteps += 1
    if agent.total_timesteps % 100 == 0:
        agent.model.save_weights('recent_weights.hdf5')
        print("Weights saved")

    next_state, reward, done = agent.step(agent.memory.states[-1], agent.memory.actions[-1])
    next_action = agent.get_action(next_state)
    agent.memory.store_experience(next_state, next_action, reward)
    
    print(reward)

    if done:
        return (score+reward), True

    if len(agent.memory) > agent.memory_threshold:
        print("Update model")
        return (score+reward), True

    return (score+reward), False
    
    
def play_episode(agent, sequences):
    global score
    initialise_new_game(agent, sequences)
    done = False
    while True:
        score, done = take_step(agent)
        if done:
            print("Episode done, score: "+str(score))
            break
    return score



a = Agent()
s = generate_sequence(5, 5, 0.2, 0.4)
score = 0
print(s)
play_episode(a, s)