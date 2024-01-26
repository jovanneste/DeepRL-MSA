import numpy as np

def initialise_new_game(agents, state):
    for agent in agents:
        agent.memory.clear()
        agent.memory.store_experience(state, (0,0), 0)
        
def take_step(agent):
    agent.total_timesteps += 1
    if agent.total_timesteps % 100 == 0:
        agent.model.save_weights('agents/recent_weights.hdf5')

    next_state, reward, done = agent.step(agent.memory.states[-1], agent.memory.actions[-1])
    next_action = agent.get_action(next_state)
    agent.memory.store_experience(next_state, next_action, reward)

    if done:
        return (score+reward), True

    if len(agent.memory) > agent.memory_threshold:
        agent.learn()
        return (score+reward), True

    return (score+reward), False
    
    
def play_single_episode(agent, sequences):
    global score
    score = 0
    initialise_new_game([agent], sequences)
    done = False
    while True:
        score, done = take_step(agent)
        if done:
            break
    return score

def play_marl_episode(white_agent, black_agent, sequences):
    global score
    score = 0
    initialise_new_game([white_agent, black_agent], sequences)
    done = False
    while True:
        score, done = take_step(agent)
        if done:
            break
    return score
    