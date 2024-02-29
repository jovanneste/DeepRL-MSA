import numpy as np

def initialise_new_game(agents, state):
    for agent in agents:
        agent.memory.clear()
        agent.memory.store_experience(state, (0,0), 0)
        
def take_step_sarl(agent):
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
    
def take_step_marl(white_agent, black_agent):
    white_agent.total_timesteps += 1
    if white_agent.total_timesteps % 100 == 0:
        white_agent.model.save_weights('agents/white_recent_weights.hdf5')
        black_agent.model.save_weights('agents/black_recent_weights.hdf5')
        
    
    if white_agent.total_timesteps % 2 == 0:
        print("WHITE MOVE")
        next_state, reward, done = white_agent.step(white_agent.memory.states[-1], white_agent.memory.actions[-1])
        next_action = white_agent.get_action(next_state)
        white_agent.memory.store_experience(next_state, next_action, reward)
        #black_agent.memory.store_experience(next_state, next_action, reward)
    else:
        print("BLACK MOVE")
        next_state, reward, done = black_agent.step(black_agent.memory.states[-1], black_agent.memory.actions[-1])
        next_action = black_agent.get_action(next_state)
        black_agent.memory.store_experience(next_state, next_action, reward)
        #white_agent.memory.store_experience(next_state, next_action, reward)
        
    if done:
        return (score+reward), True

    if len(white_agent.memory) > white_agent.memory_threshold:
        white_agent.learn()
        black_agent.learn()
        return (score+reward), True

    return (score+reward), False
        
    
def play_single_episode(agent, sequences):
    global score
    score = 0
    initialise_new_game([agent], sequences)
    done = False
    while True:
        score, done = take_step_sarl(agent)
        if done:
            break
    return score

def play_marl_episode(white_agent, black_agent, sequences):
    global score
    score = 0
    initialise_new_game([white_agent, black_agent], sequences)
    done = False
    while True:
        score, done = take_step_marl(white_agent, black_agent)
        if done:
            break
    return score
    
def play_voting_episode(agents, sequences):
    global score
    score = 0
    initialise_new_game(agents, sequences)
    done = False
    while True:
        score, done = take_step_vote(agents)
        if done:
            break
    return score