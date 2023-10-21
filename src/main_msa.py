from agent import Agent
import seq_generator
import environment

dqn_agent = Agent()
sequences = seq_generator.generate(5,5,0.2,0.4)
score = environment.play_episode(dqn_agent, sequences)