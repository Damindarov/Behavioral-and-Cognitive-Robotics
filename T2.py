import gym
import numpy as np
    
def update(ninputs, observation, W1, b1, W2, b2):
    observation.resize(ninputs,1) 

    Z1 = np.dot(W1, observation) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = np.tanh(Z2)

    if (isinstance(env.action_space, gym.spaces.box.Box)):
        action = A2
    else:
        action = np.argmax(A2)
    return action

def evaluate(env, nepisodes, W1, b1, W2, b2):
    done = False
    fitness = 0
    fitnesses = []
    for _ in range(nepisodes):
        observation = env.reset()
        while not done:
            action = update(ninputs, observation, W1, b1, W2, b2)
            env.render()
            observation, reward, done, info = env.step(action)
            fitness += reward
        fitnesses.append(fitness)
    return np.mean(fitnesses)

env = gym.make("CartPole-v0")

pvariance = 0.1     
ppvariance = 0.02   
nhiddens = 5
nepisodes = 10

ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n
     
W1 = np.random.randn(nhiddens,ninputs) * pvariance      
W2 = np.random.randn(noutputs, nhiddens) * pvariance   
b1 = np.zeros(shape=(nhiddens, 1))                     
b2 = np.zeros(shape=(noutputs, 1))                     

observation = env.reset()
mean = evaluate(env, nepisodes, W1, b1, W2, b2)
print(f"Average fitness over {nepisodes} episodes: {mean}") 