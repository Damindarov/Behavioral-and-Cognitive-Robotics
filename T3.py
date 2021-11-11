import numpy as np
import gym
import time

class Network:
    def __init__(self, env, nhiddens):
        ninputs = env.observation_space.shape[0]
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            noutputs = env.action_space.shape[0]
        else:
            noutputs = env.action_space.n

        self.ninputs = ninputs
        self.nhiddens = nhiddens
        self.noutputs = noutputs
        
    def update(self, ninputs, observation, params):
        observation.resize(ninputs,1) 
        W1, b1, W2, b2 = params[:]
        Z1 = np.dot(W1, observation) + b1
        A1 = np.tanh(Z1)

        Z2 = np.dot(W2, A1) + b2
        A2 = np.tanh(Z2)

        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)
        return action

    def evaluate(self, env, nepisodes, params, render=False):
        fitness = []
        for _ in range(nepisodes):
            done = False
            observation = env.reset()
            fit = 0
            while not done:
                action = self.update(self.ninputs, observation, params)
                observation, reward, done, _ = env.step(action)
                fit += reward
                if render:
                    env.render()
                    time.sleep(0.05)
            fitness.append(fit)
        return np.mean(fitness)

    def getnparameters(self):
        ninputs, nhiddens, noutputs = self.ninputs, self.nhiddens, self.noutputs
        nparameters = nhiddens*ninputs + noutputs*nhiddens + nhiddens + noutputs
               
        return nparameters

    def setparameters(self, genotype):
        ninputs, nhiddens, noutputs = self.ninputs, self.nhiddens, self.noutputs
 
        W1 = genotype[0:nhiddens*ninputs]
        W2 = genotype[nhiddens*ninputs:nhiddens*ninputs+noutputs*nhiddens]
        W1.resize(nhiddens,ninputs)
        W2.resize(noutputs, nhiddens)
        b1 = np.zeros(shape=(nhiddens, 1))       
        b2 = np.zeros(shape=(noutputs, 1))     
        return [W1, b1, W2, b2]

env = gym.make("CartPole-v0") # "Pendulum-v1"
network = Network(env, 5)
np.random.seed(123)

popsize = 10
generange = 0.1 
mutrange = 0.02 
nepisodes = 3
ngenerations = 100

nparameters = network.getnparameters()
population = np.random.randn(popsize, nparameters) 

for g in range(ngenerations):
    fitness = []
    for i in range(popsize):
        params = network.setparameters(population[i])
        fit = network.evaluate(env, nepisodes, params)
        fitness.append(fit) 
        index_fit = np.argsort(fitness)

    for i in range(popsize//2):
        population[index_fit[i]] = population[index_fit[popsize//2 + i]] + np.random.randn(1,nparameters) * mutrange
    print (f"generation {g+1} fitness best {np.max(fitness):.1f} fitness average {np.mean(fitness):.1f}")
    if np.max(fitness) >=200:
        break

network.evaluate(env, nepisodes, params, True)
env.close()