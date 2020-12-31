import gym
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random, choice
import json
from tempfile import TemporaryFile

outfile = TemporaryFile()

env = gym.make("MountainCar-v0")
env._max_episode_steps = 500

totalNumberOfSteps = 1000000
number_of_samples = 100

sigma_v = 0.0004
sigma_p = 0.04
diag = np.array([[sigma_p, 0], [0, sigma_v]])
diag_inv = np.linalg.inv(diag)
number_of_actions = env.action_space.n
min_pos = env.min_position
max_pos = env.max_position
max_spd = env.max_speed
min_spd = -env.max_speed
takeSample = np.append(np.zeros(1), (np.linspace(0, totalNumberOfSteps, number_of_samples)))
policy_value = []
c_p = np.linspace(min_pos, max_pos, 6)
c_v = np.linspace(min_spd, max_spd, 8)

gaussian_mat = [(p, v) for p in c_p for v in c_v]
num_of_features = c_v.size * c_p.size


def epsilon_greedy(epsilon, state, theta):
    explore = (random() < epsilon)
    if explore:
        return np.random.randint(0, number_of_actions)
    else:
        # action_value = [np.matmul(w[:, action], theta) for action in range(number_of_actions)]
        return soft_max(theta, state)


def soft_max(theta, state):
    action_values = [state @ np.exp(theta[:, action]) for action in range(number_of_actions)]
    action_prob = action_values / np.sum(action_values)
    rand = random()
    if rand < action_prob[0]:
        return 0
    elif rand - action_prob[0] < action_prob[1]:
        return 1
    else:
        return 2


def getState(p, v):
    x_vector = np.zeros((num_of_features, 2))
    theta_mat = np.zeros(num_of_features)
    for i in range(num_of_features):
        x_vector[i] = np.abs(np.array([p, v]).transpose() - np.array(gaussian_mat[i]).transpose())
        theta_mat[i] = np.exp(-0.5 * (np.matmul(np.matmul(np.transpose(x_vector[i]), diag_inv), x_vector[i])))
    return theta_mat


def simulate(w, env):
    sum = 0.0
    for sim in range(100):
        (p, v) = env.reset()
        done = False
        while not done:
            (p, v), reward, done, info = env.step(epsilon_greedy(w, 0, getState(p, v)))
            sum += reward
    return sum / 100


def getMean(theta, state):
    action_values = [state @ np.exp(theta[:, action]) for action in range(number_of_actions)]
    action_prob = action_values / np.sum(action_values)
    mean = action_prob @ action_values
    return mean


def learnActorCritic(theta):  # theta is policy approximation
    w = np.zeros(num_of_features)  # aprox value function
    l = 1
    gamma = 1
    stepSizePolicy = 0.2
    stepSizeValue = 0.12

    next_sample = 0
    epsilon = 1
    episode = 1
    curr_reward = 0
    timeSteps = 0
    while timeSteps < totalNumberOfSteps:
        # if next_sample < len(takeSample) and timeSteps >= takeSample[next_sample]:
        #     policy_value.append(simulate(w, env))
        #     next_sample += 1
        done = False
        p, v = env.reset()
        state = getState(p, v)
        while not done:
            timeSteps += 1
            action = epsilon_greedy(epsilon, state, theta)
            (p, v), reward, done, _ = env.step(action)
            nextState = getState(p, v)
            if done:
                delta = reward - (state @ w)
            else:
                delta = reward + gamma * (nextState @ w.T) - (state @ w.T)
            w += stepSizeValue * delta * state
            mean = getMean(theta, state)
            grad = state @ w.T - mean
            theta[:, action] += stepSizePolicy * l * delta * grad
            l *= gamma
            state = nextState
            curr_reward += reward
        if episode % 100 == 0 and episode > 0:
            print('episode ', episode, 'score ', curr_reward, 'epsilon %.3f' % epsilon)
            curr_reward = 0
            # render_simulation(theta)
        episode += 1
        epsilon *= 0.999


def render_simulation(theta):
    (p, v) = env.reset()
    env.render()
    done = False
    while not done:
        (p, v), reward, done, info = env.step(epsilon_greedy(0, getState(p, v), theta))
        env.render()
    env.reset()
    env.close()


if __name__ == '__main__':
    with open('aprox_policy.npy', 'rb') as f:
        theta = np.load(f)
        # render_simulation(theta)
    learnActorCritic(theta);
