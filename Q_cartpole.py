import gym 
import numpy as np
import math
import matplotlib.pyplot as plt

timesteps = 1000
total_episodes = 100000
epsilon = 0.5
alpha = 0.8
gamma = 0.999
observation_space_size = 162
time_constant = ((math.log(0.05))/(math.log(0.999)))/(3*total_episodes/4)
#Decides the rate of decay of epsilon to make epsilon fall to 5% after 3/4th of the total episodes
l1 = np.zeros(total_episodes)


def getBox(state_):
    x = state_[0]
    x_dot =  state_[1]
    theta =  state_[2]
    theta_dot =  state_[3]
    
    theta = (180/math.pi)*theta
    theta_dot = (180/math.pi)*theta_dot
    
    if(x > 2.4 or x < -2.4 or theta < -12 or theta > 12):
        return -1
    
    if(x < -0.8):
        box = 0
    elif(x > 0.8):
        box = 1
    else:
        box = 2
    
    if(x_dot < -0.5):
        pass
    elif(x_dot > 0.5):
        box = box + 3
    else:
        box = box + 6
    
    if(theta < -6):
        pass 
    elif(theta < -1):
        box = box + 9
    elif(theta < 0):
        box = box + 18
    elif(theta < 1):
        box = box + 27
    elif(theta < 6):
        box = box + 36
    else:
        box = box + 45
    
    if(theta_dot < -50):
        pass 
    elif(theta < 50):
        box = box + 54
    else:
        box = box + 108
    return box 


env = gym.make('CartPole-v0').env
Q = np.zeros([observation_space_size, env.action_space.n])
temp = np.zeros([observation_space_size, env.action_space.n])
mxm = 0

for episode in range(1, 1+total_episodes):
    epsilon = 0.99**(time_constant*episode)
    state = env.reset()
    box = getBox(state)
    
    for step in range(1, 1+timesteps):
        
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[box])
        
        newState, reward, done, info = env.step(action)
        newBox = getBox(newState)
        Q[box, action] = (1-alpha)*Q[box, action] + alpha*(reward + gamma*(np.max(Q[newBox])))
        box = newBox
        
        if done:
            #print("Episode {} ended after {} timesteps".format(episode, step))
            l1[episode-1] = step
            
            if(step>mxm):
                temp = Q
                mxm = step
            break
        
plt.plot(range(1, total_episodes+1), l1)
plt.xlabel("Episodes")
plt.ylabel("Number of timesteps before fall")
plt.title("Q Learning Agent")
plt.show()

print("Q Table", Q)

'''
R = 0
for episode in range(1, 1+9):
    state = env.reset()
    box = getBox(state)
    
    for step in range(1, 1+timesteps):
        env.render()
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[box])
        
        newState, reward, done, info = env.step(action)
        R = R + reward
        newBox = getBox(newState)
        #Q[box, action] = (1-alpha)*Q[box, action] + alpha*(reward + gamma*(np.max(Q[newBox])))
        box = newBox
        
        if done:
            print("Episode {} ended after {} timesteps".format(episode, step))

            break
env.close()
print(R/10)
'''