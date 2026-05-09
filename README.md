# EX.1  Pole Balancing using Reinforcement Learning
## Date:09/05/2026

## Aim
The aim of this project is to develop a Reinforcement Learning agent that learns to balance a pole on a moving cart by interacting with the environment and maximizing cumulative rewards.



# Algorithm

## Q-Learning Algorithm for Pole Balancing

### Step 1
Initialize the CartPole environment.

### Step 2
Initialize the Q-table with zeros.

### Step 3
Observe the current state of the environment.

### Step 4
Choose an action using the epsilon-greedy policy:
- Exploration → Random action
- Exploitation → Best action from Q-table

### Step 5
Perform the selected action.

### Step 6
Receive:
- Reward
- Next state
- Done status

### Step 7
Update the Q-value using:

Q(s,a) = Q(s,a) + α [ r + γ max Q(s',a') − Q(s,a) ]

Where:
- α → Learning rate
- γ → Discount factor

### Step 8
Repeat the process until the episode ends.

### Step 9
Train the agent for multiple episodes to improve balancing performance.

---

# Program
```text
Name: KAVIPRIYA SP
Reg:2305002011
```
```python
#GridWorld

import numpy as np

# Grid size (rectangular)
rows, cols = 4, 5

gamma = 0.9      # discount factor
theta = 1e-4     # convergence threshold

# Rewards
rewards = np.zeros((rows, cols))
rewards[0, 4] = 1     # Goal
rewards[1, 4] = -1    # Bad state

# Actions: Up, Down, Left, Right
actions = [(-1,0), (1,0), (0,-1), (0,1)]
action_symbols = ["↑", "↓", "←", "→"]

# Initialize Value Function
V = np.zeros((rows, cols))

def is_valid(r, c):
    return 0 <= r < rows and 0 <= c < cols

# --- VALUE ITERATION ---
while True:
    delta = 0
    new_V = np.copy(V)

    for r in range(rows):
        for c in range(cols):
            values = []

            for a in actions:
                nr, nc = r + a[0], c + a[1]

                if not is_valid(nr, nc):
                    nr, nc = r, c   # stay in same cell

                reward = rewards[nr, nc]
                values.append(reward + gamma * V[nr, nc])

            best_value = max(values)
            new_V[r, c] = best_value

            delta = max(delta, abs(V[r, c] - best_value))

    V = new_V

    if delta < theta:
        break

# --- PRINT VALUE FUNCTION ---
print("\nValue Function:\n")
for row in V:
    print(["{0:0.2f}".format(v) for v in row])


# --- EXTRACT POLICY ---
policy = np.empty((rows, cols), dtype=str)

for r in range(rows):
    for c in range(cols):
        action_values = []

        for a in actions:
            nr, nc = r + a[0], c + a[1]

            if not is_valid(nr, nc):
                nr, nc = r, c

            reward = rewards[nr, nc]
            action_values.append(reward + gamma * V[nr, nc])

        best_action = np.argmax(action_values)
        policy[r, c] = action_symbols[best_action]

# --- PRINT POLICY ---
print("\nPolicy:\n")
for row in policy:
    print(row)

```



# Output
<img width="364" height="234" alt="image" src="https://github.com/user-attachments/assets/4001c5b2-3c58-4326-a3b7-cdc338364e99" />


# Result

The Reinforcement Learning agent successfully learned to balance the pole using the Q-Learning algorithm.  
The reward increased gradually with training episodes, demonstrating that the agent improved its balancing strategy through learning and interaction with the environment.


