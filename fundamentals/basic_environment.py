class BasicGridWorld:
    def __init__(self, width=4, height=4):
        self.size = width * height
        self.state = (0, 0)
        self.goal = self.size - 1
        self.actions = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.current_place = 0
        self.width = width
        self.height = height
        self.step_count = 0

    def reset(self):
        self.current_place = 0
        self.step_count = 0
        return self.current_place
    

    def step(self, action):
        last_place = self.current_place
        if action == 0:  # UP
            new_place = self.current_place - self.width
        elif action == 1:  # DOWN
            new_place = self.current_place + self.width
        elif action == 2:  # LEFT
            new_place = self.current_place - 1
        elif action == 3:  # RIGHT
            new_place = self.current_place + 1
        
        # Check for valid moves (within bounds and not crossing row boundaries for left/right)
        valid_move = True
        if new_place < 0 or new_place >= self.size:
            valid_move = False
        elif action == 2 and self.current_place % self.width == 0:  # LEFT at left edge
            valid_move = False
        elif action == 3 and self.current_place % self.width == self.width - 1:  # RIGHT at right edge
            valid_move = False
        
        # Default: step cost to encourage shorter paths
        if not valid_move:
            reward = -2
            new_state = self.current_place
        else:
            self.current_place = new_place
            new_state = self.current_place
            reward = -1
        
        self.state = new_state
        done = self.state == self.goal
        self.step_count += 1

        # Bonus on reaching goal
        if done:
            reward += 10

        return new_state, reward, done, self.step_count

    

if __name__ == "__main__":
    env = BasicGridWorld()
    env.reset()
    env.step(1)
    new_state, reward, done, step_count = env.step(1)
    new_state, reward, done, step_count = env.step(1)
    new_state, reward, done, step_count = env.step(3)
    new_state, reward, done, step_count = env.step(3)
    new_state, reward, done, step_count = env.step(3)

    print(reward)
    print(done)


