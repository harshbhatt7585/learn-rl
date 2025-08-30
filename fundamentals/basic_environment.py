class BasicGridWorld:
    def __init__(self, width=4, height=4):
        self.size = width * height
        self.state = (0, 0)
        self.goal = self.size - 1
        self.actions = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.current_place = 0
        self.width = width
        self.height = height

    def reset(self):
        self.state = (0, 0)
        return self.state
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
        
        if not valid_move:
            reward = -1
            new_state = self.current_place
        else:
            self.current_place = new_place
            # give rewards based on how close the goal is than the last place
            current_distance_to_goal = abs(self.current_place - self.goal)
            last_distance_to_goal = abs(last_place - self.goal)
            reward = last_distance_to_goal - current_distance_to_goal
            new_state = self.current_place
        
        self.state = new_state
        done = self.state == self.goal

        return new_state, reward, done

    

if __name__ == "__main__":
    env = BasicGridWorld()
    env.reset()
    env.step(1)
    new_state, reward, done = env.step(1)
    new_state, reward, done = env.step(1)
    new_state, reward, done = env.step(3)
    new_state, reward, done = env.step(3)
    new_state, reward, done = env.step(3)

    print(reward)
    print(done)


