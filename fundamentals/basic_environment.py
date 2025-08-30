class BasicGridWorld:
    def __init__(self, width=4, height=4):
        self.size = width * height
        self.state = (0, 0)
        self.goal = (self.size-1, self.size-1)
        self.actions = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        self.current_place = 0
        self.width = width
        self.height = height

    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        if action == 0:
            self.current_place = self.current_place - self.width
        elif action == 1:
            self.current_place = self.current_place + self.width
        elif action == 2:
            self.current_place = self.current_place - 1
        elif action == 3:
            self.current_place = self.current_place + 1
        
        if self.current_place < 0 or self.current_place >= self.size:
            reward = -1
            new_state = self.state
        else:
            reward = # I want to give more rewards as close to the goal
            new_state = self.current_place
        
        self.state = new_state
        return new_state, reward, done
 

    

if __name__ == "__main__":
    env = GridWorld()
    env.reset()
    env.step(0)
    env.step(1)
    env.step(2)
    env.step(3)


