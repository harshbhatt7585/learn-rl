from basic_environment import BasicGridWorld

class PPOTrainer:
    def __init__(self, env, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        self.env = env
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.has_continuous_action_space = has_continuous_action_space

    
    def train(self):
        
        pass


if __name__ == "__main__":
    env = BasicGridWorld()
    trainer = PPOTrainer(env, state_dim=1, action_dim=1, lr_actor=0.001, lr_critic=0.001, gamma=0.99, K_epochs=10, eps_clip=0.2, has_continuous_action_space=False)
    trainer.train()
    
