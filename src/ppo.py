import numpy as np
import torch

class PPO():
    def __init__(self, action_size, actor, critic):
        self.log_probability = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.entropies = []
        
        self.state_size = (4,84,84)
        self.action_size = action_size
        self.gamma = 0.99
        self.lamda = 0.95    #for GAE calculation
        self.number_steps = 128
        self.batch_size = 32
        self.ppo_epoch = 3
        self.clip_param = 0.1
        
        self.actor = actor(self.state_size, action_size)
        self.critic = critic(self.state_size)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=1e-4)

        self.t_step = 0

    def reset(self):
        self.log_probability = []
        self.values = []
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []
        self.entropies = []
        
    def act(self, state):
        """Returns action, log_prob, value for given state as per current policy."""
        
        state = torch.from_numpy(state).unsqueeze(0)
        action_probs = self.actor(state)
        value = self.critic(state)

        action = action_probs.sample() #get prob of taking action
        log_prob = action_probs.log_prob(action) #log prob

        return action.item(), log_prob, value
        
    def remember(self, state, action, value, log_prob, reward, done, next_state):
        
        # Save experience in  memory
        self.log_probability.append(log_prob)
        self.values.append(value)
        self.states.append(torch.from_numpy(state).unsqueeze(0))
        self.rewards.append(torch.Tensor(np.array([reward])))
        self.actions.append(torch.from_numpy(np.array([action])))
        self.masks.append(torch.Tensor(np.array([1 - done])))

        self.t_step += 1
        
        if self.t_step == self.number_steps:
            print("reset")
            self.update_target(next_state)
            self.reset()

    def ppo_iter(self, returns, advantage):
        memory_size = self.states.size(0)
        for _ in range(self.states.size(0) // self.batch_size):
            rand_ids = np.random.randint(0, memory_size, self.batch_size)
            yield self.states[rand_ids, :], self.actions[rand_ids], self.log_probability[rand_ids], returns[rand_ids, :], advantage[rand_ids, :]  
            
    def update_target(self, next_state):
        next_state = torch.from_numpy(next_state).unsqueeze(0)
        next_value = self.critic(next_state)

        returns = torch.cat(self.calc_generalizedAE(next_value)).detach()
        self.log_probability = torch.cat(self.log_probability).detach()
        self.values = torch.cat(self.values).detach()
        self.states = torch.cat(self.states)
        self.actions = torch.cat(self.actions)
        advantages = returns - self.values

        for ep in range(self.ppo_epoch):
            for state, action, prev_prob, ea_return, advantage in self.ppo_iter(returns, advantages):

                dist = self.actor(state)
                value = self.critic(state)

                entropy_bonus = dist.entropy()
                entropy_bonus = entropy_bonus.mean()
                
                new_prob = dist.log_prob(action)

                ratio = torch.exp(new_prob - prev_prob)

                reward  = torch.min(ratio * advantage,
                                          torch.clamp(ratio, min=1.0 - self.clip_param,max=1.0 + self.clip_param)
                                          * advantage).mean()
                
                vf_loss = (ea_return - value) ** 2
                
                loss = - reward + 0.5 * vf_loss.mean() - 0.001 * entropy_bonus

                # Minimize the loss
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                
                loss.backward()
                
                self.actor_opt.step()
                self.critic_opt.step()

        self.reset()

    def calc_generalizedAE(self, next_value):
        generalized_advantage_estimation = 0
        returns = []
        values = self.values + [next_value]
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + self.gamma * values[step + 1] * self.masks[step] - values[step]
            generalized_advantage_estimation = delta + self.gamma * self.lamda * self.masks[step] * generalized_advantage_estimation
            returns.append(generalized_advantage_estimation)
        return returns[::-1]
