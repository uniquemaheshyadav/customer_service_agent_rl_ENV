import gymnasium as gym
from gymnasium import spaces
import random

class SupportEnv(gym.Env):
    """
    Custom Environment that follows gym interface for Customer Support RL.
    Queries: refund, technical_issue, general_query
    Actions: respond (0), ask_clarification (1), escalate (2), search_kb (3)
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        super(SupportEnv, self).__init__()
        
        self.queries = ["refund", "technical_issue", "general_query"]
        self.action_names = ["respond", "ask_clarification", "escalate", "search_kb"]
        
        # Action space: 4 discrete actions
        self.action_space = spaces.Discrete(4)
        
        self.observation_space = spaces.Discrete(3)
        
        self.current_query = None
        self.current_query_idx = None
        self.step_count = 0
        self.max_steps = 3

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_query_idx = random.randint(0, 2)
        self.current_query = self.queries[self.current_query_idx]
        
        info = {"query_str": self.current_query}
        return self.current_query_idx, info

    def step(self, action):
        self.step_count += 1
        done = False
        reward = 0.0
        
        action_str = self.action_names[action]
        
        if self.current_query == "refund":
            if action_str == "escalate":
                reward = 10.0
                done = True
            elif action_str == "respond":
                reward = -5.0
                done = True
            elif action_str == "ask_clarification":
                reward = 2.0
            elif action_str == "search_kb":
                reward = -1.0
                
        elif self.current_query == "technical_issue":
            if action_str == "search_kb":
                reward = 8.0
            elif action_str == "escalate":
                reward = 5.0
                done = True
            elif action_str == "ask_clarification":
                reward = 2.0
            elif action_str == "respond":
                reward = -5.0
                done = True
                
        elif self.current_query == "general_query":
            if action_str == "respond":
                reward = 7.0
                done = True
            elif action_str == "escalate":
                reward = -5.0
                done = True
            elif action_str == "search_kb":
                reward = 2.0
            elif action_str == "ask_clarification":
                reward = 0.0

        if self.step_count >= self.max_steps:
            done = True
            
        info = {"query_str": self.current_query, "action_str": action_str}
        
        terminated = done
        truncated = False
        
        return self.current_query_idx, float(reward), terminated, truncated, info
