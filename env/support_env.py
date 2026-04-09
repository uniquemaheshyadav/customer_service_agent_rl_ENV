import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class SupportEnv(gym.Env):
    """
    Advanced Customer Support RL Environment.
    Simulates multi-step support interactions requiring tool-use (KB), 
    identity verification, and department routing.
    """
    metadata = {"render_modes": ["human"]}
    
    def __init__(self):
        super(SupportEnv, self).__init__()
        
        self.action_names = [
            "respond",           # 0
            "ask_clarification", # 1
            "escalate",          # 2
            "search_kb",         # 3
            "verify_identity",   # 4
            "transfer_to_dept"   # 5
        ]
        
        # Action space: 6 discrete actions
        self.action_space = spaces.Discrete(len(self.action_names))
        
        # Structured observation space
        self.observation_space = spaces.Dict({
            "query_type": spaces.Discrete(4), # 0: refund, 1: tech, 2: billing, 3: general
            "sentiment": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "priority": spaces.Discrete(3), # 0: Low, 1: Medium, 2: High
            "identity_verified": spaces.Discrete(2),
            "kb_info_active": spaces.Discrete(2),
            "steps_taken": spaces.Discrete(10)
        })
        
        self.knowledge_base = {
            "tech": "Error 404 can be fixed by clearing cache.",
            "billing": "Invoice history is available in the Billing Portal under 'Statements'.",
            "refund": "Refunds are processed within 5-7 business days after approval."
        }
        
        self.scenarios = [
            {"type": "refund", "priority": 2, "initial_sentiment": 0.3, "require_verify": True},
            {"type": "tech", "priority": 1, "initial_sentiment": 0.5, "require_kb": True},
            {"type": "billing", "priority": 0, "initial_sentiment": 0.8, "require_transfer": True},
            {"type": "general", "priority": 0, "initial_sentiment": 0.9, "require_none": True}
        ]
        
        self.reset()

    def _get_obs(self):
        query_map = {"refund": 0, "tech": 1, "billing": 2, "general": 3}
        return {
            "query_type": query_map[self.current_scenario["type"]],
            "sentiment": np.array([self.sentiment], dtype=np.float32),
            "priority": self.current_scenario["priority"],
            "identity_verified": 1 if self.identity_verified else 0,
            "kb_info_active": 1 if self.kb_info_active else 0,
            "steps_taken": min(self.step_count, 9)
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.max_steps = 6
        self.current_scenario = random.choice(self.scenarios)
        self.sentiment = self.current_scenario["initial_sentiment"]
        self.identity_verified = False
        self.kb_info_active = False
        self.done = False
        
        info = {
            "query_str": self.current_scenario["type"],
            "description": f"Customer is asking about {self.current_scenario['type']}."
        }
        return self._get_obs(), info

    def step(self, action):
        self.step_count += 1
        reward = -0.5  # Step penalty to encourage efficiency
        action_str = self.action_names[action]
        
        # Logic for Verify Identity
        if action_str == "verify_identity":
            if not self.identity_verified:
                self.identity_verified = True
                reward += 2.0
            else:
                reward -= 1.0  # Redundant action penalty
                
        # Logic for KB Search
        elif action_str == "search_kb":
            self.kb_info_active = True
            reward += 1.0
            
        # Logic for Respond
        elif action_str == "respond":
            s = self.current_scenario
            if s["type"] == "refund":
                if self.identity_verified:
                    reward += 10.0
                    self.done = True
                else:
                    reward -= 5.0 # Failed to verify
                    self.sentiment *= 0.5
            elif s["type"] == "tech":
                if self.kb_info_active:
                    reward += 8.0
                    self.done = True
                else:
                    reward += 2.0 # Generic response
                    self.sentiment *= 0.8
            elif s["type"] == "general":
                reward += 7.0
                self.done = True
            else:
                reward -= 2.0 # Wrong department/logic
                
        # Logic for Transfer
        elif action_str == "transfer_to_dept":
            if self.current_scenario["type"] == "billing":
                reward += 10.0
                self.done = True
            else:
                reward -= 5.0
                
        # Logic for Escalate
        elif action_str == "escalate":
            if self.current_scenario["priority"] == 2 or self.step_count > 4:
                reward += 5.0
                self.done = True
            else:
                reward -= 2.0 # Unnecessary escalation
        
        # Penalize low sentiment/frustrated customers
        if self.sentiment < 0.2:
            reward -= 2.0

        if self.step_count >= self.max_steps:
            self.done = True
            
        terminated = self.done
        truncated = False
        
        info = {
            "query_str": self.current_scenario["type"],
            "action_str": action_str,
            "kb_hit": self.knowledge_base.get(self.current_scenario["type"], "N/A") if self.kb_info_active else "None"
        }
        
        return self._get_obs(), float(reward), terminated, truncated, info
