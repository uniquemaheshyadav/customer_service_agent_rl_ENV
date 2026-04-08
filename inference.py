import os
from openai import OpenAI
from env.support_env import SupportEnv

def get_action_from_llm(client, model_name, query, history_actions):
    """
    Takes a string query and history of actions, queries the LLM to select ONE action.
    actions: respond, ask_clarification, escalate, search_kb
    Returns (action_index, action_string)
    """
    actions = ["respond", "ask_clarification", "escalate", "search_kb"]
    
    prompt = f"""
You are an AI customer support routing agent.
Your task is to take a customer's query type and the history of actions already taken, 
and output EXACTLY ONE OF THE FOLLOWING WORDS as your next action:
- respond
- ask_clarification
- escalate
- search_kb

Do not add punctuation, reasoning, or extra text. Only exactly one word from the choices above.

EXAMPLES:
Query: refund
Actions taken: []
Output: escalate

Query: technical_issue
Actions taken: ['search_kb']
Output: escalate

Query: general_query
Actions taken: []
Output: respond

CURRENT STATE:
Query: {query}
Actions taken: {history_actions}
Output:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a precise classification agent. Output strictly ONE word from the allowed list."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        raw_output = response.choices[0].message.content.strip().lower()
        
        # Match output to valid action
        for idx, act in enumerate(actions):
            if act in raw_output:
                return idx, act
                
        return 0, actions[0]
        
    except Exception as e:
        return 0, actions[0]

def run_episode():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is missing.")

    api_base_url = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4.1-mini")

    client = OpenAI(
        api_key=hf_token,
        base_url=api_base_url
    )

    env = SupportEnv()
    obs, info = env.reset()
    query_str = info["query_str"]
    
    print(f"[START] task=customer_support env=custom model={model_name}")

    done = False
    step_num = 0
    rewards = []
    success = False
    
    history_actions = []

    while not done:
        step_num += 1
        
        action_idx, action_str = get_action_from_llm(client, model_name, query_str, history_actions)
        history_actions.append(action_str)
        
        try:
            obs, reward, terminated, truncated, env_info = env.step(action_idx)
            done = terminated or truncated
            error_msg = "null"
        except Exception as e:
            reward = 0.0
            done = True
            error_msg = str(e).replace(' ', '_')
            
        rewards.append(reward)
        if reward > 0:
            success = True
            
        reward_fmt = f"{reward:.2f}"
        done_fmt = "true" if done else "false"
        
        print(f"[STEP] step={step_num} action={action_str} reward={reward_fmt} done={done_fmt} error={error_msg}")

    success_fmt = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_fmt} steps={step_num} rewards={rewards_str}")

if __name__ == "__main__":
    run_episode()
