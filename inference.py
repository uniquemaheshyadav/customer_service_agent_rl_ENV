import os
from openai import OpenAI
from env.support_env import SupportEnv

def get_action_from_llm(client, model_name, state_desc, history_actions):
    """
    Queries the LLM to select ONE action based on structured state description.
    actions: respond, ask_clarification, escalate, search_kb, verify_identity, transfer_to_dept
    """
    actions = ["respond", "ask_clarification", "escalate", "search_kb", "verify_identity", "transfer_to_dept"]
    
    prompt = f"""
You are an advanced AI Customer Support Agent.
Your goal is to resolve the customer's issue efficiently and with high satisfaction.

Allowed Actions:
- respond: Give a final answer to the customer.
- ask_clarification: Ask the customer for more details.
- escalate: Higher-tier support (use for High priority or if stuck).
- search_kb: Search the Knowledge Base for technical/billing info.
- verify_identity: Required BEFORE responding to sensitive requests (like refunds).
- transfer_to_dept: Use for billing-specific queries.

STRATEGY:
1. If it's a 'refund', you MUST 'verify_identity' first.
2. If it's 'tech', 'search_kb' first to get the solution.
3. If it's 'billing', 'transfer_to_dept' is the best route.
4. If sentiment is very low, consider 'escalate' or 'ask_clarification'.

CURRENT STATE:
{state_desc}

ACTIONS TAKEN SO FAR:
{history_actions}

Output ONLY the word of the action you wish to take.
Output:"""

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a logical support routing agent. Output exactly one word from the allowed actions list."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=15
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
    
    print(f"[START] task=customer_support env=advanced model={model_name}")

    done = False
    step_num = 0
    rewards = []
    success = False
    history_actions = []
    
    query_type = info["query_str"]

    while not done:
        step_num += 1
        
        # Construct state description for LLM
        state_desc = (
            f"- Query Type: {query_type}\n"
            f"- Customer Sentiment: {obs['sentiment'][0]:.2f}\n"
            f"- Priority: {['Low', 'Medium', 'High'][obs['priority']]}\n"
            f"- Identity Verified: {'Yes' if obs['identity_verified'] else 'No'}\n"
            f"- KB Info Retrieved: {'Yes' if obs['kb_info_active'] else 'No'}\n"
        )
        
        # Add KB info to state if available
        if obs['kb_info_active']:
            state_desc += f"- KB Hit: {info.get('kb_hit', 'N/A')}\n"

        action_idx, action_str = get_action_from_llm(client, model_name, state_desc, history_actions)
        history_actions.append(action_str)
        
        try:
            obs, reward, terminated, truncated, env_info = env.step(action_idx)
            info = env_info # Update info for next step (e.g. for kb_hit)
            done = terminated or truncated
            error_msg = "null"
        except Exception as e:
            reward = 0.0
            done = True
            error_msg = str(e).replace(' ', '_')
            
        rewards.append(reward)
        if reward > 5: # Threshold for success in new environment
            success = True
            
        reward_fmt = f"{reward:.2f}"
        done_fmt = "true" if done else "false"
        
        print(f"[STEP] step={step_num} action={action_str} reward={reward_fmt} done={done_fmt} error={error_msg}")

    success_fmt = "true" if success else "false"
    rewards_str = ",".join([f"{r:.2f}" for r in rewards])
    print(f"[END] success={success_fmt} steps={step_num} rewards={rewards_str}")

if __name__ == "__main__":
    run_episode()
