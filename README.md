# Advanced Customer Support RL Environment (OpenEnv)

This repository contains a high-fidelity Reinforcement Learning environment for Customer Support Agents, built for the **Meta PyTorch OpenEnv Hackathon**.

## Features

- **Stateful Interactions**: Tracks customer sentiment, ticket priority, and identity verification status.
- **Tool-Use Simulation**: Agents must query a Knowledge Base (KB) to retrieve solutions for technical issues.
- **Complex Action Space**: Includes actions like `verify_identity`, `search_kb`, and `transfer_to_dept`.
- **Multi-Step Trajectories**: Challenges agents to resolve high-priority requests through logical sequences (e.g., Verify -> Resolve).
- **Gymnasium Compatible**: Follows the standard `gym.Env` interface.

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**:
   ```bash
   export HF_TOKEN="your_huggingface_token"
   export API_BASE_URL="https://api.openai.com/v1"
   export MODEL_NAME="gpt-4.1-mini"
   ```

## Running the Agent

To run a single episode using the LLM agent:
```bash
python inference.py
```

## Evaluation & Submission

To run a batch of evaluation episodes and generate metrics:
```bash
python submit.py
```

### Expected Flow (Example)
1. **START**: Query = `refund`, Priority = `High`.
2. **STEP 1**: Agent uses `verify_identity`. (Reward +2.0)
3. **STEP 2**: Agent uses `respond`. (Reward +10.0, Success)
4. **END**: Total Reward = 11.5 (including step penalties).

## Environment Configuration
The environment is registered via `openenv.yaml`:
- **Name**: `customer-support-env`
- **Entry Point**: `env.support_env:SupportEnv`
