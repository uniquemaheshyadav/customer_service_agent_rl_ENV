# customer_service_agent_rl_ENV

# AI Customer Support RL Environment

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set environment variables (Linux/macOS):
```bash
export HF_TOKEN="your_huggingface_token"
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4.1-mini"
```

For Windows:
```cmd
set HF_TOKEN=your_huggingface_token
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4.1-mini
```

## Running the Inference

To run the custom scenario:
```bash
python inference.py
```

## Expected Output Format

```
[START] task=customer_support env=custom model=gpt-4.1-mini
[STEP] step=1 action=escalate reward=10.00 done=true error=null
[END] success=true steps=1 rewards=10.00
```
