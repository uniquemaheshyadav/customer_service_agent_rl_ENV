import os
import sys
from unittest.mock import MagicMock

# Set environment variables
os.environ["HF_TOKEN"] = "mock_token"
os.environ["API_BASE_URL"] = "https://mock.api"
os.environ["MODEL_NAME"] = "gpt-4.1-mini"

import inference

# Mock OpenAI Client
mock_client = MagicMock()
mock_resp = MagicMock()
mock_resp.choices = [MagicMock()]

# Sequence of actions to simulate a successful path for any scenario
# 1. Search KB or Verify Identity
# 2. Respond or Transfer
mock_resp.choices[0].message.content = "search_kb"

def side_effect(*args, **kwargs):
    # Just return a consistent valid action name
    res = MagicMock()
    res.choices = [MagicMock()]
    # Alternating between search/verify and respond/transfer to simulate progress
    if mock_client.chat.completions.create.call_count % 2 == 0:
        res.choices[0].message.content = "respond"
    else:
        res.choices[0].message.content = "search_kb"
    return res

mock_client.chat.completions.create.side_effect = side_effect

# Inject into inference
inference.OpenAI = MagicMock(return_value=mock_client)

print("--- Running Test Episode (Advanced) ---")
try:
    inference.run_episode()
    print("--- Test Episode Completed Successfully ---")
except Exception as e:
    print(f"--- Test Failed: {e} ---")
    sys.exit(1)
