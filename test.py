import os
import sys

# Set environment variables
os.environ["HF_TOKEN"] = "mock_token"
os.environ["API_BASE_URL"] = "https://mock.api"
os.environ["MODEL_NAME"] = "gpt-4.1-mini"

import inference
from unittest.mock import MagicMock

# Mock OpenAI Client
mock_client = MagicMock()
mock_resp = MagicMock()
mock_resp.choices = [MagicMock()]
# We will make it output "escalate" mapping to something valid
mock_resp.choices[0].message.content = "escalate"
mock_client.chat.completions.create.return_value = mock_resp

# Inject into inference
inference.OpenAI = MagicMock(return_value=mock_client)

print("--- Running Test Episode ---")
inference.run_episode()
