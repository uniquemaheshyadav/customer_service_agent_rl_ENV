from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from env.support_env import SupportEnv
import numpy as np

app = FastAPI(title="OpenEnv Customer Support API")

# Global environment instance
# In a multi-user scenario, you'd want a session-based environment
# but for hackathon evaluation, a single global instance is usually expected.
env = SupportEnv()

class ActionRequest(BaseModel):
    action: int

def sanitize_obs(obs):
    """Convert numpy arrays in observation dict to lists for JSON serialization."""
    if isinstance(obs, dict):
        return {k: sanitize_obs(v) for k, v in obs.items()}
    elif isinstance(obs, np.ndarray):
        return obs.tolist()
    return obs

@app.get("/")
async def root():
    return {
        "name": "Customer Support RL Environment",
        "framework": "OpenEnv",
        "endpoints": ["/reset", "/step", "/health"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/reset")
async def reset():
    try:
        obs, info = env.reset()
        return {
            "observation": sanitize_obs(obs),
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/step")
async def step(request: ActionRequest):
    try:
        obs, reward, terminated, truncated, info = env.step(request.action)
        return {
            "observation": sanitize_obs(obs),
            "reward": float(reward),
            "terminated": terminated,
            "truncated": truncated,
            "info": info
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
