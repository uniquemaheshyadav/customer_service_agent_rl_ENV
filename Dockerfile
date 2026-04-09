FROM python:3.10-slim

WORKDIR /app

# Install system dependencies if any
# RUN apt-get update && apt-get install -y ...

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the standard port for Hugging Face Spaces
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
