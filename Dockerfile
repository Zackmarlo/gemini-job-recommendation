FROM python:3.10-slim

# Install required system packages
RUN apt update && apt install -y git

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app /app

ENV GEMINI_API_KEY=your_gemini_api_key_here

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
