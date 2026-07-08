FROM python:3.12-slim

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY *.py .

EXPOSE 8501

CMD ["streamlit", "run", "rag_chatbot.py", "--server.headless=true", "--server.address=0.0.0.0"]
