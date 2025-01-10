FROM python:3.12-slim

WORKDIR /  # Root directory

RUN apt-get update && apt-get install -y \
    gcc g++ make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  

COPY ./app /app 
COPY ./datasets /datasets  

CMD ["streamlit", "run", "app/agent.py"] 
