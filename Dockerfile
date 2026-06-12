FROM python:3.11-slim

WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# copy requirements first (layer caching)
COPY backend/requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy entire project
COPY . .

# expose port
EXPOSE 8000

# start command
CMD ["uvicorn", "backend.app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]