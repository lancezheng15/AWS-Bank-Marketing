# Use Python 3.9 full image instead of slim
FROM --platform=linux/amd64 python:3.9

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create a directory for the models if it doesn't exist
RUN mkdir -p models

# Expose port 80
EXPOSE 80

# Command to run the application
CMD ["streamlit", "run", "app.py", "--server.port", "80", "--server.address", "0.0.0.0"] 