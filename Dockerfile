# Use a slim Python 3.11 base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose the app on the standard port
EXPOSE 7860

# Run the server
CMD ["python", "server.py"]
