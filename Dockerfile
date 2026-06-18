FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port 8000
EXPOSE 8000

# Start the FastAPI server using the CLI serve command
CMD ["python", "main.py", "serve", "--host", "0.0.0.0", "--port", "8000"]
