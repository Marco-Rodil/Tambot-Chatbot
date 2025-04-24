# Use the official Python 3.12 slim image as the base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port that Flask will run on
EXPOSE 5000

# Set environment variables (optional defaults, will be overridden by Render)
ENV PORT=5000
ENV GEMINI_API_KEY=""

# Command to run the Flask app
CMD ["python", "app.py"]