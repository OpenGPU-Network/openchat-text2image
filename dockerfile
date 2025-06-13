FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Set the working directory
WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 5555

# Command to run the application
CMD ["python", "main.py"]