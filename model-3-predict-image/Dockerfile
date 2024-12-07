# Use an official Python runtime as a parent image
FROM python:3.12.2

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file to the container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install flask flask_cors

COPY . .
# Expose the port that the Flask app will run on
EXPOSE 8080


# Run the Flask app
CMD ["python", "app.py"]
