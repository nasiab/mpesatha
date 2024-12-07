# Use the official Python image from the Docker Hub
FROM python:3.11.2
# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt 

RUN pip install flask flask_cors


COPY . .
# Make port 5000 available to the world outside this container
EXPOSE 8080


# Command to run the app using Python
CMD ["python", "app.py"]
