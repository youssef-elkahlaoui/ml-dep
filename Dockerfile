# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV AZURE_WEBAPP_NAME="MLPRO"

# Run app.py when the container launches
CMD ["gunicorn", "-b", "0.0.0.0:80", "app:app"]