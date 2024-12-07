# Dockerfile
FROM continuumio/miniconda3

WORKDIR /app

# Copy environment.yml before running conda env create
COPY environment.yml .

# Create the environment:
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "bertopic_env", "/bin/bash", "-c"]

RUN pip install flask flask_cors

# Copy the application files
COPY . .

COPY demo_model /app/demo_model
# Set the default environment to avoid activating each time
ENV PATH /opt/conda/envs/bertopic_env/bin:$PATH

EXPOSE 8080
ENV PORT 8080


CMD ["python", "app.py"]


