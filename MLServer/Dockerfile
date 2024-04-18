# Use an official Python runtime as a parent image
FROM python:3.9-slim

WORKDIR /
ADD ./requirements.txt /MLServer/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r ./MLServer/requirements.txt

COPY . /MLServer
    

CMD ["python3", "./MLServer/server.py"]
