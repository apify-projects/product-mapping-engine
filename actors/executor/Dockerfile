# First, specify the base Docker image.
# You can see the Docker images from Apify at https://hub.docker.com/r/apify/.
# You can also use any other image from Docker Hub.
FROM apify/actor-python:3.9

RUN apt-get update
RUN apt install -y libgl1-mesa-glx

# Second, copy just requirements.txt into the actor image,
# since it should be the only file that affects "pip install" in the next step,
# in order to speed up the build
COPY requirements.txt ./

ARG GITHUB_TOKEN
RUN install_dependencies.sh $GITHUB_TOKEN

# Next, copy the remaining files and directories with the source code.
# Since we do this after installing the dependencies, quick build will be really fast
# for most source file changes.
COPY . ./

# Specify how to launch the source code of your actor.
# By default, the main.py file is run
CMD python3 main.py
