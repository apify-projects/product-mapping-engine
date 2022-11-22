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

RUN curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
RUN apt-get -y install nodejs

# Install the packages specified in requirements.txt,
# Print the installed Python version, pip version
# and all installed packages with their versions for debugging
# TODO change the branch
RUN echo "Python version:" \
 && python --version \
 && echo "Pip version:" \
 && pip --version \
 && echo "Installing dependencies from requirements.txt:" \
 && pip install -r requirements.txt \
 && echo "All installed Python packages:" \
 && pip freeze \
 && git clone -b DatasetPreprocessor https://ssh:$GITHUB_TOKEN@github.com/apify-projects/product-mapping-engine.git product_mapping_repository \
 && ln -s product_mapping_repository/engine product_mapping_engine \
 && (cd product_mapping_engine/scripts/dataset_handler/preprocessing/images/image_hash_creator && npm install)

# Next, copy the remaining files and directories with the source code.
# Since we do this after installing the dependencies, quick build will be really fast
# for most source file changes.
COPY . ./

# Specify how to launch the source code of your actor.
# By default, the main.py file is run
CMD python3 main.py
