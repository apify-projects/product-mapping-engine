#!/bin/bash

# Make this script's directory the CWD
cd "$(dirname "${BASH_SOURCE[0]}")"

apt-get update
apt install -y  apt-utils libgl1-mesa-glx curl

curl -fsSL https://deb.nodesource.com/setup_16.x | bash -
apt-get -y install nodejs

# Install the packages specified in requirements.txt,
# Print the installed Python version, pip version
# and all installed packages with their versions for debugging
echo "Python version:" \
 && python --version \
 && echo "Pip version:" \
 && pip --version \
 && echo "Installing dependencies from requirements.txt:" \
 && pip install -r requirements.txt \
 && echo "All installed Python packages:" \
 && pip freeze \

# Check if the product mapping engine was downloaded, if not, download it
if [ -d "../../engine" ]; then
    ln -s ../../engine product_mapping_engine
else
    git clone -b master https://ssh:$1@github.com/apify-projects/product-mapping-engine.git product_mapping_repository \
 && ln -s product_mapping_repository/engine product_mapping_engine
fi

(cd product_mapping_engine/scripts/dataset_handler/preprocessing/images/image_hash_creator && npm install)
