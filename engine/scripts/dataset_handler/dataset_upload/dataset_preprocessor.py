import cv2
import json
import numpy as np
import os
import pandas as pd
import base64
import requests
import imghdr
from slugify import slugify

import importlib.machinery

configuration = importlib.machinery.SourceFileLoader(
    'configuration',
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..", "configuration.py"),
).load_module()

IMAGES_CHUNK_SIZE = 1000
IMAGES_PATH = "product_images"

def is_url(potential_url):
    return "http" in potential_url or ".com" in potential_url

def download_images(dataset):
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)

    image_counts = []
    for index, item in dataset.iterrows():
        downloaded_images = 0
        for potential_url in item["image"]:
            if is_url(potential_url):
                try:
                    response = requests.get(potential_url, timeout=5)
                except:
                    print(response)
                    continue

                print(response)
                if response.ok:
                    if imghdr.what("", h=response.content) is None:
                        continue

                    image_file_path = os.path.join(
                        IMAGES_PATH,
                        'product_{}_image_{}.jpg'.format(index, downloaded_images + 1)
                    )

                    image = cv2.imdecode(np.asarray(bytearray(response.content), dtype="uint8"), cv2.IMREAD_COLOR)

                    # decreasing the size of the images when needed to increase speed and preserve memory
                    width_resize_ratio = configuration.IMAGE_RESIZE_WIDTH / image.shape[1]
                    height_resize_ratio = configuration.IMAGE_RESIZE_HEIGHT / image.shape[0]
                    resize_ratio = min(width_resize_ratio, height_resize_ratio)
                    if resize_ratio < 1:
                        image = cv2.resize(image, (0, 0), fx=resize_ratio, fy=resize_ratio)

                    cv2.imwrite(image_file_path, image)
                    downloaded_images += 1

        image_counts.append(downloaded_images)

    print(image_counts)
    return image_counts

def upload_images_to_kvs(dataset, images_kvs_client):
    counter = 0
    chunks = 0
    collected_images = {}
    for index, item in dataset.iterrows():
        for f in range(1, item['image'] + 1):
            with open(os.path.join(IMAGES_PATH, "product_{}_image_{}.jpg".format(index, f)), mode='rb') as image:
                collected_images[slugify(item['id'] + '_image_{}'.format(f - 1))] = str(
                    base64.b64encode(image.read()),
                    'utf-8'
                )

        print('Item {} uploaded'.format(counter))
        counter += 1

        if counter % IMAGES_CHUNK_SIZE == 0:
            chunks += 1
            print("images_chunk_{}".format(chunks))
            images_kvs_client.set_record("images_chunk_{}".format(chunks), json.dumps(collected_images))
            collected_images = {}

    if counter % IMAGES_CHUNK_SIZE != 0:
        chunks += 1
        print("images_chunk_{}".format(chunks))
        images_kvs_client.set_record("images_chunk_{}".format(chunks), json.dumps(collected_images))
