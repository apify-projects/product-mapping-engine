import json
import os
import pandas as pd
from slugify import slugify
from apify_client import ApifyClient

IMAGE_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/images")

def save_dataset_for_executor(labeled_dataset_id, dataset1_id, dataset2_id, images_kvs1_id, images_kvs2_id):
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])

    dataset1_id = client.datasets().get_or_create(name='pmtest2-dataset1')['id']
    print(f"Dataset1 = {dataset1_id}")
    dataset2_id = client.datasets().get_or_create(name='pmtest2-dataset2')['id']
    print(f"Dataset2 = {dataset2_id}")
    labeled_dataset_id = client.datasets().get_or_create(name='pmtest2-datasetl')['id']
    print(f"Labeled dataset = {labeled_dataset_id}")

    images_kvs1_id = client.key_value_stores().get_or_create(name='pmtest2-image-kvs1')['id']
    print(f"Images KVS 1 = {images_kvs1_id}")
    images_kvs2_id = client.key_value_stores().get_or_create(name='pmtest2-image-kvs2')['id']
    print(f"Images KVS 2 = {images_kvs2_id}")

    # Prepare storages and read data
    dataset1_client = client.dataset(dataset1_id)
    dataset2_client = client.dataset(dataset2_id)
    labeled_dataset_client = client.dataset(labeled_dataset_id)
    images_kvs1_client = client.key_value_store(images_kvs1_id)
    images_kvs2_client = client.key_value_store(images_kvs2_id)

    dataset_to_upload = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data/pairs_dataset.csv"))
    #labeled_dataset_client.push_items(dataset_to_upload.to_dict('records'))

    dataset1_to_upload = dataset_to_upload[[
        "name1",
        "short_description1",
        "long_description1",
        "specification1",
        "image1",
        "price1",
        "url1"
    ]]
    dataset1_to_upload = dataset1_to_upload.rename(columns={
        "name1": "name",
        "short_description1": "short_description",
        "long_description1": "long_description",
        "specification1": "specification",
        "image1": "image",
        "price1": "price",
        "url1": "url"
    })
    dataset1_to_upload['id'] = dataset1_to_upload['name']
    #dataset1_client.push_items(dataset1_to_upload.to_dict('records'))

    dataset2_to_upload = dataset_to_upload[[
        "name2",
        "short_description2",
        "long_description2",
        "specification2",
        "image2",
        "price2",
        "url2"
    ]]
    dataset2_to_upload = dataset2_to_upload.rename(columns={
        "name2": "name",
        "short_description2": "short_description",
        "long_description2": "long_description",
        "specification2": "specification",
        "image2": "image",
        "price2": "price",
        "url2": "url"
    })
    dataset2_to_upload['id'] = dataset2_to_upload['name']
    dataset2_client.push_items(dataset2_to_upload.to_dict('records'))

    datasets = [
        dataset1_client.list_items().items,
        dataset2_client.list_items().items
    ]

    images_kvs_clients = [
        images_kvs1_client,
        images_kvs2_client
    ]

    for e in range(2):
        counter = 0
        for item in datasets[e]:
            for f in range(1, item['image'] + 1):
                with open(os.path.join(IMAGE_DIRECTORY, "pair_{}_product_{}_image_{}".format(counter, e+1, f)), mode='rb') as image:
                    print(slugify(item['id'] + '_image_{}'.format(f-1)))
                    images_kvs_clients[e].set_record(slugify(item['id'] + '_image_{}'.format(f-1)), image.read())

            print('Item {} uploaded'.format(counter))
            counter += 1


save_dataset_for_executor("", "", "", "", "")