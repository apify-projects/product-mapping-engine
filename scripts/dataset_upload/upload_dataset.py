import json
import os
import pandas as pd
from slugify import slugify
from apify_client import ApifyClient

def save_dataset_for_executor(task_name):
    # Prepare storages and read data
    dataset_1_client = client.dataset(parameters['dataset_1'])
    dataset_2_client = client.dataset(parameters['dataset_2'])
    images_kvs_1_client = client.key_value_store(parameters['images_kvs_1'])
    images_kvs_2_client = client.key_value_store(parameters['images_kvs_2'])

    dataset_to_upload = pd.read_csv('product_mapping/wdc_dataset/dataset/product_pairs.csv')
    dataset_to_upload = dataset_to_upload.head(200)

    dataset_1_to_upload = dataset_to_upload[['id1', 'name1', 'image1']]
    dataset_1_to_upload = dataset_1_to_upload.rename(columns={
        'id1': 'id',
        'name1': 'name',
        'image1': 'image'
    })
    dataset_1_to_upload['price'] = 1
    dataset_1_to_upload['id'] = dataset_1_to_upload['name']
    dataset_1_client.push_items(dataset_1_to_upload.to_dict('records'))

    dataset_2_to_upload = dataset_to_upload[['id2', 'name2', 'image2']]
    dataset_2_to_upload = dataset_2_to_upload.rename(columns={
        'id2': 'id',
        'name2': 'name',
        'image2': 'image'
    })
    dataset_2_to_upload['price'] = 1
    dataset_2_to_upload['id'] = dataset_2_to_upload['name']
    dataset_2_client.push_items(dataset_2_to_upload.to_dict('records'))

    dataset_images_folder = 'product_mapping/wdc_dataset/dataset/images'

    datasets = [
        dataset_1_client.list_items().items,
        dataset_2_client.list_items().items
    ]

    images_kvs_clients = [
        images_kvs_1_client,
        images_kvs_2_client
    ]

    for e in range(2):
        counter = 0
        for item in datasets[e]:
            for f in range(1, item['image'] + 1):
                with open(os.path.join(dataset_images_folder, "pair_{}_product_{}_image_{}".format(counter, e+1, f)), mode='rb') as image:
                    print(slugify(item['id'] + '_image_{}'.format(f-1)))
                    images_kvs_clients[e].set_record(slugify(item['id'] + '_image_{}'.format(f-1)), image.read())

            print('Item {} uploaded'.format(counter))
            counter += 1
