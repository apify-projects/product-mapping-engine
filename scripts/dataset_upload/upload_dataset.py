import base64
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from slugify import slugify

from apify_client import ApifyClient

def save_dataset_for_executor(pm_run_name, pairs_dataset_path, images_path):
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])

    dataset1_id = client.datasets().get_or_create(name=pm_run_name+'-dataset1')['id']
    print(f"Dataset1 = {dataset1_id}")
    dataset2_id = client.datasets().get_or_create(name=pm_run_name+'-dataset2')['id']
    print(f"Dataset2 = {dataset2_id}")
    labeled_dataset_id = client.datasets().get_or_create(name=pm_run_name+'-datasetl')['id']
    print(f"Labeled dataset = {labeled_dataset_id}")

    images_kvs1_id = client.key_value_stores().get_or_create(name=pm_run_name+'-image-kvs1')['id']
    print(f"Images KVS 1 = {images_kvs1_id}")
    images_kvs2_id = client.key_value_stores().get_or_create(name=pm_run_name+'-image-kvs2')['id']
    print(f"Images KVS 2 = {images_kvs2_id}")

    # Prepare storages and read old_data
    dataset1_client = client.dataset(dataset1_id)
    dataset2_client = client.dataset(dataset2_id)
    labeled_dataset_client = client.dataset(labeled_dataset_id)
    images_kvs1_client = client.key_value_store(images_kvs1_id)
    images_kvs2_client = client.key_value_store(images_kvs2_id)

    dataset1_columns = [
        "name1",
        "short_description1",
        "long_description1",
        "specification1",
        "image1",
        "price1",
        "url1"
    ]
    dataset2_columns = [
        "name2",
        "short_description2",
        "long_description2",
        "specification2",
        "image2",
        "price2",
        "url2"
    ]
    columns_to_upload = dataset1_columns + dataset2_columns + ["match"]

    dataset_to_upload = pd.read_csv(pairs_dataset_path)
    dataset_to_upload = dataset_to_upload[columns_to_upload]
    print(dataset_to_upload)
    dataset_to_upload["long_description2"] = dataset_to_upload["long_description2"].fillna("")

    to_upload_labeled, to_upload_unlabeled = train_test_split(dataset_to_upload, test_size=0.2)
    to_upload_labeled1, to_upload_labeled2 = train_test_split(to_upload_labeled, test_size=0.5)
    labeled_dataset_client.push_items(to_upload_labeled1.to_dict('records'))
    labeled_dataset_client.push_items(to_upload_labeled2.to_dict('records'))

    to_upload_unlabeled.to_csv("aggregated_with_resized_images_unlabeled_data.csv")

    dataset1_to_upload = to_upload_unlabeled[dataset1_columns]
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
    dataset1_client.push_items(dataset1_to_upload.to_dict('records'))

    dataset2_to_upload = to_upload_unlabeled[dataset2_columns]
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
        dataset_to_upload[dataset1_columns].to_dict('records'),
        dataset_to_upload[dataset2_columns].to_dict('records'),
    ]

    images_kvs_clients = [
        images_kvs1_client,
        images_kvs2_client
    ]

    for e in range(2):
        counter = 0
        chunks = 0
        collected_images = {}
        for item in datasets[e]:
            for f in range(1, item['image{}'.format(e+1)] + 1):
                with open(os.path.join(images_path, "pair_{}_product_{}_image_{}.jpg".format(counter, e+1, f)), mode='rb') as image:
                    # TODO switch names for ids
                    print(slugify(item['name{}'.format(e+1)] + '_image_{}'.format(f-1)))
                    collected_images[slugify(item['name{}'.format(e+1)] + '_image_{}'.format(f-1))] = str(base64.b64encode(image.read()), 'utf-8')


            print('Item {} uploaded'.format(counter))
            counter += 1

            if counter % 1000 == 0:
                chunks += 1
                print("images_chunk_{}".format(chunks))
                images_kvs_clients[e].set_record("images_chunk_{}".format(chunks), json.dumps(collected_images))
                collected_images = {}

        if counter % 1000 != 0:
            chunks += 1
            print("images_chunk_{}".format(chunks))
            images_kvs_clients[e].set_record("images_chunk_{}".format(chunks), json.dumps(collected_images))

save_dataset_for_executor(
    "aggregated-with-resized-images",
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "aggregated.csv"),
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "annotated_data", "aggregated_images"),
)
