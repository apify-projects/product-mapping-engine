import os
import pandas as pd
from apify_client import ApifyClient

client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])

if __name__ == '__main__':
    data_to_upload = pd.read_json("extraToUpload.json")
    dataset_collection_client = client.datasets()

    apify_dataset_info = dataset_collection_client.get_or_create(name="raw-extra-data")
    apify_dataset_client = client.dataset(apify_dataset_info['id'])
    apify_dataset_client.push_items(
        data_to_upload.to_dict(orient='records')
    )
