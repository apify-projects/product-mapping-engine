import json
import math
import os
import pandas as pd
from apify_client import ApifyClient
import pysftp
from datetime import datetime, timezone

DEFAULT_MAX_FILE_SIZE = 100

if __name__ == '__main__':
    # Read input
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])
    default_kvs_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])

    is_on_platform = "APIFY_IS_AT_HOME" in os.environ and os.environ["APIFY_IS_AT_HOME"] == "1"

    if not is_on_platform:
        with open("input.json", "r") as input_file:
            # Set default input if not on platform
            default_kvs_client.set_record(
                'INPUT',
                json.load(input_file)
            )

    parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
    if not is_on_platform:
        print('Actor input:')
        print(json.dumps(parameters, indent=2))

    with open("login.json", "r") as login_file:
        parameters = {**json.load(login_file), **parameters}

    print(f"Uploading dataset {parameters['datasets_to_upload']} to the SFTP server")

    now = datetime.now(timezone.utc)
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"competitors_db_{date_time}"
    complete_file_name = f"{file_name}.csv"

    datasets_to_upload = []
    for dataset_id in parameters["datasets_to_upload"]:
        dataset_to_upload_client = client.dataset(dataset_id)
        dataset_to_upload = pd.DataFrame(dataset_to_upload_client.list_items().items)
        print(f"Partial dataset of shape {dataset_to_upload.shape}")
        datasets_to_upload.append(dataset_to_upload)
    complete_dataset_to_upload = pd.concat(datasets_to_upload, ignore_index=True)

    complete_dataset_to_upload.to_csv(complete_file_name, encoding="utf-8", sep=";", index=False, header=True)

    max_file_size = float(parameters['max_file_size']) if 'max_file_size' in parameters else DEFAULT_MAX_FILE_SIZE
    file_stats = os.stat(complete_file_name)
    print(file_stats)
    print(f'File Size in Bytes is {file_stats.st_size}')
    complete_file_size_megabytes = file_stats.st_size / (1024 * 1024)
    print(f'File Size in MegaBytes is {complete_file_size_megabytes}')

    cnopts = pysftp.CnOpts()
    cnopts.hostkeys = None

    conn = pysftp.Connection(
        host=parameters["host"],
        port=int(parameters["port"]),
        username=parameters["username"],
        password=parameters["password"],
        cnopts=cnopts
    )

    print("Connection successfully established")

    print(f"Total amount of items: {complete_dataset_to_upload.shape[0]}")

    if complete_file_size_megabytes > max_file_size:
        desired_file_amount = math.ceil(complete_file_size_megabytes / max_file_size)
        desired_file_row_count = math.ceil(complete_dataset_to_upload.shape[0] / desired_file_amount)
        for e in range(desired_file_amount):
            dataset_chunk_to_upload = complete_dataset_to_upload[e * desired_file_row_count: (e + 1) * desired_file_row_count]
            print(f"Chunk {e+1} amount of items: {dataset_chunk_to_upload.shape[0]}")
            chunk_file_name = f"{file_name}_part{e+1}.csv"
            dataset_chunk_to_upload.to_csv(chunk_file_name, encoding="utf-8", sep=";", index=False, header=True)

            conn.put(chunk_file_name, f"{parameters['path']}/{chunk_file_name}")
    else:
        conn.put(complete_file_name, f"{parameters['path']}/{complete_file_name}")

    print("Dataset successfully uploaded")
    conn.close()
