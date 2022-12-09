import json
import os
from math import ceil
import time

import pandas as pd
from apify_client import ApifyClient
from price_parser import Price

from product_mapping_engine.scripts.dataset_handler.dataset_upload.dataset_preprocessor import download_images, upload_images_to_kvs

def fix_price(price_string):
    price = Price.fromstring(price_string)
    price_amount = price.amount_float
    return price_amount

def fix_specification(specification):
    fixed_specification = []
    if specification:
        for parameter, value in specification.items():
            fixed_specification.append({
                'key': parameter,
                'value': value
            })

    return fixed_specification

def squishAttributesIntoAListAttribute(row):
    result = []
    for value in row:
        if value != "":
            result.append(value)

    return result

if __name__ == '__main__':
    # Read input
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])
    default_kvs_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])

    is_on_platform = "APIFY_IS_AT_HOME" in os.environ and os.environ["APIFY_IS_AT_HOME"] == "1"

    if not is_on_platform:
        # Set default input if not on platform
        with open("input.json", "r") as input_file:
            # Set default input if not on platform
            default_kvs_client.set_record(
                'INPUT',
                json.load(input_file)
            )

    parameters = default_kvs_client.get_record(os.environ['APIFY_INPUT_KEY'])['value']
    print('Actor input:')
    print(json.dumps(parameters, indent=2))

    competitor_scrape_run_client = client.run(parameters["competitor_scrape_run_id"])
    competitor_scrape_run_info = competitor_scrape_run_client.get()

    scraped_dataset_id = competitor_scrape_run_info["defaultDatasetId"]
    parameters["target"]["dataset_id"] = scraped_dataset_id

    scraper_kvs_client = client.key_value_store(competitor_scrape_run_info["defaultKeyValueStoreId"])
    scraper_input = scraper_kvs_client.get_record("INPUT")["value"]
    trigger_preprocessing = scraper_input["trigger_preprocessing"]
    if trigger_preprocessing:
        scrape_info_kvs_id = scraper_input["scrape_info_kvs_id"]
        competitor_name = scraper_input["competitor_name"]
        scrape_info_kvs_client = client.key_value_store(scrape_info_kvs_id)
        competitor_record = scrape_info_kvs_client.get_record(competitor_name)["value"]

        ready_for_preprocessing = False
        while not ready_for_preprocessing:
            everything_ready = True
            if "scraper_run_ids" in competitor_record:
                for scraper_run_id in competitor_record["scraper_run_ids"]:
                    scraper_run_client = client.run(scraper_run_id)
                    scraper_run_info = scraper_run_client.get()
                    if scraper_run_info["status"] != "SUCCEEDED":
                        everything_ready = False
                        break
            else:
                everything_ready = False

            if everything_ready:
                ready_for_preprocessing = True
                break
            else:
                time.sleep(30)
                competitor_record = scrape_info_kvs_client.get_record(competitor_name)["value"]

        scraper_run_ids = competitor_record["scraper_run_ids"]
        for scraper_run_id in scraper_run_ids:
            scraper_run_client = client.run(scraper_run_id)
            scraper_run_info = scraper_run_client.get()
            parameters["target"]["dataset_id"] = scraper_run_info["defaultDatasetId"]

            no_scraped_items = False
            source_dataset = {}
            target_dataset = pd.DataFrame()
            target_scraped_data = pd.DataFrame()
            for dataset in ['source', 'target']:
                dataset_parameters = parameters[dataset]
                # Read the dataset
                scraped_dataset_id = dataset_parameters['dataset_id']
                scraped_dataset_client = client.dataset(scraped_dataset_id)
                scraped_data = scraped_dataset_client.list_items().items
                scraped_dataset = pd.DataFrame(scraped_data).fillna("").drop_duplicates() # PM system expects empty strings, not nulls
                if dataset == "target":
                    target_scraped_data = scraped_data

                    if scraped_dataset.empty:
                        no_scraped_items = True
                        break

                # Change attributes of the dataset to the ones needed by the Product Mapping system
                partial_product_mapping_dataset = pd.DataFrame()
                attribute_names = dataset_parameters['attribute_names']
                for product_mapping_attribute, scraped_attribute in attribute_names.items():
                    if type(scraped_attribute) is list:
                        partial_product_mapping_dataset[product_mapping_attribute] = scraped_dataset[scraped_attribute].apply(squishAttributesIntoAListAttribute, axis=1)
                    else:
                        partial_product_mapping_dataset[product_mapping_attribute] = scraped_dataset[scraped_attribute]

                if dataset_parameters['specification_format'] == 'parameter-value':
                    partial_product_mapping_dataset['specification'] = partial_product_mapping_dataset['specification'].apply(fix_specification)

                if dataset_parameters['fix_prices']:
                    partial_product_mapping_dataset['price'] = partial_product_mapping_dataset['price'].apply(fix_price)

                partial_product_mapping_dataset['price'] = partial_product_mapping_dataset['price'].fillna("")

                # Download images
                '''
                product_mapping_dataset['image'] = download_images(product_mapping_dataset)
            
                target_kvs_name = 'PM-Prepro-Images-' + parameters['task_id'] + '-' + scraped_dataset_id
                target_kvs_id = client.key_value_stores().get_or_create(name=target_kvs_name)['id']
                target_kvs_client = client.key_value_store(target_kvs_id)
            
                upload_images_to_kvs(product_mapping_dataset, target_kvs_client)
                '''

                if dataset == 'target':
                    target_dataset = partial_product_mapping_dataset
                else:
                    connecting_attribute = parameters[dataset]['connecting_attribute']
                    for index, product_row in partial_product_mapping_dataset.iterrows():
                        product = dict(product_row)
                        source_dataset[product[connecting_attribute]] = product

            if not no_scraped_items:
                target_connecting_attribute = parameters['target']['source_connecting_attribute']
                dataset_to_upload = []
                for index, target_product_row in target_dataset.iterrows():
                    target_product = dict(target_product_row)
                    corresponding_source_product = source_dataset[target_product[target_connecting_attribute]]
                    del target_product[target_connecting_attribute]

                    corresponding_source_product = {f"{key}1": val for key, val in corresponding_source_product.items()}
                    target_product = {f"{key}2": val for key, val in target_product.items()}

                    candidate_pair = {**corresponding_source_product, **target_product}
                    dataset_to_upload.append(candidate_pair)

                '''
                # Save to file for debugging
                product_mapping_dataset.to_json('xcite_extra.json', orient='records')
                '''
                # Upload the preprocessed dataset
                preprocessed_dataset_name = 'PM-Prepro-Data-' + scrape_info_kvs_id + '-' + competitor_name
                preprocessed_dataset_id = client.datasets().get_or_create(name=preprocessed_dataset_name)['id']
                preprocessed_dataset_client = client.dataset(preprocessed_dataset_id)

                for e in range(ceil(len(dataset_to_upload) / 500)):
                    chunk_to_upload = dataset_to_upload[e * 500: (e+1) * 500]
                    preprocessed_dataset_client.push_items(chunk_to_upload)

                print(f"{len(dataset_to_upload)} items uploaded to dataset {preprocessed_dataset_id}")

                # Upload the scraped data so that it can later be used by aggregator
                aggregated_scraped_dataset_name = 'PM-Scraped-Data-' + scrape_info_kvs_id + '-' + competitor_name
                aggregated_scraped_dataset_id = client.datasets().get_or_create(name=aggregated_scraped_dataset_name)['id']
                aggregated_scraped_dataset_client = client.dataset(aggregated_scraped_dataset_id)

                for e in range(ceil(len(target_scraped_data) / 500)):
                    chunk_to_upload = target_scraped_data[e * 500: (e + 1) * 500]
                    aggregated_scraped_dataset_client.push_items(chunk_to_upload)

                competitor_record["preprocessed_dataset_id"] = preprocessed_dataset_id
                competitor_record["scraped_dataset_id"] = aggregated_scraped_dataset_id
                scrape_info_kvs_client.set_record(competitor_name, competitor_record)

        if parameters["run_executor"]:
            executor_task_client = client.task(parameters["executor_task_id"])
            executor_task_client.start(task_input={
                "scrape_info_kvs_id": scrape_info_kvs_id,
                "competitor_name": competitor_name
            })
    else:
        print("No preprocessing required yet, ending the preprocessor")