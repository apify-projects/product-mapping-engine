import json
import math
import os
import pandas as pd
from apify_client import ApifyClient
import pysftp
from datetime import datetime, timezone
from price_parser import Price

DEFAULT_MAX_FILE_SIZE = 100

def fix_price(price_string):
    price = Price.fromstring(price_string)
    price_amount = price.amount_float
    return price_amount

if __name__ == '__main__':
    # Read input
    client = ApifyClient(os.environ['APIFY_TOKEN'], api_url=os.environ['APIFY_API_BASE_URL'])

    is_on_platform = "APIFY_IS_AT_HOME" in os.environ and os.environ["APIFY_IS_AT_HOME"] == "1"

    # prepare the input
    default_kvs_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])
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

    print("before loading")

    source_dataset_attributes = [
        "shopSpecificId",
        "url"
    ]

    competitor_dataset_attributes = [
        "SKU",
        "name",
        "brand",
        "price",
        "originalPrice",
        "productUrl"
    ]

    source_dataset = pd.DataFrame(client.dataset(parameters["sourceDataset"]).list_items(fields=",".join(source_dataset_attributes)).items)
    print("source loaded")
    competitor_dataset = pd.DataFrame(client.dataset(parameters["competitorDataset"]).list_items(fields=",".join(competitor_dataset_attributes)).items)
    print("competitor loaded")
    mapped_pairs_dataset = pd.DataFrame(client.dataset(parameters["mappedPairsDataset"]).list_items().items)
    print("mapped pairs loaded")

    print(source_dataset.info())
    print(competitor_dataset.info())

    print("after loading")

    preprocessed_source_dataset = source_dataset[source_dataset_attributes].rename(columns={
        "shopSpecificId": "SKUID",
        "url": "url1"
    })

    preprocessed_competitor_dataset = competitor_dataset[competitor_dataset_attributes].rename(columns={
        "SKU": "CSKUID",
        "price": "netPrice",
        "productUrl": "url2"
    })

    print("Before merge")

    final_dataset = mapped_pairs_dataset[["url1", "url2"]]\
        .merge(preprocessed_source_dataset, on="url1")\
        .merge(preprocessed_competitor_dataset, on="url2")

    print("After merge")

    competitor = parameters["competitorName"]

    final_dataset["COMPID"] = competitor
    final_dataset["date"] = parameters["scrapingDate"]
    final_dataset["competitor"] = competitor

    final_dataset = final_dataset.rename(columns={"url2": "skuUrl"}).drop(columns=["url1"])

    # TODO deal with originalPrice, netPrice and discountAmount
    final_dataset['netPrice'] = final_dataset['netPrice'].apply(fix_price)
    final_dataset['originalPrice'] = final_dataset['originalPrice'].apply(fix_price)
    fixed_original_price = []
    for index, row in final_dataset.iterrows():
        original_price = row["netPrice"] if row["originalPrice"] is None else row["originalPrice"]
        fixed_original_price.append(original_price)

    final_dataset['originalPrice'] = fixed_original_price
    final_dataset['discountAmount'] = final_dataset['originalPrice'] - final_dataset['netPrice']
    final_dataset.to_csv("final_dataset.csv")

    aggregation_kvs_info = client.key_value_stores().get_or_create(
        name=f"pm-aggregation-{parameters['scrapeId']}"
    )
    aggregation_kvs_client = client.key_value_store(aggregation_kvs_info["id"])

    aggregation_dataset_info = client.datasets().get_or_create(
        name=f"pm-aggregation-{parameters['scrapeId']}"
    )
    aggregation_dataset_client = client.dataset(aggregation_dataset_info["id"])

    aggregation_dataset_client.push_items(
        final_dataset.to_dict(orient='records')
    )

    processed_competitors_response = aggregation_kvs_client.get_record("processed_competitors")
    if processed_competitors_response:
        if parameters["upload"]:
            processed_competitors = set(processed_competitors_response["value"])
    else:
        processed_competitors = set([])

    processed_competitors.add(competitor)
    aggregation_kvs_client.set_record('processed_competitors', list(processed_competitors))
    if processed_competitors == set(parameters["competitorList"]):
        if parameters["upload"]:
            uploader_task_client = client.task(parameters["uploaderTaskId"])
            uploader_task_client.start(task_input={
                "datasets_to_upload": [aggregation_dataset_info["id"]],
            })
