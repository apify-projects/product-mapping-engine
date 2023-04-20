import json
import math
import os
import time
import pandas as pd
from apify_client import ApifyClient
import pysftp
from datetime import datetime, timezone
from price_parser import Price

def calculate_additional_filters(product):
    if product["brand"].lower() == "apple":
        if product["productSpecificId_source"] and product["productSpecificId_competitor"] and product["productSpecificId_source"] != product["productSpecificId_competitor"]:
            return False

    return True

def extract_price(price_object):
    return price_object["formattedPrice"] if price_object and type(price_object) == dict else ""

def extract_original_price(price_object):
    return price_object["formattedPriceUnmodified"] if price_object and type(price_object) == dict else ""

def fix_amazon_sku(product):
    if not product["SKU"]:
        return product["productUrl"].split("amazon.sa/dp/")[1].split("/")[0]

    return product["SKU"]

def fix_price(price_string):
    price = Price.fromstring(price_string)
    price_amount = price.amount_float
    return price_amount

def calculate_discount_amount(product):
    if product["netPrice"]:
        if not product["originalPrice"]:
            return 0
        else:
            return round(product["originalPrice"] - product["netPrice"], 2)

    return None

discountAttributes = [
    "bundle",
    "freeInstallation",
    "freeItem",
    "couponDiscount",
    "bankPromo"
]

def getDiscountTypeOrName(row, whatToGet):
    typeOrName = ""
    for attribute in discountAttributes:
        if row[attribute]:
            if typeOrName != "":
                typeOrName += ","

            typeOrName += '"'
            if whatToGet == "type":
                typeOrName += attribute
            else:
                value = row[attribute]
                if not isinstance(value, str):
                    value = json.dumps(value)

                typeOrName += value
            typeOrName += '"'

    return typeOrName


def getDiscountType(row):
    return getDiscountTypeOrName(row, "type")


def getDiscountName(row):
    return getDiscountTypeOrName(row, "name")


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

    scrape_info_kvs_id = parameters["scrape_info_kvs_id"]
    scrape_info_kvs_client = client.key_value_store(scrape_info_kvs_id)
    competitor_name = parameters["competitor_name"]

    competitor_record = scrape_info_kvs_client.get_record(competitor_name)["value"]
    if competitor_record["finished"] != True:
        source_dataset_id = scrape_info_kvs_client.get_record("source_dataset_id")["value"]

        source_dataset_attributes = [
            "shopSpecificId",
            "productSpecificId",
            "url"
        ]

        competitor_dataset_attributes = [
            "SKU",
            "name",
            "brand",
            "price",
            "originalPrice",
            "productUrl",
            "productSpecificId"
        ]

        competitor_dataset_attributes_to_fetch = competitor_dataset_attributes + discountAttributes + ["sku"]

        source_dataset = pd.DataFrame(client.dataset(source_dataset_id).list_items(fields=",".join(source_dataset_attributes)).items)

        competitor_dataset = pd.DataFrame(client.dataset(competitor_record["scraped_dataset_id"]).list_items(fields=",".join(competitor_dataset_attributes_to_fetch)).items)

        mapped_pairs_dataset = pd.DataFrame(client.dataset(competitor_record["mapped_dataset_id"]).list_items().items)

        aggregation_kvs_info = client.key_value_stores().get_or_create(
            name=f"pm-aggregation-{parameters['scrape_info_kvs_id']}"
        )
        aggregation_kvs_client = client.key_value_store(aggregation_kvs_info["id"])

        aggregation_dataset_info = client.datasets().get_or_create(
            name=f"pm-aggregation-{parameters['scrape_info_kvs_id']}"
        )
        aggregation_dataset_client = client.dataset(aggregation_dataset_info["id"])

        scrape_info_kvs_client.set_record("aggregated_dataset_id", aggregation_dataset_info["id"])

        if not mapped_pairs_dataset.empty:
            preprocessed_source_dataset = source_dataset[source_dataset_attributes].rename(columns={
                "shopSpecificId": "SKUID",
                "url": "url1"
            })

            print(competitor_dataset.info())
            discountAttributes = competitor_dataset.columns.intersection(set(discountAttributes))

            competitor_dataset["discountType"] = competitor_dataset[discountAttributes].apply(getDiscountType, axis=1)
            competitor_dataset["discountName"] = competitor_dataset[discountAttributes].apply(getDiscountName, axis=1)

            # xcite has a slightly different scraper output
            if competitor_name == "xcite":
                competitor_dataset["SKU"] = competitor_dataset["sku"]
                competitor_dataset["originalPrice"] = competitor_dataset["price"].apply(extract_original_price)
                competitor_dataset["price"] = competitor_dataset["price"].apply(extract_price)

            if competitor_name == "amazon":
                competitor_dataset["SKU"] = competitor_dataset.apply(fix_amazon_sku, axis=1)

            preprocessed_competitor_dataset = competitor_dataset[competitor_dataset_attributes + ["discountType", "discountName"]].rename(columns={
                "SKU": "CSKUID",
                "price": "netPrice",
                "productUrl": "url2"
            })

            final_dataset = mapped_pairs_dataset[["url1", "url2"]]\
                .merge(preprocessed_source_dataset, on="url1")\
                .merge(preprocessed_competitor_dataset, on="url2", suffixes=("_source", "_competitor"))

            final_dataset = final_dataset.drop_duplicates(subset=["url1", "url2"]).fillna("")

            # Apple is problematic, so a filter based on the codes is needed
            print(final_dataset.info())
            final_dataset['keep'] = final_dataset.apply(calculate_additional_filters, axis=1)
            final_dataset = final_dataset[final_dataset['keep'] == True]
            final_dataset = final_dataset.drop(columns=["keep", "productSpecificId_source", "productSpecificId_competitor"])

            now = datetime.now(timezone.utc)
            date = now.strftime("%Y_%m_%d")
            final_dataset["COMPID"] = competitor_name
            final_dataset["date"] = date
            final_dataset["competitor"] = competitor_name

            final_dataset = final_dataset.rename(columns={"url2": "skuUrl"}).drop(columns=["url1"])

            # TODO deal with originalPrice, netPrice and discountAmount
            final_dataset['netPrice'] = final_dataset['netPrice'].apply(fix_price)
            final_dataset['originalPrice'] = final_dataset['originalPrice'].apply(fix_price)
            fixed_original_price = []
            for index, row in final_dataset.iterrows():
                original_price = row["netPrice"] if not row["originalPrice"] or math.isnan(row["originalPrice"]) else row["originalPrice"]
                fixed_original_price.append(original_price)

            final_dataset['originalPrice'] = fixed_original_price
            final_dataset['discountAmount'] = final_dataset[['originalPrice', 'netPrice']].apply(calculate_discount_amount, axis=1)

            final_dataset = final_dataset.fillna('')

            print(final_dataset.info())

            final_dataset.to_csv("final_dataset.csv")

            aggregation_dataset_client.push_items(
                final_dataset.to_dict(orient='records')
            )

        competitor_record["finished"] = True
        scrape_info_kvs_client.set_record(competitor_name, competitor_record)

        # TODO fix race condition and remove this quick hack
        time.sleep(15)

        everything_aggregated = True
        competitors_list = scrape_info_kvs_client.get_record("competitors_list")["value"]
        for checked_competitor_name in competitors_list:
            checked_competitor_record = scrape_info_kvs_client.get_record(checked_competitor_name)["value"]
            if not checked_competitor_record["finished"]:
                everything_aggregated = False
                break

        if everything_aggregated and parameters["upload"]:
            # Check if another run hasn't started the upload already
            upload_triggered_record = scrape_info_kvs_client.get_record("upload_triggered")
            if not upload_triggered_record or upload_triggered_record["value"] is False:
                scrape_info_kvs_client.set_record("upload_triggered", True)

                for uploader_task_id in parameters["uploader_task_ids"]:
                    uploader_task_client = client.task(uploader_task_id)
                    uploader_task_client.start(task_input={
                        "datasets_to_upload": [aggregation_dataset_info["id"]],
                    })
