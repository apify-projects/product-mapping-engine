import json
import os
from apify_client import ApifyClient
from datetime import datetime, timezone


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

    now = datetime.now(timezone.utc)
    date_time = now.strftime("%Y-%m-%d")
    scrape_id = f"{date_time}-scrape-{parameters['scrapeId']}"
    scrape_info_kvs_name = f"pm-scrape-{scrape_id}-info"

    scrape_info_kvs_id = client.key_value_stores().get_or_create(name=scrape_info_kvs_name)['id']
    scrape_info_kvs_client = client.key_value_store(scrape_info_kvs_id)

    scrape_info_kvs_client.set_record("source_dataset_id", parameters["source_dataset_id"])
    scrape_info_kvs_client.set_record("product_mapping_model_name", parameters["product_mapping_model_name"])

    competitors = parameters["competitor_scraper_task_ids"]
    for competitor_name, competitor_scraper_task_id in competitors.items():
        competitor_kvs_record = {
            "finished": False
        }
        scrape_info_kvs_client.set_record(competitor_name, competitor_kvs_record)

    for competitor_name, competitor_scraper_task_id in competitors.items():
        scraper_task_client = client.task(competitor_scraper_task_id)
        scraper_task_client.start(task_input={
            "scrape_info_kvs_id": scrape_info_kvs_id,
            "competitor_name": competitor_name
        })
