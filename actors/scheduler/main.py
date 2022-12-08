import json
import math
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

    parallelization_factor = parameters["parallelization_factor"] if "parallelization_factor" in parameters else 1
    all_queries = parameters["queries"]
    query_chunks = []
    if parallelization_factor == 1:
        query_chunks.append(all_queries)
    else:
        chunk_size = math.ceil(len(all_queries) / parallelization_factor)
        for e in range(parallelization_factor):
            query_chunks.append(queries[e * chunk_size: (e+1) * chunk_size])

    competitors_list = []
    competitors = parameters["competitor_scraper_task_ids"]
    competitor_kvs_records = {}
    for competitor_name, competitor_scraper_task_id in competitors.items():
        competitor_kvs_record = {
            "finished": False
        }
        competitor_kvs_records[competitor_name] = competitor_kvs_record
        scrape_info_kvs_client.set_record(competitor_name, competitor_kvs_record)
        competitors_list.append(competitor_name)

    scrape_info_kvs_client.set_record("competitors_list", competitors_list)

    for competitor_name, competitor_scraper_task_id in competitors.items():
        competitor_kvs_record = competitor_kvs_records[competitor_name]
        competitor_kvs_record["scraper_run_ids"] = []
        scraper_task_client = client.task(competitor_scraper_task_id)

        for e in range(parallelization_factor):
            run_info = scraper_task_client.start(task_input={
                "scrape_info_kvs_id": scrape_info_kvs_id,
                "competitor_name": competitor_name,
                "queries": query_chunks[e]
            })
            competitor_kvs_record["scraper_run_ids"].append(run_info["id"])

        scrape_info_kvs_client.set_record(competitor_name, competitor_kvs_record)

