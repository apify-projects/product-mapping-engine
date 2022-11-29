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
    date_time = now.strftime("%Y_%m_%d")
    scrape_id = f"{date_time}_scrape_{parameters['scrapeId']}"

    taskIds = parameters["taskIds"]
    for taskId in taskIds:
        scraper_task_client = client.task(taskId)
        scraper_task_client.start(task_input={
            "scrape_id": scrape_id
        })
