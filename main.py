from apify import Actor
from actors.executor.main import perform_mapping
from apify_client import ApifyClient
import os

async def main():
    async with Actor() as actor:
        # Get the value of the actor input
        actor_input = await actor.get_input() or {}

        # TODO remove this option
        different_user_token = actor_input.get('different_user_token')

        data_client = ApifyClient(
            os.environ['APIFY_TOKEN'] if not different_user_token else different_user_token,
            api_url=os.environ['APIFY_API_BASE_URL']
        )

        # Structure of input is defined in INPUT_SCHEMA.json
        pair_dataset_id = actor_input.get('pair_dataset_id')

        client = ApifyClient(
            os.environ['APIFY_TOKEN'],
            api_url=os.environ['APIFY_API_BASE_URL']
        )
        output_dataset_id = os.environ['APIFY_DEFAULT_DATASET_ID']
        output_dataset_client = client.dataset(output_dataset_id)

        default_kvs_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])

        is_on_platform = "APIFY_IS_AT_HOME" in os.environ and os.environ["APIFY_IS_AT_HOME"] == "1"

        perform_mapping(
            pair_dataset_id,
            output_dataset_client,
            default_kvs_client,
            data_client,
            is_on_platform,
            task_id="__local__"
        )
