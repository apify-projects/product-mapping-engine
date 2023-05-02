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

        if 'ACTOR_MAX_PAID_DATASET_ITEMS' in os.environ and os.environ['ACTOR_MAX_PAID_DATASET_ITEMS'] != "" and os.environ['ACTOR_MAX_PAID_DATASET_ITEMS'] != 0:
            max_items_to_process = os.environ['ACTOR_MAX_PAID_DATASET_ITEMS']
            if max_items_to_process < 1:
                raise Exception("When specifying the maximum number of results you want to get, it needs to be 1 or more")
        else:
            max_items_to_process = None

        data_client = ApifyClient(
            os.environ['APIFY_TOKEN'] if not different_user_token else different_user_token,
            api_url=os.environ['APIFY_API_BASE_URL']
        )

        client = ApifyClient(
            os.environ['APIFY_TOKEN'],
            api_url=os.environ['APIFY_API_BASE_URL']
        )
        output_dataset_id = os.environ['APIFY_DEFAULT_DATASET_ID']
        output_dataset_client = client.dataset(output_dataset_id)

        default_kvs_client = client.key_value_store(os.environ['APIFY_DEFAULT_KEY_VALUE_STORE_ID'])

        # TODO change if needed
        #is_on_platform = "APIFY_IS_AT_HOME" in os.environ and os.environ["APIFY_IS_AT_HOME"] == "1"
        is_on_platform = True

        parameters = {}

        pair_dataset_ids = actor_input.get("pair_dataset_ids")
        if pair_dataset_ids:
            parameters["pair_dataset_ids"] = pair_dataset_ids
        else:
            dataset1_ids = actor_input.get("dataset1_ids")
            dataset2_ids = actor_input.get("dataset2_ids")
            if dataset1_ids and dataset2_ids:
                parameters["dataset1_ids"] = dataset1_ids
                parameters["dataset2_ids"] = dataset2_ids
            else:
                raise Exception("You need to provide either pair_dataset_ids or dataset1_ids and dataset2_ids")

        parameters["input_mapping"] = actor_input["input_mapping"]
        if actor_input.get("output_mapping"):
            parameters["output_mapping"] = actor_input["output_mapping"]

        perform_mapping(
            parameters,
            output_dataset_client,
            default_kvs_client,
            data_client,
            is_on_platform,
            task_id="__local__##sep##" + actor_input.get("precision_recall"),
            return_all_considered_pairs=True,
            max_items_to_process=max_items_to_process
        )
