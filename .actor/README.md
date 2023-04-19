# What is the AI Product Matcher and how does it work?
The AI Product Matcher actor uses a custom machine learning model we developed to solve the issue of mapping products across various e-shops,
meaning you can use it to find the same products across different e-shops for the purposes of dynamic pricing, competitor analysis and market research.
You can use it both to completely replace manual mapping of products or to make it more efficient, using different settings explained in the [Input](#markdown-header-input) section of this readme.

In order to use the matcher, you will need to already have the datasets of products that you want to match.
To get them, you can either scrape them directly on our platform (by making your own custom scraper or using one of the many
available to you [on our store](https://apify.com/store/categories/ecommerce)) or upload them to the platform using [our API](https://docs.apify.com/api/v2/) (with clients available in [Javascript](https://docs.apify.com/api/client/js/) and in [Python](https://docs.apify.com/api/client/python/)).

In case you want our help with building the scrapers or even want a complete data pipeline fully managed by us and tailored to your specific use-case
you can contact our enterprise team [here](https://apify.com/enterprise) and we will be happy to help you.
## Input
There are two ways to use the actor depending on the format your dataset is in:
1. **two separate datasets of products** -
2. **a dataset of candidate pairs** -

As mentioned in the previous section, you can either use the matcher to replace manual mapping of products, or make it more efficient,
using different settings. For this, you need to specify a setting which is generally called a **precision/recall tradeoff**, represented by
the "Precision/recall tradeoff" input in the form, or the "precision_recall" attribute in the json form of input. Since no machine learning model is flawless (even humans aren't, after all), its results can contain mistakes. This setting allows you to specify which type of mistake is more of an issue to you and thus which of them the model should try to minimize.  You can set it to one of two settings:
1. **precision** - the model will try to make sure that if it marks two products as the same, they will be the same with the highest possible degree of accuracy. While this produces more reliable product pairs, it also means that more true product pairs will be marked as different products, since the model has to be more discerning to achieve higher precision.
2. **recall** - the model will try to make sure that as many true pairs of products are found, even if it means that more

For specific numbers on the performance, check the [Expected Performance section](#markdown-header-expected-performance) of this readme.
### Sample input for two separate datasets of products
```json
{
    "dataset1_ids": [
        "Insert your dataset IDs here"
    ],
    "dataset2_ids": [
        "Insert your dataset IDs here"
    ],
    "input_mapping": {
        "eshop1": {
            "id": "url",
            "name": "name",
            "price": "price",
            "short_description": "shortDescription",
            "long_description": "longDescription",
            "specification": "specification",
            "code": [
                "codeA",
                "codeB"
            ]
        },
        "eshop2": {
            "id": "productUrl",
            "name": "name",
            "price": "price",
            "short_description": "shortDescription",
            "long_description": "longDescription",
            "specification": "specifications",
            "code": [
                "codeA",
                "codeB"
            ]
        }
    },
    "output_mapping": {
        "eshop1": {
            "id_source": "url",
            "name_source": "name"
        },
        "eshop2": {
            "id_target": "productUrl",
            "name_target": "name"
        }
    },
    "precision_recall": "precision"
}
```
### Sample input for a dataset of candidate pairs
```json
{
    "input_mapping": {
        "eshop1": {
            "id": "id1",
            "name": "name1",
            "price": "price1",
            "short_description": "short_description1",
            "long_description": "long_description1",
            "specification": "specification1",
            "code": [
                "codeA",
                "codeB"
            ]
        },
        "eshop2": {
            "id": "id2",
            "name": "name2",
            "price": "price2",
            "short_description": "short_description2",
            "long_description": "long_description2",
            "specification": "specification2",
            "code": [
                "codeA",
                "codeB"
            ]
        }
    },
    "output_mapping": {
        "eshop1": {
            "id_source": "id1",
            "name_source": "name1"
        },
        "eshop2": {
            "id_target": "id2",
            "name_target": "name2"
        }
    },
    "pair_dataset_ids": [
        "Insert your dataset IDs here"
    ],
    "precision_recall": "precision"
}

```
## Output
## Expected performance
Please note that the matcher is trained to consider different color variants of the same product to be the same product. Also note that the matcher's decisions might differ with different build versions of this actor, due to changes to the underlying machine learning model we might make in the future in order to improve its general performance. If you want to make sure the matcher's decisions don't change, pick a specific build version and set the actor to it in your account instead of the "latest" version.
## Cost
