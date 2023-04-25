## What is the AI Product Matcher and how does it work?
The AI Product Matcher Actor uses a custom machine learning model we’ve developed to solve the issue of mapping products across various e-shops. You can use this tool to find the same products across different e-shops for the purposes of dynamic pricing, competitor analysis, and market research. You can use it both to completely replace the manual mapping of products or to make it more efficient, using different settings explained in the [Input](#how-should-the-input-look) section of this readme.

In order to use the matcher, you will need to already have the datasets of products that you want to match. To get them, you can either scrape them directly on our platform (by making your own custom scraper or using one of the many available to you [in our Store](https://apify.com/store/categories/ecommerce)) or upload them to the platform using [our API](https://docs.apify.com/api/v2/) (with clients available in [Javascript](https://docs.apify.com/api/client/js/) and in [Python](https://docs.apify.com/api/client/python/)). Keep in mind that the matcher currently only works with English data.

In case you want our help with building the scrapers or even want a complete data pipeline fully managed by us and tailored to your specific use case, you can [contact our enterprise team](https://apify.com/enterprise) and we will be happy to help you.

## **How should the input look?**

This section describes how to prepare the input for the matcher Actor. At the end of it, you can find examples of what the filled input could look like in practice.

### **How to specify input datasets?**

There are two ways to use the Actor depending on the format your dataset is in:

1. **a dataset containing candidate pairs** - You might already have a dataset in which each row contains information about the two products that you want the matcher to compare. In this case, you put the ids of these pair datasets in the *pair_dataset_ids* input (you can put multiple ids there in case you have multiple datasets that you want to be processed at once). The matcher will check each row and output a decision on whether the products in each row are the same.
2. **two separate datasets of products** - In other use cases, you might have two separate datasets of products, each containing products from one e-shop. In this case, you put the ids of these datasets into the inputs *dataset1_ids* (for datasets of products from the first e-shop) and *dataset2_ids* (for datasets of products from the second e-shop). The matcher will then look at all the possible pairings of products between the dataset and output a decision about each of them.

### **How to specify the input dataset format?**

The next part of the Actor's input is telling the matcher what format the dataset you are giving to it is in, represented by the *input_mapping* Actor input. It should be a JSON object containing two attributes: *eshop1* and *eshop2*. Each one describes under what attributes can the Actor find the necessary data for the specific e-shop. Each of these two attributes should contain another object that looks like this, for example:

```json
{
  "id": "productUrl",
  "name": "productName",
  "price": "currentPrice",
  "short_description": "short_description",
  "long_description": "long_description",
  "specification": "specification",
  "code": [
    "SKU",
    "ASIN"
  ]
}

```

Each attribute of this object (such as *name*) specifies where each necessary attribute of the product can be found in the product dataset (continuing with the same example, it says that the product's name can be found in the attribute *productName* of the dataset which you gave to the matcher). These necessary attributes are as follows:

1. **id** - unique identifier of the product. Has to be provided for the matcher to function, but isn't used as input of the machine learning model, so it can be anything convenient, such as the URL.
2. **name** - the product's name.
3. **price** - the current selling price of the product. Can be empty if not available. Can also contain the currency symbol (such as "$50" instead of just the number 50). However, the matcher currently disregards the currency so if you want to compare products in different currencies, you need to perform the currency conversion yourself.
4. **short_description** - most e-shops provide short (several lines at most) descriptions of the product close to the product name, price, and image. It usually describes the most important features or specifies some of the product's parameters (e.g. "32GB RAM, 500GB Hard Drive, Intel Core i3" for a laptop).
5. **long_description** - most e-shops also provide a longer description of the product, often including text provided by the manufacturer.
6. **specification** - this should be a JSON array containing within it the product's parameters (such as weight, dimensions, components in case of electronics, color, etc.) often provided in a big table on the product's page.
7. **code** - this attribute is special in that it allows you to specify more than one input dataset attribute, as you can see in the example above. The attributes specified should contain codes of the product if these are available, such as ASIN, SKU, EAN, etc. The more of them you can get, the better.

**You don't always have to provide all of these, some of them might not even be present on the specific e-shops you wish to use the matcher for. However, not providing them might result in degraded accuracy of the matcher. For more details, see the [section about performance](#how-accurate-is-the-matcher).**

### **How to specify the output dataset format?**

After specifying the format of the input dataset, you should specify which attributes should be included in the output dataset of the matcher. This can be done using the *output_mapping* Actor input, which is very similar to *input_mapping*, as you can see from this example:

```json
{
  "eshop1": {
    "id_source": "productUrl",
    "name_source": "productName"
  },
  "eshop2": {
    "id_target": "EAN",
    "name_target": "productName"
  }
}

```

Same as before, you specify the attributes separately for each e-shop. Each line then specifies what will the attribute be called in the output dataset (e.g. *id_source*) and which attribute it was in the corresponding input dataset (e.g. *productUrl*). Apart from these, the output dataset will also contain two more attributes for each considered product pair:

1. *predicted_match* - will be 1 in case the matcher thinks the two products in the considered pair are the same, 0 if not.
2. *predicted_scores* - specifies how much the matcher thinks the two products are the same. Will be close to 1 for those that it considers to definitely be the same, and close to 0 for those it considers to definitely not be the same. This output attribute will be useful if you decide to apply your own threshold. For example, if you need to be as sure as possible, you could only take pairs with very high *predicted_scores*.

### **Precision/recall tradeoff**

As mentioned in the previous section, you can either use the matcher to replace the manual mapping of products, or make it more efficient by using different settings. For this, you need to specify a setting which is generally called a **precision/recall tradeoff**, represented by the "Precision/recall tradeoff" input in the form, or the "precision_recall" attribute in the JSON form of input. Since no machine learning model is flawless (even humans aren't, after all), its results can contain mistakes. This setting allows you to specify which type of mistake is more of an issue to you and thus which of them the model should try to minimize. You can set it to one of two settings:

1. **precision** - the model will try to make sure that if it marks two products as the same, they will be the same with the highest possible degree of accuracy. While this produces more reliable product pairs, it also means that more true product pairs will be marked as different products, since the model has to be more discerning to achieve higher precision.
2. **recall** - the model will try to make sure that as many pairs of products where both products are the same are found as possible, even if it means that there will also be more pairs where the model made a mistake and the two products aren't actually the same.

For specific numbers on the performance, check the [expected performance section](#how-accurate-is-the-matcher) of this readme.

### **Sample input for a dataset of candidate pairs**

```json
{
    "pair_dataset_ids": [
        "Insert your dataset IDs here"
    ],
    "input_mapping": {
        "eshop1": {
            "id": "id1",
            "name": "name1",
            "price": "price1",
            "short_description": "short_description1",
            "long_description": "long_description1",
            "specification": "specification1",
            "code": [
                "SKU",
                "ASIN"
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
                "EAN",
                "ASIN"
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
    "precision_recall": "precision"
}

```

### **Sample input for two separate datasets of products**

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
                "SKU",
                "ASIN"
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
                "EAN",
                "ASIN"
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

## **What will the results look like and where can you find them?**

The results will be stored in the default dataset of the Actor run, accessible through the run page in Apify Console. You can download them both manually and through an API in a variety of formats including JSON, CSV, and Excel.

For a detailed description of the format of the results, see the [output format subsection](#how-to-specify-the-output-dataset-format) of the previous section.

## **How accurate is the matcher?**

We have striven to make the matcher as accurate as we can, gathering thousands of manually annotated pairs of products from various categories and using them to train the model. As is the common practice, we’ve also prepared a separate dataset of product pairs that the model never saw during the training process and then estimated the model's performance using these pairs. The precise results of course depend on the precision/recall tradeoff model setting:

1. With the AI model trained for **precision**, we measured precision of 95% (meaning that when the matcher said that two products from different e-shops were the same product, it was true in 95% of cases) with recall being 60% (meaning that the matcher found 60% of the pairs containing the same product that could be found).
2. With the AI model trained for **recall**, we measured recall of 95% with precision being 55%.

**Of course, since you will be providing your own data which will probably be coming from different e-shops than we tested, the performance you see might differ, even though we tried to make the matcher as general as possible. For that reason, we recommend you do your own investigation of the accuracy of the matcher's results before you use the results for your use case.** Future versions of this Actor will give you the option to use your data to train the matcher in order to alleviate this issue.

Please also note that:

1. The matcher's performance is heavily dependent on what data you provide to it. Missing any of the expected data attributes might result in degraded performance, especially in case of the name, price, and code attributes.
2. The matcher is trained to consider different color variants of the same product to be the same product (e.g. different color variants of the same smartphone will be considered the same product).
3. The matcher's decisions might differ with different build versions of this Actor, due to changes to the underlying machine learning model we might make in the future in order to improve its general performance. If you want to make sure the matcher's decisions don't change, pick a specific build version and set the Actor to it in your account instead of the "latest" version.

## **How much will it cost?**

This Actor is paid using the pay-per-result model, meaning you will pay a small amount for each row in the output dataset (you can find the amount per 1000 results at the top right of this Actor's detail page in Store). In this case, a result is a decision about whether a specific pair of products is the same or not. This means that the amount you pay also depends on the type of input you provided:

1. **a dataset of candidate pairs** - this case is very simple - the number of results is the same as the number of rows in the input datasets.
2. **two separate datasets of products** - this one is more complicated because the Actor has to try all the possible pairings of products.

In case you want to limit the number of potential results (and thus limit how much you pay at maximum), you can set the maximum number of results in the Actor's options.
