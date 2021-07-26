import csv
import json
import os
import gzip
import requests
import imghdr
import shutil

DATASET_DIRECTORY  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
IMAGE_DIRECTORY  = os.path.join(DATASET_DIRECTORY, "images")
PREPARED_DATASET_PATH = os.path.join(DATASET_DIRECTORY, "product_pairs.csv")
DATA_FILES_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_files")
OFFERS_FILE = "offers_english.json.gz"

def is_url(potential_url):
    return "http" in potential_url or ".com" in potential_url

def download_images(pair_index, product_index, pair):
    downloaded_images = 0
    for potential_url in pair["image{}".format(product_index)].split('"'):
        if is_url(potential_url):
            try:
                response = requests.get(potential_url, timeout=5)
            except:
                continue

            if response.ok:
                if imghdr.what("", h=response.content) is None:
                    continue

                image_file_path = os.path.join(IMAGE_DIRECTORY, 'pair_{}_product_{}_image_{}'.format(pair_index, product_index, downloaded_images + 1))
                with open(image_file_path, 'wb') as image_file:
                    image_file.write(response.content)

                downloaded_images += 1

    return downloaded_images

matchesCount = 10000
nonMatchToMatchRatio = 1
nonMatchesCount = matchesCount * nonMatchToMatchRatio

dataFileUrls = {
    "pairs": { 
        "cameras_train.txt": "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/trainingSubsets/cameras_train.txt",
        "computers_train.txt": "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/trainingSubsets/computers_train.txt",
        "shoes_train.txt": "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/trainingSubsets/shoes_train.txt",
        "watches_train.txt": "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/trainingSubsets/watches_train.txt",
        "gs_watches.txt": "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/gs_watches.txt",
        "gs_shoes.txt": "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/gs_shoes.txt",
        "gs_cameras.txt": "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/gs_cameras.txt",
        "gs_computers.txt": "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/gs_computers.txt"
    },
    "offers": { 
        OFFERS_FILE: "http://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/offers_english.json.gz" 
    }
}
if not os.path.exists(DATA_FILES_DIRECTORY):
    os.mkdir(DATA_FILES_DIRECTORY)

if not os.path.exists(DATASET_DIRECTORY):
    os.mkdir(DATASET_DIRECTORY)

if os.path.exists(IMAGE_DIRECTORY):
    shutil.rmtree(IMAGE_DIRECTORY)
os.mkdir(IMAGE_DIRECTORY)

for fileType in dataFileUrls:
    for fileName in dataFileUrls[fileType]:
        filePath = os.path.join(DATA_FILES_DIRECTORY, fileName)
        if not os.path.isfile(filePath):
            url = dataFileUrls[fileType][fileName]
            response = requests.get(url, allow_redirects=True)
            open(filePath, 'wb').write(response.content)


with gzip.open(os.path.join(DATA_FILES_DIRECTORY, OFFERS_FILE), 'r') as offersFile:
    with open(PREPARED_DATASET_PATH, "w") as preprocessedDatasetFile:
        outputWriter = csv.DictWriter(preprocessedDatasetFile, delimiter=',', quotechar='"', fieldnames=[ "id1", "name1", "image1", "id2", "name2", "image2", "match" ])
        outputWriter.writeheader()

        offers = {}
        counter = namedCounter = imagedCounter = bothCounter = neitherCounter = 0
        while True:
            line = offersFile.readline()

            if not line:
                break

            offer = json.loads(line)
            schemaProperties = offer["schema.org_properties"]
            newOffer = {}

            for schemaProperty in schemaProperties:
                for key in schemaProperty:
                    if key == "/name":
                        if "name" in newOffer:
                            print("Multiple names!")

                        newOffer["name"] = schemaProperty[key]
                        namedCounter += 1
                    elif key == "/image":
                        if "image" in newOffer:
                            print("Multiple images!")

                        image_url = schemaProperty[key]
                        if is_url(image_url):
                            newOffer["image"] = image_url
                            imagedCounter += 1
            
            newOffer["url"] = offer["url"]
            newOffer["id"] = offer["nodeID"] + " " + offer["url"]

            if "name" in newOffer:
                offers[newOffer["id"]] = newOffer

                if "image" in newOffer:
                    bothCounter += 1

            if "name" not in newOffer and "image" not in newOffer:
                neitherCounter += 1

            counter += 1
            if counter % 1000000 == 0:
                print("Row: {}".format(counter))

        print("Overall: {}".format(counter))
        print("Name: {}".format(namedCounter))
        print("Image: {}".format(imagedCounter))
        print("Both: {}".format(bothCounter))
        print("Neither: {}".format(neitherCounter))

        pairsChecked = 0
        matchingPairs = 0
        nonMatchingPairs = 0
        pairs = []
        for fileName in dataFileUrls["pairs"]:
            with open(os.path.join(DATA_FILES_DIRECTORY, fileName), "r") as pairsFile:
                while True:
                    line = pairsFile.readline()
                    line = line.rstrip("\n")

                    if not line:
                        break

                    pairInfo = line.split("#####")
                    if pairInfo[0] in offers and pairInfo[1] in offers:
                        pairsChecked += 1
                        offer1 = offers[pairInfo[0]]
                        offer2 = offers[pairInfo[1]]
                        if "image" in offer1 and "image" in offer2:
                            pair = {
                                "id1": offer1["id"],
                                "name1": offer1["name"],
                                "image1": offer1["image"] if "image" in offer1 else "",
                                "id2": offer2["id"],
                                "name2": offer2["name"],
                                "image2": offer2["image"] if "image" in offer2 else "",
                                "match": pairInfo[2]
                            }

                            pair["image1"] = download_images(len(pairs), 1, pair)
                            pair["image2"] = download_images(len(pairs), 2, pair)

                            if pair["match"] == "0":
                                nonMatchingPairs += 1
                            else:
                                matchingPairs += 1

                            pairs.append(pair)

        print("Matching pairs: {}".format(matchingPairs))
        print("Non matching pairs: {}".format(nonMatchingPairs))
        print(pairsChecked)

        outputWriter.writerows(pairs)
        preprocessedDatasetFile.flush()
        



