# Product Mapping Project
Project for automatic mapping among products from different websites.

## Folder *data*
- folder *preprocessed*
  - folder *dataset_name*
    - folder *names*
    - folder *images*
      - folder *cropped*
      - folder *cropped_masked*
      - folder *cropped_resized*
      - folder *hashes*
    - scores.txt
 

- folder *source*
  - folder *dataset_name*
    - folder *names*
    - folder *images*
      - folder *cropped*
      - folder *cropped_masked*
      - folder *cropped_resized*
      - folder *hashes*
    - scores.txt
- folder *vocabularies*
  - folder *corpus*
    - contains text corpus to parse dictionary=ie from it
    - folder source
      - contains source corpus (not in git as it has 8GB)
    - folder preprocessed
      - contains preprocessed corpus - parsed to Czech and English vocabulary and cleaned them
  - brands.txt
  - colors.txt

## Folder *results*
- folder *similarity_score*
  - folder *dataset_name*
    - folder *names*
    - folder *images*
    - folder *names_and_images*


## Folder *scripts*
This is the folder containing all the necessary scripts for product mapping

### folder *preprocessing* 
- folder *corpus_stuff*
  - contains all necessary scripts for corpus pareprocessing
  - *corpus_preprocessing.py*
  - *run_corpus_preprocessing.py*
    - load corpus file, split Czech and English sententes and create dictionary of unique words for each language 
  - *vocabulary_cleaner.py*
  - *run_vocab_cleaner.py*
    - check whether all words in manually created vocabulary from source corpus are existing words using MORPHODITA

- folder *images*
  - folder image_hash_creator
    - contains javascript code to create hashes from images
  - other scripts serve to preprocess images  
    - run_crop_images_contour_detection.py
      -
    - run_crop_images_simple.py
      -
    - run_unify_image_size.py
      -
  
- folder *names*
  - names_preprocessing.py
  - run_names_preprocessing.py
    - preprocess names of products - detects, ids, colors, brands
    
### folder *score_computation* 
- folder *images*
  - compare_hashes.py
  - run_compare_hashes.py
- folder *images_and_names*
  - compute_total_score.py
  - run_compute_total_score.py
- folder *names*
  - compute_name_similarity.py
  - run_compute_name_similarity.py
    -  compute similarity between two names

creating and comparison of image hashes including image crop for preprocessing.



	- contains imaga data of manually created 10 pairs of products
	- pairs of products are separated by name 01a_* and 01_b are corresponding pairs
folder image_data_cropped
	- contains cropped images of products
folder results
	- contains files with hashes and computed distances
crop_images_simple.py
	- crop white background around images using finding the edge nonwhite pixels
	- does not work well because there can be small logo in the corner of the image
crop_images_contour_detection.py
	- crop image using object detection using edges detection and crop image to the biggest object found in it
compare_hashes.py
	- using apify run from cmd call main.js which creates hashes of images in given folder)
	- compare_hashes compares created hashes using bit distance (comparison of % of bits that differs)
	- select the most similar images - they have the highest similarity of hashes
unify_image_size.py
	- unify all images in the folder to the chosen shape
  
  
  This is the folder containing all the encessary stuff for preprocessing names of the products

folder data 
	- contains all necessary data for names preprocessing
	- sub folders
		- names
			- contains all files with names of the products
			- names_a.txt and names_b.txt are two files with the same 10 product with different names from different web pages
			- names_czc.txt, names_datart.txt, names_mall.txt are just random examples of product names
		- small_corpus
			- contains all files before and after processing CZ corpus ParaCrawl v5.1 from http://www.statmt.org/wmt20/translation-task.html
			- en-cs_short.txt is the corpus file
			- cz_short.csv and en_short.csv are the files with unique words parsed from corpus
		- bigger_corpus
			- contains all files before and after processing CZ corpus from https://www.paracrawl.eu/index.php
			- en-cs.txt is the corpus file
			- cz.csv and en.csv are the files with unique words parsed from corpus
			- en_cleaned.txt and cz_cleaned.txt contains words from dictionary that were also found in MORPHODITA
		- vocabularies
			- folder containing all the vocabularies for name spreprocessing
			- brands.txt contains brands of notebooks extracted from Alza, Datart and CZC
			- colors.txt contains colors in English and Czech extracted from https://www.color-ize.com/color-list.php and https://cs.wikipedia.org/wiki/Seznam_barev
		- results
			- tf.idf.txt result of names_tf_idf.py that has computed tf.idfs in two files of product names
ReadMe.txt
	- File that you should definitely read!
