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
vocabulary_cleaner.py
	- check whether the word in manually craeted vocabulary from corpus are existing words using MORPHODITA
names_tf_idf.py
	- compute tf.ids in files with names
names_prepro.py
	- main script for names preprocseeing - detects, ids, colors, brands and computes similarity
corpus_prepro.py
	- load corpus file and create dictionary of unique words