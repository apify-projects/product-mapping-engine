This is the folder containing all the encessary stuff for creating and comparison of image hashes including image crop for preprocessing.

folder image_data 
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