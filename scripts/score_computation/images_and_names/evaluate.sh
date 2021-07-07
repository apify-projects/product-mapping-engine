# preprocess images and create image hash
python3 scripts/preprocessing/images/run_crop_images_contour_detection.py -i *inputimagefolder* -o *outputimagefile*
node scripts/images/image_hash_creator/main.js *inputimagefolder* *outputimagefile*

# preprocess names
python3 scripts/preprocessing/names/run_names_preprocessing.py -i *inputnamefile* -o *outputnamefile*

# compute total similarity
python3 scripts/score_computation/names/run_compute_names_similarity.py  -i *inputnamefile* -o *outputnamefile* -iw *weightid* -bw *weightbrand*  -cw *cosweight* -ww *wordweight* -t *thresh*
python3 scripts/score_computation/images_and_names/run_compute_total_similarity.py -n *namesfile* -i *imagesfile* -o *outputfile* -nw *nameweight* -iw *imageweight*
