import json, sys
import numpy as np
import pandas as pd

input_file = 'hashes_cropped.json' 
out_file = 'distances_cropped_bin.csv' #'distances.csv'
out_file_all = 'all_dist_cropped_bin.csv' #all_dist.csv
metric = 'binary' #binary, mean, thresh

BIT_GROUPS = 4
NAME_CHAR_SUBSET = 3
THRESHOLD = 20000
all_dist = []

# load file and split name and hash into dictionary
def load_and_parse_data(input_file):
    data = {}
    with open(input_file) as json_file:
        loaded_data = json.load(json_file)
    
    for d in loaded_data:
        dsplit = d.split(';')
        data[dsplit[0]] = dsplit[1]
    return data

# create list of lists of image hashes and names for each product
def create_hash_sets(data):      
    hashes = []
    names = []
    last_name = list(data.keys())[0][:NAME_CHAR_SUBSET]
    hash_set = []
    img_names = []
    for name, hashval in data.items():
        if name[:NAME_CHAR_SUBSET] == last_name:
            hash_set.append(hashval)
            img_names.append(name)
        else:    
            hashes.append(hash_set)
            names.append(img_names)
            hash_set = []
            img_names = []
            hash_set.append(hashval)
            img_names.append(name)
            last_name = name[:NAME_CHAR_SUBSET]
    return hashes, names

# takes every 4ths of characters and converts them from hex to dec
def hex_to_dec(hex_val):
    fourths = [hex_val[i:i+BIT_GROUPS] for i in range(0, len(hex_val), BIT_GROUPS)]
    return [int(f, 16) for f in fourths]

# convert hex value to binary char by char
def hex_to_bin(val):
    hash_bin = []
    for i in val:
        hash_bin.append(bin(int(i, 16))[2:].zfill(4))
    return str.join('', [val for sub in hash_bin for val in sub])

# compute difference of two list values
def compute_difference(list1, list2):
    diff = 0
    for i, j in zip(list1, list2):
        diff+=(abs(i-j))
    return diff   
 
# find nearest image to given one from the set of images
def get_nearest_image(img, name, imgset, names):
    dist = sys.maxsize
    nearest_name = None
    for i, (img2, name2) in enumerate(zip(imgset, names)):  
        diff = compute_difference(img, img2)
        all_dist.append([name, name2, diff])
        if diff < dist:
            dist = diff
            nearest_name = name2
    return dist, nearest_name

# compute mean distance of values in array
def mean_distance(data, non_zero=True):
    if len(data)==0:
        return None
    if non_zero:
        non_zero_vals = [float(v) for v in data if v!=None]
        if len(non_zero_vals)==0:
            return None
        return sum(non_zero_vals)/len(non_zero_vals)
    return data.mean()

# compute distance from threshold in array
def thresh_distance(data):
    if len(data)==0:
        return None
    
    distance = 0
    for d in data:
        if d == None:
            d=0
        distance += abs(THRESHOLD-d)
    return distance/len(data)

# find image from the image set for given image with the highest number of matching bits
def bit_distance(img, name, imgset, names):
    match = 0
    nearest_name = None
    for img2, name2 in zip (imgset, names):
        matches = 0
        for i1, i2 in zip(img, img2):
            if i1==i2:
                matches+=1
        if matches > match:
            match = matches
            nearest_name = name2
        all_dist.append([name, name2, matches/len(img)])
    return match/len(img), nearest_name    
 
# swap first and second set of images, if second has less images than first
def swap_image_sets(first_hash, first_name, second_hash, second_name):
    return second_hash, second_name, first_hash, first_name
        
# convert hashes to dec and compute distances between two set of images of one product
def compute_distances(hashes, names, metric):
    total_dist = 0
    distance_set = []
    
    for first_hash, second_hash, first_name, second_name in zip(hashes[::2], hashes[1::2], names[::2], names[1::2]):
        distances = []
        if len(first_hash) > len(second_hash):
            first_hash, first_name, second_hash, second_name = swap_image_sets(first_hash, first_name, second_hash, second_name)
        
        if metric == 'binary':
            first_hash = [hex_to_bin(i) for i in first_hash]
            second_hash = [hex_to_bin(i) for i in second_hash]
        else:
            first_hash = [hex_to_dec(i) for i in first_hash]
            second_hash = [hex_to_dec(i) for i in second_hash]
        
        
        # for each image from the first set find the most similar in the second set and check whether is the distance below threshold
        for num, name in zip(first_hash, first_name):
            dist, nearest_name = 0, None
            if metric == 'binary':
                dist, nearest_name = bit_distance(num, name, second_hash, second_name)
            else:
                dist, nearest_name = get_nearest_image(num, name, second_hash, second_name)
                if dist>THRESHOLD:
                    dist, nearest_name = None, None
                
            distance_set.append([name, nearest_name, dist])
            distances.append(dist)
            
        dst = [float(v) for v in distances if v!=None]
        total_dist += sum(dst)
        
        # compute total distance of sets images
        mean_total_distance = mean_distance(np.array(distances))
        thresh_total_distance = thresh_distance(np.array(distances))
            
    return total_dist, distance_set

        
def main():
    data = load_and_parse_data(input_file)
    hashes, names = create_hash_sets(data)
    suma, distance_set = compute_distances(hashes, names, metric=metric)
    print(suma)
    df = pd.DataFrame(distance_set, columns=['image1', 'image2', 'dist'])
    df.to_csv(out_file)
    
    df = pd.DataFrame(all_dist, columns=['image1', 'image2', 'dist'])
    df.to_csv(out_file_all)
    
if __name__ == "__main__":
    main()
    
    
# TODO: jak resit ze 2 obrazky muzou mit jeden stejny nejpodovnejsi
        
    


