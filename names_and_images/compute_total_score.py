

# PARAMS
IMG_WEIGHT = 1
NAME_WEIGHT = 1

names_file = '../names/data/results/scores_ab.txt'
images_file = '../img_hash/data/results/all_dist_bin_cropped.txt'

def load_file(file_name):
    data = []
    file = open(file_name, 'r', encoding='utf-8')
    lines = file.read().splitlines() 
    for line in lines:
        data.append(line)
    return data
      
# save names, similarities and whether they are the same to output file
def save_to_file(output_file, name1, name2, score, are_names_same):
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f'{name1}, {name2}, {score}, {are_names_same}\n')
        
def main():
    names = load_file(names_file)
    images = load_file(images_file)


if __name__ == "__main__":
    main()
        