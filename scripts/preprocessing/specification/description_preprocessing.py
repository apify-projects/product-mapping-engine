import re

import pandas as pd

from scripts.preprocessing.names.names_preprocessing import detect_ids_brands_and_colors

test_text = 'Notebook - AMD Ryzen 7 4800H, dotykový 14" IPS lesklý 2160 × 1440, RAM 16GB DDR4, AMD Radeon Vega Graphics, SSD 512GB, podsvícená klávesnice, webkamera, USB 3.2 Gen 1, USB-C, čtečka otisků prstů, WiFi 5, Hmotnost 1,49 kg, Windows 10 Home 53012GDQ'
test_texts = [
    'Notebook - AMD Ryzen 7 4800H, dotykový 14" IPS lesklý 2160 × 1440, RAM 16GB DDR4, AMD Radeon Vega Graphics, SSD 512GB, podsvícená klávesnice, webkamera, USB 3.2 Gen 1, USB-C, čtečka otisků prstů, WiFi 5, Hmotnost 1,49 kg, Windows 10 Home 53012GDQ',
    'Herní PC AMD Ryzen 7 5800X 4.7 GHz, NVIDIA GeForce RTX 3070 8GB, RAM 32GB DDR4, SSD 1000 GB + HDD 6 TB, Bez mechaniky, Wi-Fi, HDMI a DisplayPort, 2× USB 3.2, 2× USB 2.0, typ skříně: Midi Tower, Windows 10 Pro',
    'Fitness náramek - unisex, s přímým měřením tepu ze zápěstí, krokoměr, výpočet kalorií, monitoring spánku, hodnota vodotěsnosti 50 m (5 ATM), kompatibilní s iOS a Android',
    'Chytré hodinky - unisex s měřením tepu ze zápěstí, Apple Pay, monitoring spánku, telefonování pomocí hodinek přes spárovaný telefon, hodnota vodotěsnosti 50 m (5 ATM), watchOS, kompatibilní s iOS, vyžadují iPhone 6s a novější, kapacita úložiště 32 GB, ion-x sklíčko, materiál řemínku: silikon, materiál pouzdra: hliník',
    'MacBook - Apple M1, 13.3" IPS lesklý 2560 × 1600 , RAM 8GB, Apple M1 7jádrová GPU, SSD 256GB, podsvícená klávesnice, webkamera, USB-C, čtečka otisků prstů, WiFi 6, baterie 49,9 Wh, Hmotnost 1,25 kg, MAC OS',
    'Mobilní telefon - 6,5" AMOLED 2400 × 1080, 120Hz, procesor Qualcomm Snapdragon 778G 8jádrový, RAM 6 GB, interní paměť 128 GB, Micro SD až 1000 GB, zadní fotoaparát 64 Mpx (f/1,8) + 12 Mpx (f/2,2) + 5 Mpx (f/2,4) + 5 Mpx (f/2,4), přední fotoaparát 32 Mpx, optická stabilizace, GPS, Glonass, NFC, LTE, 5G, Jack (3,5mm) a USB-C, čtečka otisků v displeji, voděodolný dle IP67, hybridní slot, neblokovaný, rychlé nabíjení 25W, baterie 4500 mAh, Android 11']
UNITS_PATH = 'data/vocabularies/units.tsv'
PREFIXES_PATH = 'data/vocabularies/prefixes.tsv'

def split_params(text):
    """
    Split text to single parameteres separated by comma
    @param text: input text
    @return: split parameters
    """
    return text.split(',')


def remove_useless_spaces(text):
    text = re.sub(r'(?<=\d) - (?=\d)', r'-', text)
    text = re.sub(r'(?<=\d),(?=\d)', r'.', text)
    text = re.sub(r'(?<=\d)"', r' inch', text)
    text = text.replace(' × ', '×')
    text = text.replace('(', '')
    text = text.replace(')', '')
    return text


def split_words(text_list):
    """
    Split list of specifications to the single words
    @param text: list of specifications to be split
    @return: list of words of specifications
    """
    split_text = []
    rgx = re.compile("\w+[\"\-'×.,]?\w*")
    for text in text_list:
        words = rgx.findall(text)
        split_text.append(words)
    return split_text


def load_units_with_prefixes():
    """
    Load vocabulary with units and their prefixes and create all possible units variants and combination
    @return: Dataset with units and their prefixes
    """
    prefixes_df = pd.read_csv(PREFIXES_PATH, sep='\t', keep_default_na=False)
    units_df = pd.read_csv(UNITS_PATH, sep='\t', keep_default_na=False)
    units = pd.DataFrame(columns=units_df.columns)
    for idx, row in units_df.iterrows():
        if row['prefixes'] != '':
            shortcut = row['shortcut'].split(',')
            prefixes = row['prefixes'].split(',')
            name = row['name']
            plural = row['plural']
            czech = row['czech']
            for p in prefixes:
                for s in shortcut:
                    row['shortcut'] += f',{p}{s}'
                prefix_name = prefixes_df.loc[prefixes_df.prefix == p, "english"].values[0]
                row['name'] += f',{prefix_name}{name}'
                if row['plural'] != '':
                    row['plural'] += f',{prefix_name}{plural}'
                if row['czech'] != '':
                    row['czech'] += f',{prefixes_df.loc[prefixes_df.prefix == p, "czech"].values[0]}{czech}'
        units = units.append(row)
    return units.iloc[:, :-1]


def create_unit_vocabulary(units):
    """
    Create one list of all units from czech and english names and shortcuts
    @param units: dataframe with english, czech, plural and shortcuts of units
    @return: one list of all units variants
    """
    units_vocabulary = []
    for c in units.columns:
        col_list = units[c].tolist()
        col_words = [word.split(',') for word in col_list]
        col_words = [item.lower() for sublist in col_words for item in sublist if item != '']
        units_vocabulary.append(col_words)
    units_vocabulary = [item for sublist in units_vocabulary for item in sublist]
    return units_vocabulary


def detect_parameters(text):
    """
    Detect units in text according to the loaded dictionary
    @param text: text to detect parameters and units
    @return: text with detected parameters, separated parameters and values
    """
    params = []
    units = load_units_with_prefixes()
    unit_vocab = create_unit_vocabulary(units)
    detected_text = []
    for sentence in text:
        new_sentence = []
        previous = ''
        for word in sentence:
            word_new = word
            if word in unit_vocab and previous.replace('.', '', 1).isnumeric():
                word_new = "#UNIT#" + word
                params.append([word, float(previous)])
            new_sentence.append(word_new)
            previous = word
        detected_text.append(new_sentence)
    return detected_text, params


def main():
    for t in test_texts:
        text_cleaned = remove_useless_spaces(t)
        text_lower = text_cleaned.lower()
        text_split = split_params(text_lower)
        text_split = split_words(text_split)

        data, cnt_voc, cnt_lem = detect_ids_brands_and_colors(text_split, compare_words=False)
        detected_text, params = detect_parameters(data)
        # print(detected_text)
        print(params)
    print('wohoo!')


if __name__ == "__main__":
    main()
