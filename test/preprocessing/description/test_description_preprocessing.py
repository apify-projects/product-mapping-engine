from scripts.preprocessing.description.description_preprocessing import remove_useless_spaces, split_words, \
    split_params, detect_parameters, compare_units_in_descriptions
from scripts.preprocessing.names.names_preprocessing import split_units_and_values

test_text = 'Notebook - AMD Ryzen 7 4800H, dotykový 14" IPS lesklý 2160 × 1440, RAM 16GB DDR4, AMD Radeon Vega Graphics, SSD 512GB, podsvícená klávesnice, webkamera, USB 3.2 Gen 1, USB-C, čtečka otisků prstů, WiFi 5, Hmotnost 1,49 kg, Windows 10 Home 53012GDQ'
test_texts = [
    'Notebook - AMD Ryzen 7 4800H, dotykový 14" IPS lesklý 2160 × 1440, RAM 16GB DDR4, AMD Radeon Vega Graphics, SSD 512GB, podsvícená klávesnice, webkamera, USB 3.2 Gen 1, USB-C, čtečka otisků prstů, WiFi 5, Hmotnost 1,49 kg, Windows 10 Home 53012GDQ',
    'Herní PC AMD Ryzen 7 5800X 4.7 GHz, NVIDIA GeForce RTX 3070 8GB, RAM 32GB DDR4, SSD 1000 GB + HDD 6 TB, Bez mechaniky, Wi-Fi, HDMI a DisplayPort, 2× USB 3.2, 2× USB 2.0, typ skříně: Midi Tower, Windows 10 Pro',
    'Fitness náramek - unisex, s přímým měřením tepu ze zápěstí, krokoměr, výpočet kalorií, monitoring spánku, hodnota vodotěsnosti 50 m (5 ATM), kompatibilní s iOS a Android',
    'Chytré hodinky - unisex s měřením tepu ze zápěstí, Apple Pay, monitoring spánku, telefonování pomocí hodinek přes spárovaný telefon, hodnota vodotěsnosti 50 m (5 ATM), watchOS, kompatibilní s iOS, vyžadují iPhone 6s a novější, kapacita úložiště 32 GB, ion-x sklíčko, materiál řemínku: silikon, materiál pouzdra: hliník',
    'MacBook - Apple M1, 13.3" IPS lesklý 2560 × 1600 , RAM 8GB, Apple M1 7jádrová GPU, SSD 256GB, podsvícená klávesnice, webkamera, USB-C, čtečka otisků prstů, WiFi 6, baterie 49,9 Wh, Hmotnost 1,25 kg, MAC OS',
    'Mobilní telefon - 6,5" AMOLED 2400 × 1080, 120Hz, procesor Qualcomm Snapdragon 778G 8jádrový, RAM 6 GB, interní paměť 128 GB, Micro SD až 1000 GB, zadní fotoaparát 64 Mpx (f/1,8) + 12 Mpx (f/2,2) + 5 Mpx (f/2,4) + 5 Mpx (f/2,4), přední fotoaparát 32 Mpx, optická stabilizace, GPS, Glonass, NFC, LTE, 5G, Jack (3,5mm) a USB-C, čtečka otisků v displeji, voděodolný dle IP67, hybridní slot, neblokovaný, rychlé nabíjení 25W, baterie 4500 mAh, Android 11']


def main():
    parameters = []
    datas = []
    for t in test_texts:
        text_cleaned = remove_useless_spaces(t)
        text_lower = text_cleaned.lower()
        text_split = split_params(text_lower)
        text_split = split_words(text_split)
        data = split_units_and_values(text_split)
        for d in data:
            d_detected, params = detect_parameters(d)
            parameters.append(params)
            datas.append(d_detected)
    print(datas)
    print(parameters)
    similarity_scores = compare_units_in_descriptions(parameters, parameters)
    print(similarity_scores)
    print('wohoo!')


if __name__ == "__main__":
    main()
