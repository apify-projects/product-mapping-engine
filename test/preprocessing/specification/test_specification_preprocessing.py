from scripts.preprocessing.texts.specification_preprocessing import preprocess_specification
from scripts.score_computation.texts.compare_specification_similarity import compare_specifications

test_dataset1 = [["Provedení: Pecky 4ml", "Konstrukce: Uzavřená", "Mikrofon: Ano", "Typ připojení: Bluetooth",
                  "Verze: Bluetooth 5.0", "Maximální výdrž baterie: 25 h", "Výdrž baterie (sluchátka): 5 h",
                  "Výdrž baterie (pouzdro): 20 h", "Nabíjení: USB-C, V pouzdře", "Barva: Zlatá", "Hmotnost: 57 g"],
                 ["SSD Kapacita: 256 GB (0,26 TB)", "Velikost operační paměti RAM: 8 GB",
                  "Čip grafické karty: Apple M1 7jádrová GPU", "Modelové označení procesoru: Apple M1",
                  "Typ úložiště: SSD", "Úhlopříčka displeje: 13,3inch", "Kapacita úložiště: 256 GB",
                  "Výbava: Podsvícená klávesnice , Čtečka otisků prstů, Operační systém"]
                 ]
test_dataset2 = [["Provedení: Špunty", "Konstrukce: Uzavřená", "Mikrofon: Ano", "Typ připojení: Bluetooth",
                  "Verze Bluetooth: 5.0", "Typ připojení: Bluetooth", "Verze Bluetooth: 5.0",
                  "Maximální výdrž baterie: 32 h", "Výdrž baterie (sluchátka): 8 h", "Výdrž baterie (pouzdro): 24 h",
                  "Nabíjení: USB-C, V pouzdře", "Barva: Bílá", "Hmotnost: 73 g"],
                 ["SSD Kapacita: 1 000 GB (1 TB)", "Velikost operační paměti RAM: 16 GB",
                  "Čip grafické karty: NVIDIA GeForce RTX 3060", "Modelové označení procesoru: AMD Ryzen 5 5600H",
                  "Frekvence procesoru: 3,3 GHz (3 300 MHz)", "Typ úložiště: SSD", "Úhlopříčka displeje: 15,6inch",
                  "Kapacita úložiště: 1 000 GB",
                  "Výbava: Numerická klávesnice, Operační systém, RGB podsvícená klávesnice", ]
                 ]


def main():
    preprocessed_dataset1 = preprocess_specification(test_dataset1, separator=': ')
    preprocessed_dataset2 = preprocess_specification(test_dataset2, separator=': ')
    scores = compare_specifications(preprocessed_dataset1, preprocessed_dataset2)
    print('Similarity scores and cosine sim scores')
    print(scores)


if __name__ == "__main__":
    main()
