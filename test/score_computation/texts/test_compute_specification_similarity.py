from scripts.dataset_handler.preprocessing.texts.specification_preprocessing import convert_specifications_to_texts, \
    parse_specifications
from scripts.dataset_handler.preprocessing.texts.text_preprocessing import preprocess_text
from scripts.dataset_handler.similarity_computation.texts.compute_specifications_similarity import \
    preprocess_specifications_and_compute_similarity
from scripts.dataset_handler.similarity_computation.texts.compute_texts_similarity import compute_similarity_of_texts

test_dataset1 = ['[{"key": "Provedení", "value": "1 GB Pecky"}, {"key": "Konstrukce", "value": "Uzavřená"},\
                  {"key": "Mikrofon", "value": "Ano"}, {"key": "Typ připojení", "value": "Bluetooth"},\
                  {"key": "Verze Bluetooth", "value": "5.0"}, {"key": "Maximální výdrž baterie", "value": "25 h"},\
                  {"key": "Výdrž baterie (sluchátka)", "value": "5 h"},\
                  {"key": "Výdrž baterie (pouzdro)", "value": "20 h"},\
                  {"key": "Nabíjení", "value": "USB-C, V pouzdře"}, {"key": "Barva", "value": "Zlatá"},\
                  {"key": "Hmotnost", "value": "57 g"}]',
                 '[{"key": "SSD Kapacita", "value": "256 GB (0,26 TB)"},\
                  {"key": "Velikost operační paměti RAM", "value": "8 GB"},\
                  {"key": "Čip grafické karty", "value": "Apple M1 7jádrová GPU"},\
                  {"key": "Modelové označení procesoru", "value": "Apple M1"},\
                  {"key": "Typ úložiště", "value": "SSD"},\
                  {"key": "Úhlopříčka displeje", "value": "13,3inch"}, {"key": "Kapacita úložiště", "value": "256 GB"},\
                  {"key": "Výbava", "value": "Podsvícená klávesnice, Čtečka otisků prstů, Operační systém"}]']
test_dataset2 = ['[{"key": "Provedení", "value": "Špunty"}, {"key": "Konstrukce", "value": "Uzavřená"}, \
                  {"key": "Mikrofon", "value": "Ano"}, {"key": "Typ připojení", "value": "Bluetooth"}, \
                  {"key": "Verze Bluetooth", "value": "5.0"}, {"key": "Maximální výdrž baterie", "value": "32 h"}, \
                  {"key": "Výdrž baterie (sluchátka)", "value": "8 h"}, \
                  {"key": "Výdrž baterie (pouzdro)", "value": "24 h"}, \
                  {"key": "Nabíjení", "value": "USB-C, V pouzdře"}, {"key": "Barva", "value": "Bílá"}, \
                  {"key": "Hmotnost", "value": "73 g"}]',
                 '[{"key": "SSD Kapacita", "value": "1 000 GB (1 TB)"}, \
                  {"key": "Velikost operační paměti RAM", "value": "16 GB"}, \
                  {"key": "Čip grafické karty", "value": "NVIDIA GeForce RTX 3060"}, \
                  {"key": "Modelové označení procesoru", "value": "AMD Ryzen 5 5600H"}, \
                  {"key": "Frekvence procesoru", "value": "3,3 GHz (3 300 MHz)"}, \
                  {"key": "Typ úložiště", "value": "SSD"}, \
                  {"key": "Úhlopříčka displeje", "value": "15,6inch"}, {"key": "Kapacita úložiště", "value": "1000 GB"}, \
                  {"key": "Výbava", "value": "Numerická klávesnice, Operační systém, RGB podsvícená klávesnice"}]']


def main():
    test_dataset_prepro1 = parse_specifications(test_dataset1)
    test_dataset_prepro2 = parse_specifications(test_dataset2)

    preprocessed_specification_as_text1 = convert_specifications_to_texts(test_dataset_prepro1)
    preprocessed_specification_as_text2 = convert_specifications_to_texts(test_dataset_prepro2)

    preprocessed_specification_as_text1 = preprocess_text(preprocessed_specification_as_text1)
    preprocessed_specification_as_text2 = preprocess_text(preprocessed_specification_as_text2)

    similarity_score_of_texts = compute_similarity_of_texts(
        preprocessed_specification_as_text1,
        preprocessed_specification_as_text2,
        id_detection=False,
        color_detection=True,
        brand_detection=False,
        units_detection=True
    )
    print('Cosine similarity scores:')
    print(similarity_score_of_texts)

    scores = preprocess_specifications_and_compute_similarity(test_dataset_prepro1, test_dataset_prepro2)
    print('Matching similarity scores:')
    print(scores)


if __name__ == "__main__":
    main()
