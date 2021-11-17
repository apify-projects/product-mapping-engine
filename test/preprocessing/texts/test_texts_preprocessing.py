from scripts.preprocessing.texts.keywords_detection import detect_ids_brands_colors_and_units
from scripts.preprocessing.texts.text_preprocessing import preprocess_text, set_czech_lemmatizer

test_texts = [
    '1,1 cm 3 h Notebook Lenovo IdeaPad 3 splní všechna vaše přání. Nabízí skvělý displej s jemným rozlišením, procesor s vysokým výkonem, rychlou RAM paměť a až 9,6 hodinovou výdrží baterie s podporou rychlého nabíjení. Dvojice výkonných reproduktorů s technologií Dolby Audio vám zprostředkuje kvalitní zvuk a zásluhou nízké hmotnosti si jej můžete vzít kamkoliv s sebou, aniž by se pronesl. A to nejlepší - jeho stylový design v mnoha barevných provedeních vám vyrazí dech.',
    'Představujeme notebook IdeaPad Gaming 3, který se postará o dokonalý herní zážitek. S procesory AMD, grafickou kartou Nvidia řady GTX a rychlou DDR4 RAM pamětí zajistí plynulé hraní i těch nejnáročnějších her. Navíc vypadá dokonale a systém chlazení dvou ventilátorů jej ponechá ledově chladným za všech situací!',
    'S notebookem Legion 5 nebude mít váš soupeř jedinou šanci. Je vybaven výkonným procesorem, velkou pamětí RAM a skvělou grafickou kartou, se kterou můžete hrát hry na vysoké FPS bez sekání. IPS displej s jemným rozlišením, vysokým jasem a obnovovací frekvencí 144 Hz ještě umocní skvělý zážitek z hraní her. A navíc má i nové chlazení Legion Coldfront 2.0, které je tiché a vysoce účinné!',
    'Lenovo ThinkBook 15 je firemní notebook se špičkovým zabezpečením, vysokou odolností, dlouhou výdrží baterie a rychlým nabíjením. S výkonným hardwarem a chytrými funkcemi budete s ThinkBookem 15 pracovat efektivněji a rychleji. Na první pohled upoutá pouhých 18,9 mm tenkou konstrukcí a stylovým šedým barevným provedením.',
    'Lenovo IdeaPad Duet 3i je unikátní zařízení, které v sobě kombinuje tablet a notebook. Můžete tak připojit klávesnici a psát, nebo ji odpojit a vychutnat si plnohodnotný tablet. Jeho vysoký výkon se postará o bezproblémový chod systému i aplikací, displej s dokonalým barevným podáním a jemným rozlišením spolu s dvojicí reproduktorů Dolby Audio zajistí dokonalou zábavu. Baterie s dlouhou výdrží umožní používání na cestách a nízká hmotnost umožní snadné přenášení. IdeaPad Duet 3 je tedy ideální společník pro každého.'
]


def main():
    lemmatizer = set_czech_lemmatizer()
    dataset = preprocess_text(test_texts, lemmatizer)
    dataset_detected = detect_ids_brands_colors_and_units(dataset, id_detection=False,
                                                          color_detection=True,
                                                          brand_detection=False,
                                                          units_detection=True)

    print(dataset_detected)


if __name__ == "__main__":
    main()
