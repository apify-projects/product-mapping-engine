from scripts.preprocessing.texts.text_preprocessing import preprocess_text, set_czech_lemmatizer
from scripts.score_computation.texts.compute_texts_similarity import compute_similarity_of_texts

test_texts1 = [
    '123e45 1 cm 3 h Notebook Lenovo IdeaPad 3 splní všechna vaše přání. Nabízí skvělý displej s jemným rozlišením, procesor s vysokým výkonem, rychlou RAM paměť a až 9,6 hodinovou výdrží baterie s podporou rychlého nabíjení. Dvojice výkonných reproduktorů s technologií Dolby Audio vám zprostředkuje kvalitní zvuk a zásluhou nízké hmotnosti si jej můžete vzít kamkoliv s sebou, aniž by se pronesl. A to nejlepší - jeho stylový design v mnoha barevných provedeních vám vyrazí dech.',
    'Představujeme notebook IdeaPad Gaming 3, který se postará o dokonalý herní zážitek. S procesory AMD, grafickou kartou Nvidia řady GTX a rychlou DDR4 RAM pamětí zajistí plynulé hraní i těch nejnáročnějších her. Navíc vypadá dokonale a systém chlazení dvou ventilátorů jej ponechá ledově chladným za všech situací!',
    'S notebookem Legion 5 nebude mít váš soupeř jedinou šanci. Je vybaven výkonným procesorem, velkou pamětí RAM a skvělou grafickou kartou, se kterou můžete hrát hry na vysoké FPS bez sekání. IPS displej s jemným rozlišením, vysokým jasem a obnovovací frekvencí 144 Hz ještě umocní skvělý zážitek z hraní her. A navíc má i nové chlazení Legion Coldfront 2.0, které je tiché a vysoce účinné!',
    'Lenovo ThinkBook 15 je firemní notebook se špičkovým zabezpečením, vysokou odolností, dlouhou výdrží baterie a rychlým nabíjením. S výkonným hardwarem a chytrými funkcemi budete s ThinkBookem 15 pracovat efektivněji a rychleji. Na první pohled upoutá pouhých 18,9 mm tenkou konstrukcí a stylovým šedým barevným provedením.',
    'Lenovo IdeaPad Duet 3i je unikátní zařízení, které v sobě kombinuje tablet a notebook. Můžete tak připojit klávesnici a psát, nebo ji odpojit a vychutnat si plnohodnotný tablet. Jeho vysoký výkon se postará o bezproblémový chod systému i aplikací, displej s dokonalým barevným podáním a jemným rozlišením spolu s dvojicí reproduktorů Dolby Audio zajistí dokonalou zábavu. Baterie s dlouhou výdrží umožní používání na cestách a nízká hmotnost umožní snadné přenášení. IdeaPad Duet 3 je tedy ideální společník pro každého.'
]
test_texts2 = [
    '123e45 1 cm Lenovo V14 G2 se svým tenkým a kompaktním provedení nejen že vypadá vypadá moderně a profesionálně, ale odpovídá tomu i moderní výbava a výkon pro profesionální použití. Je to spolehlivý a cenově dostupný společník pro každý den vašeho podnikání. Dalšími přednostmi notebooku Lenovo V14 G2 jsou vysoká odolnost a kvalitní zabezpečení vašich dat.',
    'Lenovo Legion 5 Pro 16ITH6H zaujme na první pohled do posledního detailu precizním zpracováním kovového těla v elegantním šedém barevném provedení v tradičním minimalistickém Lenovo designu s čistými liniemi. S ohromujícím výkonem procesoru, herní grafikou NVIDIA GeForce RTX a displejem s vysokou obnovovací frekvencí vám Lenovo Legion 5 Pro zprostředkuje dokonalé herní zážitky.    ',
    'Lenovo Legion 5 Pro je notebook pro hráče, kteří chtějí vyhrávat a užít si hraní her na maximum. Disponuje výkonným procesorem AMD nejnovější generace a špičkovou grafickou kartou Nvidia GeForce RTX pro plynulé hraní při nejvyšších detailech bez sekání. Jeho velký 16" displej s jemným QHD rozlišením má obnovovací frekvenci až 165 Hz a zaručuje tak ještě lepší zážitek z hraní her. Navíc má pokročilý systém chlazení Coldfront 3.0, který umožní vymáčknout z jeho hardwaru maximum.',
    'Rychlý, jednoduchý a bezpečný 11" Chromebook IdeaPad Flex 3 je poháněn procesorem MediaTek MT8183 a má flash úložiště pro plynulou odezvu systému. S hmotností pouhých 1,2 kg a dlouhou výdrží baterie vás může doprovázet po celý den. Chromebook IdeaPad Flex 3 nevyžaduje žádné nastavení - stačí se přihlásit pomocí účtu Google a můžete začít pracovat.',
    'Lenovo ThinkBook 14 je firemní notebook se špičkovým zabezpečením, vysokou odolností, dlouhou výdrží baterie a rychlým nabíjením.'
    'S výkonným hardwarem a chytrými funkcemi budete s ThinkBookem 14 pracovat efektivněji a rychleji. Na první pohled upoutá pouhých 17,9 mm tenkou konstrukcí a stylovým šedým barevným provedením.'
]


def main():
    lemmatizer = set_czech_lemmatizer()
    dataset1 = preprocess_text(test_texts1, lemmatizer)
    dataset2 = preprocess_text(test_texts2, lemmatizer)
    datasets_similarity = compute_similarity_of_texts(
        dataset1,
        dataset2,
        id_detection=True,
        color_detection=True,
        brand_detection=False,
        units_detection=True
    )
    print(datasets_similarity)


if __name__ == "__main__":
    main()
