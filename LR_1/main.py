
#Импортируем библиотеку с предобученными эмбеддингами
import string

from natasha import (
    Segmenter,

    NewsEmbedding,
    NewsMorphTagger,
    NewsSyntaxParser,

    Doc
)
from fonetika.soundex import RussianSoundex  # Импорт библиотеки для фонетического кодирования
from nltk import word_tokenize

if __name__ == '__main__':
    #Определяем сегментатор
    segmenter = Segmenter()

    #Определяем предобученный эмбеддинг для морфологического и синтаксичесого анализа
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)
    syntax_parser = NewsSyntaxParser(emb)

    #Открываем текст
    f = open('input.txt', "r", encoding="utf-8")
    text = f.read()

    #Собираем Doс объект. Применяем сгементацию и анализаторы Для каждого токена извлекаем теги морфологии
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    doc.parse_syntax(syntax_parser)
    #Определяем предложение, которое будем анализировать
    sent = doc.sents[4]
    #Вывод результатов
    print("---------------------------------Морфологический анализ---------------------------------")
    sent.morph.print()
    print("---------------------------------Синтаксический анализ---------------------------------")
    sent.syntax.print()

    #Фонетическое кодирование
    print("---------------------------------Фонетическое кодирование---------------------------------")
    #Предварительная обработка предложения
    spec_chars = string.punctuation + '\n\t—…«»'
    text = "".join([ch for ch in sent.text if ch not in spec_chars])
    text_tokens = word_tokenize(text)
    soundex = RussianSoundex(delete_first_letter=False)
    for i in range(len(text_tokens)):
        print('Фонетический код слова ' + '"' + text_tokens[i] + '"' + ': ' + soundex.transform(text_tokens[i]))

    #Фонетическое кодирование
    print("---------------------------------Выделение словосочетаний---------------------------------")

    from razdel import sokr
    tokens = list(tokenize('Кружка-термос на 0.5л (50/64 см³, 516;...)'))
    print(tokens)