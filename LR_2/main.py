import string
import sys
import pymorphy2
morph = pymorphy2.MorphAnalyzer(lang='ru')
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk import word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords

def obrabotka(text):
    # Приводим текст к нижнему регистру, удаляем пунтуацию
    text = text.lower()
    spec_chars = string.punctuation + '\n\t—…«»'
    text = "".join([ch for ch in text if ch not in spec_chars])

    # Токенизация текста
    text_tokens = word_tokenize(text)
    return (text_tokens)

def compute_tf(text):
    #На вход берем текст в виде токенов
    #Считаем частотность всех терминов во входном массиве с помощью
    #метода Counter библиотеки collections
    e = nltk.Text(text)
    tf_text = FreqDist(e)
    for i in tf_text:
        #для каждого слова в tf_text считаем TF путём деления
        #встречаемости слова на общее количество слов в тексте
        tf_text[i] = tf_text[i]/float(len(text))
    #возвращаем объект типа Counter c TF всех слов текста
    return tf_text

def delete_objets(text_tokens):
    # Удаление стоп слов
    russian_stopwords = stopwords.words('russian')
    # Посмотрев на первый график можем сделать вывод
    #Что самые популярыне слова - это местоимения и различные стоп слова
    #Удалим их, так же добавив в список стоп слов слова очень и это, которые так же не несут cмысловой нагрузки
    #Но Имеют высокую частоту встречаемости
    russian_stopwords.extend(['это','очень'])
    tokens_without_sw = [word for word in text_tokens if not word in russian_stopwords]
    return tokens_without_sw

def sklon(a):
    sklon = []
    p = morph.parse(a)[0]
    sklon.append(p.inflect({'nomn'}).word)
    sklon.append(p.inflect({'gent'}).word)
    sklon.append(p.inflect({'datv'}).word)
    sklon.append(p.inflect({'accs'}).word)
    sklon.append(p.inflect({'ablt'}).word)
    sklon.append(p.inflect({'loct'}).word)
    sklon.append(p.inflect({'voct'}).word)
    sklon.append(p.inflect({'gen2'}).word)
    sklon.append(p.inflect({'acc2'}).word)
    sklon.append(p.inflect({'loc2'}).word)
    return sklon

def search(text_tokens):
    count = 0
    print('\nХотите ли вы произвести какой то поиск? (+/-)')
    a = (str(input()))
    while a != '+':
        print('Пока :-( ')
        if a == '-': sys.exit()
        a = (str(input()))
    print('\nВведите слово для поиска в статье:')
    a = (str(input()))
    lel = sklon(a)
    qeq = list(set(lel))
    for i in range (len(qeq)):
        for j in range (len(text_tokens)):
            if text_tokens[j] == qeq[i]:
                count = count + 1
                print('Слово похожее на ваше в статье: ' + text_tokens[j])
    if count == 0: print('Ничего не найдено')


if __name__ == '__main__':
    f = open('1.txt', "r", encoding="utf-8")
    example_article = f.read()
    #Проведем предобработку текста(удалим пунтуацию и приведем в нижний регистр)
    text1 = obrabotka(example_article)

    # Подсчитаем TF каждого слова в текста
    tf_text_1 = compute_tf(text1)

    #Посмотрим график встречаемости слов первых 50 самых популярных слов
    tf_text_1.plot(50)

    text2 = obrabotka(example_article)
    #Удалим некоторые объекты
    tf_text_2_del = delete_objets(text2)
    text_2_finish = compute_tf(tf_text_2_del)
    text_2_finish.plot(50)
    e = nltk.Text(text1)
    text_for_search = FreqDist(e)
    search(list(text_for_search.keys()))