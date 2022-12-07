import json
import sys
import pandas as pd
from string import punctuation
import pymorphy2
import nltk
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering
from sklearn import mixture
nltk.download('punkt')
nltk.download('stopwords')
import requests as requests
from bs4 import BeautifulSoup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score


def parsing(url, list):
    links_filter = []
    get_url = requests.get(url)
    if get_url.status_code == 200:
        #Декодируем байты в строку
        get_html = get_url.text
        soup = BeautifulSoup(get_html, 'html.parser')

        links = soup.findAll('a', {'class': ['article-link']})

        #Получаем ссылки
        for i in range(len(links)):
            k = (links[i].get('href'))
            links_filter.append(k)

        for i in range(len(links)):
            k = ('https://tproger.ru/news/' + links_filter[i])

            site = requests.get(k)
            get_html = site.text
            soup = BeautifulSoup(get_html, 'html.parser')

            # Получаем текст
            article = soup.find('div', attrs={'class': 'entry-content'})
            all_p = article.find_all('p')
            text_from_p = (p.text.strip() for p in all_p)
            text = ' '.join(text_from_p)

            #Источник
            author = soup.find('meta', attrs={'name': 'author'})
            aut = author.attrs.get('content')

            list.append(
                {'Text': text,
                 'Author': aut}
            )
    else:
        if get_url.status_code == 404:
            print('\n Page not found!')
            sys.exit()
        print('\n Fatal error!')
        sys.exit()

    return list

def replace(count):
    if pd.isnull(count):
        return nan_count
    return counts[count]

def lemmatize(input_text):
    tokens = nltk.word_tokenize(input_text)
    normed_tokens = [morph.parse(s)[0].normal_form for s in tokens]
    normed_tokens = [word for word in normed_tokens if word not in nltk.corpus.stopwords.words("russian")]
    normed_tokens = [word for word in normed_tokens if word not in punctuation]
    return ' '.join(normed_tokens)

if __name__ == '__main__':
    # list = []
    # for i in range(1,20):
    #     url = ('https://tproger.ru/news/page/'+ str(i) + '/')
    #     list = parsing(url, list)
    # with open("data.json", "w", encoding="utf-8") as write_file:
    #     json.dump(list, write_file, ensure_ascii=False, indent=4)
    pd.set_option('display.max_rows', 500)
    data = pd.read_json('data.json')
    #Дропнем все пустые значения
    data = data.dropna()

    #print(data['From'].value_counts())
    #Заменим значения в столбце From на количество признаков в группе
    counts = data['From'].value_counts()
    nan_count = data['From'].isnull().sum()
    data['From'] = data['From'].apply(replace)

    # Токенизируем тексты
    morph = pymorphy2.MorphAnalyzer()
    data['Text'] = data['Text'].apply(lemmatize)

    # Разделяем выборку на обучающую и тестовую
    X_train, X_test, y_train, y_test = train_test_split(data['Text'], data['From'], test_size=0.2,
                                                        stratify=data['From'])
    #Векторизация
    tfidf_vect = TfidfVectorizer()
    tfidf_vect.fit(np.hstack([X_train, X_test]))
    X_train = tfidf_vect.transform(X_train)
    X_test = tfidf_vect.transform(X_test)

    #Определим гиперпарамтер
    kfold = KFold(n_splits=6, shuffle=True, random_state=10)  # генератор разбиений
    accuracy = []
    C_list = (10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5)

    for i in range(0, len(C_list)):
        clf = LogisticRegression(C=C_list[i], random_state=10, max_iter=200)
        current_score = cross_val_score(clf, X_train, y_train, cv=kfold).mean()
        accuracy.append(current_score)
        print('При С =', C_list[i], 'точность равна', current_score)
    max_score = 0
    for i in range(0, len(accuracy)):
        if max_score < accuracy[i]:
            max_score = accuracy[i]
            C_for_max_score = i
    print('\n')
    print('Максимальная точность, равная', max_score, ', получается при C, равном', C_list[C_for_max_score])

    #В качестве вероятностной модели будем использовать логистическую регрессию
    Log_Pred = LogisticRegression(C=C_list[C_for_max_score], random_state=10).fit(X_train.toarray(), y_train.tolist()).predict(
        X_test.toarray())
    # Метрики качества
    print('\nLogistic_Accuracy:', accuracy_score(Log_Pred, y_test.tolist()))
    print('Logistic_Precision:',
          precision_score(Log_Pred, y_test.tolist(), average='weighted'))
    print('Logistic_Recall:',
          recall_score(Log_Pred, y_test.tolist(), average='weighted'))
    print('Logistic_F:', f1_score(Log_Pred, y_test.tolist(), average='weighted'))

    # Метод ближайших соседей, гиперпараметр модели - количество соседей = 5
    KNN_pred = KNeighborsClassifier(n_neighbors=5).fit(X_train.toarray(), y_train.tolist()).predict(
        X_test.toarray())

    print('\nKNN_Accuracy:', accuracy_score(KNN_pred, y_test.tolist()))
    print('KNN_Precision:', precision_score(KNN_pred, y_test.tolist(), average='weighted'))
    print('KNN_Recall:', recall_score(KNN_pred, y_test.tolist(), average='weighted'))
    print('KNN_F:', f1_score(KNN_pred, y_test.tolist(), average='weighted'))

    # Кластеризация
    # k-средние
    X = np.concatenate([X_train.toarray(), X_test.toarray()])
    y = y_train.append(y_test)

    k_means = KMeans(init='k-means++', n_clusters=5, n_init=10)
    k_means_pred = k_means.fit_predict(X)

    tsne = TSNE().fit_transform(X)

    plt.scatter(
        tsne[:, 0], tsne[:, 1],
        c=k_means_pred
    )
    plt.show()

    # Метрики качества кластеризации
    print("\nK_me_Thomogeneity_score: ", metrics.homogeneity_score(y, k_means_pred))
    print("K_me_Completeness_score: ", metrics.completeness_score(y, k_means_pred))
    print("K_me_V_Measure_score: ", metrics.v_measure_score(y, k_means_pred))

    # EM кластеризация
    EM = mixture.GaussianMixture(n_components=5, covariance_type='diag')
    EM_pr = EM.fit_predict(X)
    plt.scatter(
        tsne[:, 0], tsne[:, 1],
        c=EM_pr
    )
    plt.show()
    print("\nEM_Thomogeneity_score: ", metrics.homogeneity_score(y, EM_pr))
    print("EM_Completeness_score: ", metrics.completeness_score(y, EM_pr))
    print("EM_V_Measure_score: ", metrics.v_measure_score(y, EM_pr))


    #Иерархическая кластеризация
    cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward')
    cluster_pr = cluster.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=cluster.labels_, cmap='rainbow')
    plt.show()


    print("\nIE_Thomogeneity_score: ", metrics.homogeneity_score(y, cluster_pr))
    print("IE_Completeness_score: ", metrics.completeness_score(y, cluster_pr))
    print("IE_V_Measure_score: ", metrics.v_measure_score(y, cluster_pr))

