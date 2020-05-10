import re
from math import log

import numpy as np

print('=====================tf========================')
documents_list = list()
with open('./tf-idf.dat', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = re.sub('[^A-Za-z\s]', ' ', line)
        line = line.lower()
        documents_list.append(line.split())

terms = list()
for document in documents_list:
    for term in document:
        if term not in terms:
            terms.append(term)
print(terms)

tf = list()
for document in documents_list:
    temp = [0] * len(terms)
    for term in document:
        temp[terms.index(term)] += 1
    tf.append(temp)

for count in tf:
    print(count)
print('================================================')
print('')
print('')
print('')
print('=====================Normalized========================')
normalized_tf = list()
print(terms)
for document in tf:
    max_value = max(document)
    document = list(map(lambda x: x / max_value, document))
    normalized_tf.append(document)

for count in normalized_tf:
    print(np.around(count, decimals=3))
print('================================================')
print('')
print('')
print('')
print('=====================df========================')
print(terms)
document_num = len(documents_list)
df = [0] * len(terms)
for document in tf:
    for index, count in enumerate(document):
        if count != 0:
            df[index] += 1

print(df)
print('================================================')
print('')
print('')
print('')
print('=======================idf======================')
idf = list(map(lambda x: log((document_num / x), 10), df))
print(np.around(idf, decimals=3))
print('================================================')
print('')
print('')
print('')
print('=======================tf-idf======================')
temp = list()
temp.append(idf)
temp *= document_num
tf_idf = np.array(normalized_tf) * np.array(temp)
print(np.around(tf_idf, decimals=3))
print('================================================')
print('')
print('')
print('')
print('=======================query======================')
print(terms)
query_list = list()
with open('./query.dat', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = re.sub('[^A-Za-z\s]', ' ', line)
        line = line.lower()
        query_list.append(line.split())

query_tf = list()
for document in query_list:
    temp = [0] * len(terms)
    for term in document:
        temp[terms.index(term)] += 1
    query_tf.append(temp)
print('tf')
print(query_tf)

print('')
query_normalized_tf = list()
for document in query_tf:
    max_value = max(document)
    document = list(map(lambda x: x / max_value, document))
    query_normalized_tf.append(document)
print('normalized_tf')
for count in query_normalized_tf:
    print(np.around(count, decimals=3))

print('')
print('df')
print(df)

print('')
print('idf')
print(np.around(idf, decimals=3))

print('')
print('tf-idf')
query_tf_idf = np.array(query_normalized_tf) * np.array(idf)
print(np.around(query_tf_idf, decimals=3))
print('===================================================')
