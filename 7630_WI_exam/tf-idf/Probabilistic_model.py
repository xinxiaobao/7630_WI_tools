import re
import math
from math import log
import numpy as np
import  pandas as pd

# parameter lambda
lambda1 = 0.9


print('\n===================== terms ========================\n')
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


# translate to pandas DataFrame
raw_freq = pd.DataFrame(tf, columns = terms)
print('\n===================== Raw Frequence ========================\n')
print(raw_freq)

print('\n================================================\n')

print('\n===================== P(t|d) not smoothing========================\n')
prob_td = []
for i in tf:
    prob_td.append(np.array(i)/sum(i))
# print(prob_td)
print(pd.DataFrame(np.around(prob_td, decimals=2), columns=terms))


print('\n================================================\n')

print('\n===================== P(t|d) smoothing========================\n')
prob_td_smooth = []
for i in tf:
    prob_td_smooth.append((np.array(i)+lambda1)/(lambda1*len(terms) + sum(i)))

prob_td_smooth_pandas = pd.DataFrame(np.around(prob_td_smooth, decimals=2), columns=terms)
print(prob_td_smooth_pandas)


print('================================================')
print('')
print('')
print('')
print('=======================query======================')
# print(terms)
query_list = list()
with open('./query.dat', 'r') as f:
    for line in f.readlines():
        line = line.strip()
        line = re.sub('[^A-Za-z\s]', ' ', line)
        line = line.lower()
        query_list.extend(line.split())
print(query_list)

m, n = prob_td_smooth_pandas.shape
for i in range(m):
    prob_td_temp = 1
    for j in query_list:
        prob_td_temp *= prob_td_smooth_pandas.iloc[i,:][j]
    print(i+1, '  p(q|d) = ', prob_td_temp)
    print(i+1, '  -log p(q|d) = ',np.round(-log(prob_td_temp, 10), 3))


# query_tf = list()
# for document in query_list:
#     temp = [0] * len(terms)
#     for term in document:
#         temp[terms.index(term)] += 1
#     query_tf.append(temp)
# print('tf')
# # print(query_tf)
# query_tf_pandas = pd.DataFrame(np.array(query_tf).reshape(1, len(terms)), columns=terms)
# print(query_tf_pandas)


# print('')
# query_normalized_tf = list()
# for document in query_tf:
#     max_value = max(document)
#     document = list(map(lambda x: x / max_value, document))
#     query_normalized_tf.append(document)
# print('normalized_tf')
# # for count in query_normalized_tf:
# #     print(np.around(count, decimals=3))

# # translate to pandas
# print(pd.DataFrame(np.around(query_normalized_tf, decimals=3).reshape(1, len(terms)), columns=terms))


# print('')
# print('df')
# # print(df)
# # translate to pandas
# print(pd.DataFrame(np.array(df).reshape(1, len(terms)), columns=terms))

# # pandas_df = pd.DataFrame(df, columns=terms)
# # print(pandas_df)
# # print(len(df))
# # print(len(terms))


# print('')
# print('idf')
# # print(np.around(idf, decimals=3))

# # translate to pandas
# print(pd.DataFrame(np.around(idf, decimals=3).reshape(1, len(terms)), columns=terms))


# print('')
# print('tf-idf')
# query_tf_idf = np.array(query_normalized_tf) * np.array(idf)
# # print(np.around(query_tf_idf, decimals=3))

# # translate to pandas
# print(pd.DataFrame(np.around(query_tf_idf, decimals=3).reshape(1, len(terms)), columns=terms))

# print('===================================================')
