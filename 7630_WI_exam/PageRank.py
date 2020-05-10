# Q1 b
import numpy as np 

# ATD = [[0, 0.5, 0.5],
#         [0, 0, 0.5],
#         [0, 0.5, 0]]


ATD = [[0, 0.5, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0]]

    
ce = [1/3, 1/3, 1/3]

ATD = np.array(ATD)
ce = np.array(ce).T
print(ATD, ce)
print('=============')


n = 20
for i in range(n):
    ce = ATD @ ce
    print('Iteration'+ str(i+1), ':',ce)
# print('=============')
# print(ce)


# Q1 c 
# m = 20
# for i in range(m):
#     ce = 0.05 + 0.85 * (ATD @ ce)
#     print('Iteration'+ str(i+1), ':',ce)


# print(np.array([1/3]*3))
# print([1/3]*3)