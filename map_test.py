# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 19:59:56 2022

@author: HYF
"""

import time

def func(x):
    x = (x**2)**2
    return x

lists = list(range(10000000))

start = time.time()
a=[]
for num in lists:
    a.append(func(num))
end = time.time()
print('Serial computing time:\t',end - start)

start = time.time()
b = map(func,lists)
end = time.time()
print('Parallel Computing time:\t',end - start)

print(a == list(b))


list(zip(1,2,3,4,5,6))
