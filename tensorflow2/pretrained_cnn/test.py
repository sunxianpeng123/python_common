# encoding: utf-8

"""
@author: sunxianpeng
@file: test.py
@time: 2020/7/14 10:11
"""
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os

for i in [tf, np, tfds]:
    print(i.__name__, ": ", i.__version__, sep="")

(raw_train, raw_validation, raw_test), metadata = tfds.load("cats_vs_dogs",
                                                            split=["train[:80%]", "train[80%:90%]",
                                                                   "train[90%:]"],
                                                            with_info=True,
                                                            as_supervised=True)
for i in (raw_train, raw_validation, raw_test):
    print(i)
exit(0)

class Main():
    def __init__(self):
        pass

    def load_data(self):
        (raw_train, raw_validation, raw_test), metadata = tfds.load("cats_vs_dogs",
                                                                    split=["train[:80%]", "train[80%:90%]",
                                                                           "train[90%:]"],
                                                                    with_info=True,
                                                                    as_supervised=True)
        for i in (raw_train, raw_validation, raw_test):
            print(i)


if __name__ == '__main__':
    m =Main()
    m.load_data()
