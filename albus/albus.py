import numpy
import pandas
import missingno
# -*- coding: utf-8 -*-
from dataset import Dataset


class Albus:
    def __init__(self, data, *args, **kwargs):
        self.dataset = Dataset(data)
        self.null_info = self.dataset.get_null_info()


data = numpy.random.randn(5, 3)
dict_data = {
     0: ["olamilekan", "ayomide", "ale", "bbaba"],
     1: [53, 48, 0, 0],
     2: [56, 53, 0, 0],
     3: [54, 49, 0, 0],
     4: [53, 48, 0, 0],
     5: [50, 45, 0, 0]}
testing = Albus(dict_data)
print(testing.null_info)
