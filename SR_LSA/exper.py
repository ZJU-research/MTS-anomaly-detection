import numpy
import os
import pandas as pd
df = pd.DataFrame()
print(df)
a = numpy.zeros(12)
df = df.append(pd.Series(a.tolist()), ignore_index=True)
b = numpy.arange(6).tolist()
c = [10]
print(b + c)