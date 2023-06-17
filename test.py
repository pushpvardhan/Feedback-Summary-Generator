import numpy as np
import pandas as pd

df = pd.read_excel("Peer Form 1.xlsx")
# print(df.to_string())
df1 = df.drop(['Options'], axis=1)
# print(df1.to_string())

x = np.array([df1['Criteria'], df1['Select Option'], df1['Cite Examples and Additional Comments - As Applicable.']])
x = x.transpose()
var = x.shape
print(var)
