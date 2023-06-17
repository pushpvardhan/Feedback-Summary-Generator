import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast

torch.device("cuda")
df = pd.read_excel("Peer Form 1.xlsx")
print(df.to_string())
df1 = df.drop(['Options'], axis=1)
print(df1.to_string())

train_criteria, temp_criteria, train_select_option, temp_select_option, train_example, temp_example, train_summary, temp_summary = train_test_split(df1['Criteria'], df1['Select Option'], df1['Cite Examples and Additional Comments - As Applicable.'], df1['Overall Summary'])