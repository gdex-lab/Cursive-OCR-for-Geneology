import pandas as pd
import os
path = os.getcwd() + "\\n_letters.txt"
df = pd.read_csv(path, delimiter='/')

print(df.X)
