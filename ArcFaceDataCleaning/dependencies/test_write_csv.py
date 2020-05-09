import os, csv
import numpy as np
import pandas as pd

test_array = np.arange(0,512)
print('original array=', type(test_array))
test_array = np.append(test_array, "luqmanr")

header = []
for i in range(512):
    header.append('ftr '+str(i))
header.append('person_id')

database_df = pd.DataFrame(columns=header)
pd.DataFrame(database_df).to_csv("./csv/test.csv")

array_df = pd.DataFrame(columns=test_array)
pd.DataFrame(array_df).to_csv("./csv/test.csv", mode='a', index_label=True)

with open('./csv/test.csv', newline='') as f:
    reader = csv.reader(f)
    datas = list(reader)

for index, data in enumerate(datas):
    if index == 0:
        pass
    else:
        data.pop(0)
        data.pop(-1)

        for i in range(len(data)):
            data[i] = int(data[i])
        
        test_array = np.array(data)

print('read array=', test_array)