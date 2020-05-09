import os, cv2, csv
import numpy as np
import pandas as pd
from core_arcface_cleaning import generate_distance 

image1 = './dataset/player1.jpg'
image2 = './dataset/player2.jpg'

# image1 = cv2.imread(image_path1)
# image2 = cv2.imread(image_path2)

embeddings = generate_distance(image1, image2)

header = []
for i in range(512):
    header.append('ftr '+str(i))
header.append('person_id')

database_df = pd.DataFrame(columns=header)
pd.DataFrame(database_df).to_csv("./csv/test.csv")

for index, embedding in enumerate(embeddings):
    embedding = np.append(embedding, "luqmanr")

    array_df = pd.DataFrame(columns=embedding)
    pd.DataFrame(array_df).to_csv("./csv/test.csv", mode='a', index_label=index)

def read_csv():
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
                data[i] = float(data[i])
            
            test_array = np.array(data)

    print('read array=', test_array)

read_csv()