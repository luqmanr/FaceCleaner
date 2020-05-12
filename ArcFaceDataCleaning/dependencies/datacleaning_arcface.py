import os, cv2, csv, argparse, shutil
import numpy as np
import pandas as pd
import core_arcface_cleaning
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--input', required=True,
                help='input path to folderID containing "train_n.jpg" image(s)')
ap.add_argument('-o', '--output',
                help='your desired output path, containing processed images \
                      if None specified, will be on "../output/')
args = vars(ap.parse_args())

input_path = '/Data/RKB-Dataset/Instagram_to_process/reference/14anra'
output_path = '/Data/RKB-Dataset/Instagram_to_process/arcface_testing'

def arrange_folder(input_path, output_path):
    files = os.listdir(input_path)
    folder_ID = os.path.basename(input_path)
    dst_folder_reference = os.path.join(output_path, 'reference', folder_ID)
    dst_folder_raw = os.path.join(output_path, 'raw', folder_ID)

    ## make folder for reference and raw images
    if not os.path.exists(dst_folder_reference):
        os.makedirs(dst_folder_reference)
    if not os.path.exists(dst_folder_raw):
        os.makedirs(dst_folder_raw)

    ## if image name contains the string 'train' it will be used as the reference image of that ID
    for index, file in enumerate(files):
        if "train" in str.lower(file):
            src = os.path.join(input_path, file)
            dst = os.path.join(dst_folder_reference, file)

            ##copy 'train' files to new destination
            shutil.copy(src, dst)
        else:
            src = os.path.join(input_path, file)
            dst = os.path.join(dst_folder_raw, 'raw_{}.jpg'.format(str(index)))

            ##copy non 'train' files to new destination
            shutil.copy(src, dst)

def generate_csv(input_path, output_path):
    csv_mode = ['train', 'raw']
    for mode in csv_mode:
        folder_ID = os.path.basename(input_path)
        csv_folder = os.path.join(output_path, 'csv_{}'.format(mode))
        csv_output = os.path.join(csv_folder, (folder_ID + '.csv'))

        if not os.path.exists(csv_folder):
            os.makedirs(csv_folder)

        ## arrange header for csv
        header = []
        for i in range(512):
            header.append('ftr '+str(i)) ## will be header for features
        header.append('img_ref') ## will be path to image where the embedding is from

        database_df = pd.DataFrame(columns=header)
        pd.DataFrame(database_df).to_csv(csv_output)

def reference_image_embeddings(input_path, output_path):
    folder_ID = os.path.basename(input_path)
    folder_reference = os.path.join(output_path, 'reference', folder_ID)
    
    ## get all images within the reference folder
    def get_reference_images():
        reference_images = []
        images = os.listdir(folder_reference)
        for image in images:
            img_path = os.path.join(folder_reference, image)
            reference_images.append(img_path)
        return reference_images

    reference_images = get_reference_images()

    ## Generate embeddings for each images, and write it into a csv
    csv_folder = os.path.join(output_path, 'csv_train')
    csv_output = os.path.join(csv_folder, (folder_ID + '.csv'))

    for reference_image in reference_images:
        try:
            embedding = core_arcface_cleaning.generate_embedding(reference_image)
            img_name = os.path.basename(reference_image)
            embedding = np.append(embedding, img_name)
            write_embedding_to_csv(embedding, csv_output)
        except:
            print('failed to generate embedding for', img_name)
            pass

def write_embedding_to_csv(embedding, csv_output):
    array_df = pd.DataFrame(columns=embedding)
    pd.DataFrame(array_df).to_csv(csv_output, mode='a', index=True)

def raw_image_embeddings(input_path, output_path):
    folder_ID = os.path.basename(input_path)
    folder_raw = os.path.join(output_path, 'raw', folder_ID)
    
    ## get all images within the reference folder
    def get_raw_images():
        raw_images = []
        images = os.listdir(folder_raw)
        for image in images:
            img_path = os.path.join(folder_raw, image)
            raw_images.append(img_path)
        return raw_images

    raw_images = get_raw_images()

    ## Generate embeddings for each images, and write it into a csv
    csv_folder = os.path.join(output_path, 'csv_raw')
    csv_output = os.path.join(csv_folder, (folder_ID + '.csv'))

    for raw_image in raw_images:
        try:
            embedding = core_arcface_cleaning.generate_embedding(raw_image)
            img_name = os.path.basename(raw_image)
            embedding = np.append(embedding, img_name)
            write_embedding_to_csv(embedding, csv_output)
        except:
            print('failed to generate embedding for', img_name)
            pass

def read_csv_db(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        embeddings = list(reader)

    reference_embeddings = []
    image_refs = []
    for index, embedding in enumerate(embeddings):
        if index == 0:
            pass
        else:
            image_refs.append(embedding[-1])
            embedding.pop(0)
            embedding.pop(-1)

            for i in range(len(embedding)):
                embedding[i] = float(embedding[i])
            
            embedding = np.array(embedding)
            reference_embeddings.append(embedding)
    
    return reference_embeddings, image_refs

def generate_distance(input_path, output_path):
    folder_ID = os.path.basename(input_path)
    csv_folder = os.path.join(output_path, 'csv')
    csv_path = os.path.join(csv_folder, (folder_ID + '.csv'))

    raw_embeddings = raw_image_embeddings(input_path, output_path)
    reference_embeddings, image_refs = read_csv_db(csv_path)
    for index, raw_embedding in enumerate(raw_embeddings):
        for reference_embedding in reference_embeddings:
            # Compute squared distance between embeddings
            dist = np.sum(np.square(raw_embedding-reference_embedding))
            # Compute cosine similarity between embedddings
            sim = np.dot(raw_embedding, reference_embedding.T)
            # Print predictions
            print('image_ref =', image_refs[index])
            print('Distance = %f' %(dist))
            print('Similarity = %f' %(sim))

print('======================================================')   
print('TESTING GROUND') 
# arrange_folder(input_path, output_path) 
generate_csv(input_path, output_path)       
# reference_image_embeddings(input_path, output_path)
raw_image_embeddings(input_path, output_path)
# generate_distance(input_path, output_path)

'''
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
'''