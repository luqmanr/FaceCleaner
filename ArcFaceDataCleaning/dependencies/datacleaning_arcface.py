''' Run Docker
nvidia-docker run --rm -it -d \
-v /home/luqmanr/.Xauthority:/home/luqmanr/.Xauthority \
-v /tmp/.X11-unix/:/tmp/.X11-unix \
-e DISPLAY=${DISPLAY} \
--shm-size=12g \
--name=face-research \
--privileged \
-v /mnt/Data/RKB-Face-Git/FaceCleaner:/workspace/ \
-v /mnt/Data/:/mnt/Data/ risetai/research:face-recognition
'''

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
    train_index = 0
    raw_index = 0
    for file in files:
        if "train" in str.lower(file):
            src = os.path.join(input_path, file)
            dst = os.path.join(dst_folder_reference, 'train_{}.jpg'.format(str(train_index)))

            ##copy 'train' files to new destination
            shutil.copy(src, dst)
            train_index += 1
        else:
            src = os.path.join(input_path, file)
            dst = os.path.join(dst_folder_raw, 'raw_{}.jpg'.format(str(raw_index)))

            ##copy non 'train' files to new destination
            shutil.copy(src, dst)
            raw_index += 1

def generate_csv(input_path, output_path):
    csv_modes = ['train', 'raw']
    for csv_mode in csv_modes:
        folder_ID = os.path.basename(input_path)
        csv_folder = os.path.join(output_path, 'csv_{}'.format(csv_mode))
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
        img_name = os.path.basename(reference_image)
        try:
            embedding = core_arcface_cleaning.generate_embedding(reference_image)
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
        img_name = os.path.basename(raw_image)
        try:
            embedding = core_arcface_cleaning.generate_embedding(raw_image)
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

    ## locate the csv files
    csv_train_folder = os.path.join(output_path, 'csv_train')
    csv_train_path = os.path.join(csv_train_folder, (folder_ID + '.csv'))

    csv_raw_folder = os.path.join(output_path, 'csv_raw')
    csv_raw_path = os.path.join(csv_raw_folder, (folder_ID + '.csv'))

    ## read the csv files
    reference_embeddings, reference_img = read_csv_db(csv_train_path)
    raw_embeddings, raw_img = read_csv_db(csv_raw_path)

    similar_imgs = []

    for index_raw, raw_embedding in enumerate(raw_embeddings):
        print('processing', raw_img[index_raw], '===================')
        sim_list = []
        for index_ref, reference_embedding in enumerate(reference_embeddings):
            # Compute squared distance between embeddings
            dist = np.sum(np.square(reference_embedding - raw_embedding))
            # Compute cosine similarity between embedddings
            sim = np.dot(reference_embedding, raw_embedding.T)

            if sim > 0.5:
                print('SIMILAR IMAGE', raw_img[index_raw])
                # Print predictions
                print('image_ref =', reference_img[index_ref])
                print('image_raw =', raw_img[index_raw])
                # print('Distance = %f' %(dist))
                print('Similarity = %f' %(sim), '\n')
                sim_list.append(sim)
                # similar_imgs.append(raw_img[index_raw])
            else:
                print(raw_img[index_raw], 'not similar enough to', reference_img[index_ref])
                print('Similarity = %f' %(sim), '\n')

        if len(sim_list) > 0:
            average_sim = sum(sim_list) / len(sim_list)
            print('average_sim for', raw_img[index_raw], 'is =', average_sim)
            if average_sim > 0.5:
                similar_imgs.append(raw_img[index_raw])

    similar_imgs = list(set(similar_imgs))
    return similar_imgs

def move_similar_to_train(input_path, output_path, similar_imgs):
    folder_ID = os.path.basename(input_path)
    folder_reference = os.path.join(output_path, 'reference', folder_ID)
    folder_raw = os.path.join(output_path, 'raw', folder_ID)

    folder_dst = os.path.join(output_path, 'reference', folder_ID)
    # if not os.path.exists(folder_dst):
    #     os.makedirs(folder_dst)

    train_index = len(os.listdir(folder_reference))

    for similar_img in similar_imgs:
        src = os.path.join(folder_raw, similar_img)
        dst = os.path.join(folder_dst, 'train_{}.jpg'.format(str(train_index)))

        shutil.move(src, dst)
        train_index += 1

def write_new_csv(reference_csv, raw_csv, similar_imgs):
    with open(raw_csv, newline='') as f:
        reader = csv.reader(f)
        embeddings_raw_csv = list(reader)

    ## initialise raw embeddings that are going to be train embeddings
    new_reference_embeddings = []
    new_image_refs = []
    ## the rest of embeddings that aren't going to be train embeddings are going to be written back to the raw csv
    new_reference_embeddings = []
    new_image_refs = []

    for index, embedding in enumerate(embeddings_raw_csv):
        if index == 0:
            pass
        else:
            if embedding[-1] in similar_imgs:
                new_image_refs.append(embedding[-1])
                embedding.pop(0)
                embedding.pop(-1)

                for i in range(len(embedding)):
                    embedding[i] = float(embedding[i])
                
                embedding = np.array(embedding)
                new_reference_embeddings.append(embedding)


print('======================================================')   
print('TESTING GROUND')

to_scan_path = '/mnt/Data/RKB-Dataset/Instagram_to_process/combined/'
output_path = '/mnt/Data/RKB-Dataset/Instagram_to_process/arcface_testing'

id_paths = os.listdir(to_scan_path)
for id_name in id_paths:
    input_path = os.path.join(to_scan_path, id_name)

    arrange_folder(input_path, output_path)
    for i in range(1):
        generate_csv(input_path, output_path)       
        reference_image_embeddings(input_path, output_path)
        raw_image_embeddings(input_path, output_path)
        similar_imgs = generate_distance(input_path, output_path)
        move_similar_to_train(input_path, output_path, similar_imgs)

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