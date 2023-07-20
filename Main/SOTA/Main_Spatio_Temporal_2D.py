import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from Architectures.SOTA.Architecture_Spatio_Temporal_2D import spatio_temporal_network
from Data_preprocessing.CropImages import img_crop_v2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import re

x_dim = 64
y_dim = 64
nb_slices = 10
nb_frames = 20
z_dim = nb_slices * nb_frames
input_shape = (x_dim, y_dim, z_dim, 1)

batch_size = 64
nb_epochs = 500

seed_num = 1964
os.environ['PYTHONHASHSEED'] = str(seed_num)
np.random.seed(seed_num)

model_name = ""
path_to_save_weights = ""
checkpoint = ModelCheckpoint(filepath=path_to_save_weights, save_best_only=True, monitor="val_loss", mode="min", verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=70, mode="min")
model = spatio_temporal_network(64, 64, 200, 1)
model.summary()

training_data = pd.read_excel("")
list_of_patients = training_data["Patient_ID"].values
path_to_images = ""

x_train = []
y_train = []

print("Loading training dataset")
for patient in tqdm(list_of_patients):
    list_of_slices = os.listdir(path_to_images + patient)
    if (len(list_of_slices) < nb_slices):
        continue
    list_of_slices.sort(key=lambda f: int(re.sub('\D', '', f)))
    temp = []

    for slice in list_of_slices[:nb_slices]:
        i = 1
        list_of_images = os.listdir(path_to_images + patient + "/" + slice)
        list_of_images.sort(key=lambda f: int(re.sub('\D', '', f)))
        for image in list_of_images[:nb_frames]:
            im = np.load(path_to_images + patient + "/" + slice + "/" + str(i) + ".npy")
            im = img_crop_v2(im, x_dim, y_dim)
            im = im / im.max()
            temp.append(im)
            i += 1

    temp = np.asarray(temp)
    if (temp.shape != (z_dim, x_dim, y_dim)):
        continue

    temp = np.asarray(temp)
    x_train.append(temp)
    y_train.append(float(training_data.loc[training_data["Patient_ID"] == patient]["Scar"].values))

print("Loading validation dataset")
validation_data = pd.read_excel("")
list_of_patients = validation_data["Patient_ID"].values
x_validation = []
y_validation = []
for patient in tqdm(list_of_patients):
    list_of_slices = os.listdir(path_to_images + patient)
    if (len(list_of_slices) < nb_slices):
        continue
    list_of_slices.sort(key=lambda f: int(re.sub('\D', '', f)))
    temp = []

    for slice in list_of_slices[:nb_slices]:
        list_of_images = os.listdir(path_to_images + patient + "/" + slice)
        list_of_images.sort(key=lambda f: int(re.sub('\D', '', f)))
        i = 1
        for image in list_of_images[:nb_frames]:
            im = np.load(path_to_images + patient + "/" + slice + "/" + str(i) + ".npy")
            im = img_crop_v2(im, x_dim, y_dim)
            im = im / im.max()
            temp.append(im)
            i += 1

    temp = np.asarray(temp)
    if (temp.shape != (z_dim, x_dim, y_dim)):
        continue
    y_validation.append(float(validation_data.loc[validation_data["Patient_ID"] == patient]["Scar"].values))
    x_validation.append(temp)


del temp, im
x_train = np.asarray(x_train, dtype=np.float32)
# print(x_train.shape)
x_train = x_train.transpose(0, 2, 3, 1)
x_train = x_train.reshape(x_train.shape[0], x_dim, y_dim, z_dim, 1)
y_train = np.asarray(y_train, dtype=np.float32)
x_train = shuffle(x_train, random_state=1964)
y_train = shuffle(y_train, random_state=1964)
print("x_train_shape: ", x_train.shape)
print("y_train_shape: ", y_train.shape)
print("Number of patients with scar training: ", len(np.where(y_train==1)[0]))

x_validation = np.asarray(x_validation, dtype=np.float32)
x_validation = x_validation.transpose(0, 2, 3, 1)
x_validation = x_validation.reshape(x_validation.shape[0], x_dim, y_dim, z_dim, 1)
y_validation = np.asarray(y_validation, dtype=np.float32)
print("x_validation: ", x_validation.shape)
print("y_validation: ", y_validation.shape)
print("Number of patients with scar validation: ", len(np.where(y_validation == 1)[0]))

history = model.fit(x_train, y_train, batch_size=batch_size,
                    validation_data=(x_validation, y_validation),
                    epochs=nb_epochs,  verbose=1,
          callbacks=[checkpoint, early_stopping])

def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['accuracy']
    val_acc = hist['val_accuracy']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()
    plt.savefig("")

plot_accuracy_and_loss(history)