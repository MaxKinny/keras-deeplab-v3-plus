import numpy as np
import pandas as pd
import glob
from tqdm import tqdm_notebook
from matplotlib import pyplot as plt
import sys
import pydicom
from mask_functions import rle2mask, mask2rle
import keras as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.callbacks import TensorBoard
from model import Deeplabv3
from keras import backend as K
from accum_optimizer import AccumOptimizer
from keras.optimizers import Adam
from keras.utils import multi_gpu_model


def show_dcm_info(dataset):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)

    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)


def plot_pixel_array(dataset, figsize=(10, 10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


if __name__ == '__main__':
    basestr = 'deeplabv3P'
    im_height = 1024
    im_width = 1024
    im_chan = 1
    #################################
    #########Dataset#################
    #################################
    # Load Full Dataset
    train_glob = './data/dicom-images-train/*/*/*.dcm'
    test_glob = './data/dicom-images-test/*/*/*.dcm'
    train_fns = sorted(glob.glob(train_glob))
    test_fns = sorted(glob.glob(test_glob))
    # for file_path in train_fns:
    #     dataset = pydicom.dcmread(file_path)
    #     show_dcm_info(dataset)
    #     plot_pixel_array(dataset)
    #     break  # Comment this out to see all
    df_full = pd.read_csv('../data/train-rle.csv', index_col='ImageId')
    # Get train images and masks
    X_train = np.zeros((len(train_fns), im_height, im_width, im_chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_fns), im_height, im_width, im_chan), dtype=np.bool)
    print('Getting train images and masks ... ')
    sys.stdout.flush()
    for n, _id in tqdm_notebook(enumerate(train_fns), total=len(train_fns)):
        dataset = pydicom.read_file(_id)
        X_train[n] = np.expand_dims(dataset.pixel_array, axis=2)
        try:
            if '-1' in df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels']:
                Y_train[n] = np.zeros((1024, 1024, 1))
            else:
                if type(df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels']) == str:
                    Y_train[n] = np.expand_dims(
                        rle2mask(df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels'], 1024, 1024), axis=2)
                else:
                    Y_train[n] = np.zeros((1024, 1024, 1))
                    for x in df_full.loc[_id.split('/')[-1][:-4], ' EncodedPixels']:
                        Y_train[n] = Y_train[n] + np.expand_dims(rle2mask(x, 1024, 1024), axis=2)
        except KeyError:
            print(f"Key {_id.split('/')[-1][:-4]} without mask, assuming healthy patient.")
            Y_train[n] = np.zeros((1024, 1024, 1))  # Assume missing masks are empty masks.
    print('Done!')
    # Build Patches
    # Reshape to get non-overlapping patches.
    divide_num = 2
    im_reshape_height = im_height // divide_num
    im_reshape_width = im_width // divide_num
    X_train = X_train.reshape((-1, im_reshape_height, im_reshape_width, 1))
    Y_train = Y_train.reshape((-1, im_reshape_height, im_reshape_width, 1))

    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.imshow(X_train[0, :, :, 0], cmap=plt.cm.bone)
    # plt.subplot(2, 2, 2)
    # plt.imshow(X_train[1, :, :, 0], cmap=plt.cm.bone)
    # plt.subplot(2, 2, 3)
    # plt.imshow(X_train[2, :, :, 0], cmap=plt.cm.bone)
    # plt.subplot(2, 2, 4)
    # plt.imshow(X_train[3, :, :, 0], cmap=plt.cm.bone)
    # plt.show()

    #################################
    #########Training Part###########
    #################################
    file_path = "vgg_face_" + basestr + ".h5"
    # for saving the checkpoint
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    # adaptively change the learning rate
    reduce_on_plateau = ReduceLROnPlateau(monitor="val_acc", mode="max", factor=0.1, patience=20, verbose=1)

    tbCallBack = TensorBoard(log_dir='./logs/' + basestr,
                             histogram_freq=0,
                             write_graph=True,
                             write_images=True)

    callbacks_list = [checkpoint, reduce_on_plateau, tbCallBack]

    model = Deeplabv3(weights=None, input_shape=(im_reshape_height, im_reshape_width, im_chan), classes=1)
    accum_factor = 1
    # opt = AccumOptimizer(Adam(), accum_factor)  # accumulation gradient(soft batch)
    model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=[dice_coef, 'acc', 'mse'])
    model.summary()
    model.fit(X_train, Y_train, validation_split=.2, batch_size=8, epochs=30*accum_factor, callbacks=callbacks_list)

    #################################
    #########Testing Part############
    #################################
    test_path = "../input/test/"


# # Generates labels using most basic setup.  Supports various image sizes.  Returns image labels in same format
# # as original image.  Normalization matches MobileNetV2
#
# trained_image_width=512
# mean_subtraction_value=127.5
# image = np.array(Image.open('imgs/image1.jpg'))
#
# # resize to max dimension of images from training dataset
# w, h, _ = image.shape
# ratio = float(trained_image_width) / np.max([w, h])
# resized_image = np.array(Image.fromarray(image.astype('uint8')).resize((int(ratio * h), int(ratio * w))))
#
# # apply normalization for trained dataset images
# resized_image = (resized_image / mean_subtraction_value) - 1.
#
# # pad array to square image to match training images
# pad_x = int(trained_image_width - resized_image.shape[0])
# pad_y = int(trained_image_width - resized_image.shape[1])
# resized_image = np.pad(resized_image, ((0, pad_x), (0, pad_y), (0, 0)), mode='constant')
#
# # make prediction
# deeplab_model = Deeplabv3(backbone='xception')
# res = deeplab_model.predict(np.expand_dims(resized_image, 0))
# labels = np.argmax(res.squeeze(), -1)
#
# # remove padding and resize back to original image
# if pad_x > 0:
#     labels = labels[:-pad_x]
# if pad_y > 0:
#     labels = labels[:, :-pad_y]
# labels = np.array(Image.fromarray(labels.astype('uint8')).resize((h, w)))
#
# plt.imshow(labels)
# plt.waitforbuttonpress()


