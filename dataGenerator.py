import numpy as np
#import keras
from tensorflow import keras
import mrcfile

class DataGenerator(keras.utils.Sequence):

    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=2, dim=(None, None, None), n_channels=1, shuffle=True):

        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        #'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.n_channels))
        Y = np.empty((self.batch_size, self.dim[0], self.dim[1], self.dim[2], self.n_channels))


        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            if ID[-3:] == '_f0':
                flip = True;
                flip_axis = 0;
                ID = ID[:-3];
            elif ID[-3:] == '_f1':
                flip = True;
                flip_axis = 1;
                ID = ID[:-3];
            elif ID[-3:] == '_f2':
                flip = True;
                flip_axis = 2;
                ID = ID[:-3];
            else:
                flip = False;

            # Store halfmap1
            hafl1_mrc = mrcfile.open("data/maps/train/half1/" + ID + ".map", mode='r')
            half1 = np.copy(hafl1_mrc.data)
            sizeMap = half1.shape;
            start = (np.array(sizeMap)/2) - int(self.dim[0]/2);
            start = start.astype(int);
            tmpX = half1[start[0]:start[0]+self.dim[0], start[1]:start[1]+self.dim[0], start[2]:start[2]+self.dim[0]] ;
            tmpX = np.expand_dims(tmpX, axis=3);


            # Store halfmap2
            half2_mrc = mrcfile.open("data/maps/train/half2/" + ID + ".map", mode='r')
            half2 = np.copy(half2_mrc.data)
            tmpY = half2[start[0]:start[0]+self.dim[0], start[1]:start[1]+self.dim[0], start[2]:start[2]+self.dim[0]] ;
            tmpY = np.expand_dims(tmpY, axis=3)


            if flip:
                X[i,] = np.flip(tmpX, flip_axis);
                Y[i,] = np.flip(tmpY, flip_axis);
            else:
                X[i,] = tmpX;
                Y[i,] = tmpY;


        return X, Y
