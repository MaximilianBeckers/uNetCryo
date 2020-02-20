import numpy as np
from tensorflow import keras
import model
from dataGenerator import DataGenerator
import mrcfile

# Parameters
params = {'dim': (64, 64, 64),
          'batch_size': 2,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = {'train': ['emd_20642', 'emd_10617', 'emd_20739', 'emd_20277', 'emd_20278', 'emd_3061', 'emd_9333', 'emd_0500', 'emd_4469', 'emd_10652', 'emd_2842', 'emd_10651', 'emd_0086', 'emd_0087', 'emd_0993', 'emd_20911', 'emd_20912',
                       'emd_20753', 'emd_5778', 'emd_21144', 'emd_0025', 'emd_20839', 'emd_4518', 'emd_20101', 'emd_0501', 'emd_10092', 'emd_10502', 'emd_10501', 'emd_2984', 'emd_9779', 'emd_9625', 'emd_30003', 'emd_20913', 'emd_10579',
                       'emd_20642_f0', 'emd_10617_f0', 'emd_20739_f0', 'emd_20277_f0', 'emd_20278_f0', 'emd_3061_f0', 'emd_9333_f0', 'emd_0500_f0', 'emd_4469_f0', 'emd_10652_f0', 'emd_2842_f0', 'emd_10651_f0', 'emd_0086_f0', 'emd_0087_f0', 'emd_0993_f0', 'emd_20911_f0', 'emd_20912_f0',
                       'emd_20753_f0', 'emd_5778_f0', 'emd_21144_f0', 'emd_0025_f0', 'emd_20839_f0', 'emd_4518_f0', 'emd_20101_f0', 'emd_0501_f0', 'emd_10092_f0', 'emd_10502_f0', 'emd_10501_f0', 'emd_2984_f0', 'emd_9779_f0', 'emd_9625_f0','emd_30003_f0', 'emd_20913_f0','emd_10579_f0',
                       'emd_20642_f1', 'emd_10617_f1', 'emd_20739_f1', 'emd_20277_f1', 'emd_20278_f1', 'emd_3061_f1','emd_9333_f1', 'emd_0500_f1', 'emd_4469_f1', 'emd_10652_f1', 'emd_2842_f1', 'emd_10651_f1', 'emd_0086_f1', 'emd_0087_f1','emd_0993_f1', 'emd_20911_f1','emd_20912_f1',
                       'emd_20753_f1', 'emd_5778_f1', 'emd_21144_f1', 'emd_0025_f1', 'emd_20839_f1', 'emd_4518_f1', 'emd_20101_f1', 'emd_0501_f1', 'emd_10092_f1', 'emd_10502_f1', 'emd_10501_f1', 'emd_2984_f1', 'emd_9779_f1', 'emd_9625_f1', 'emd_30003_f1','emd_20913_f1','emd_10579_f1',
                       'emd_20642_f2', 'emd_10617_f2', 'emd_20739_f2', 'emd_20277_f2', 'emd_20278_f2', 'emd_3061_f2', 'emd_9333_f2', 'emd_0500_f2', 'emd_4469_f2', 'emd_10652_f2', 'emd_2842_f2', 'emd_10651_f2', 'emd_0086_f2', 'emd_0087_f2', 'emd_0993_f2', 'emd_20911_f2', 'emd_20912_f2',
                       'emd_20753_f2', 'emd_5778_f2', 'emd_21144_f2', 'emd_0025_f2', 'emd_20839_f2', 'emd_4518_f2', 'emd_20101_f2', 'emd_0501_f2', 'emd_10092_f2', 'emd_10502_f2', 'emd_10501_f2', 'emd_2984_f2', 'emd_9779_f2', 'emd_9625_f2', 'emd_30003_f2','emd_20913_f2', 'emd_10579_f2'],
             'validation': ['emd_21144']}

# Generators
training_generator = DataGenerator(partition['train'], **params)
validation_generator = DataGenerator(partition['validation'], **params)

# take the specified model
model = model.uNet()
model_checkpoint = keras.callbacks.ModelCheckpoint('uNetCryo.hdf5', monitor='loss', verbose=1, save_best_only=False)


# Train model on dataset
model.fit(training_generator,
                    #validation_data=validation_generator,
                    use_multiprocessing=True,
                    shuffle=True,
                    epochs=100,
                    callbacks=[model_checkpoint]);

#model.save('uNetCryo.hdf5')


#***********************
#***** prediction ******
#***********************

#read test data
hafl1_mrc = mrcfile.open("data/maps/train/half1/" + "emd_3061" + ".map", mode='r');
half1 = np.copy(hafl1_mrc.data);

sizeMap = half1.shape;
start = (np.array(sizeMap) / 2) - int(64 / 2);
start = start.astype(int);
half1 = half1[start[0]:start[0] + 64, start[1]:start[1] + 64, start[2]:start[2] + 64];

#save the prediction
orig_mrc = mrcfile.new("orig.map", overwrite=True);
orig_mrc.set_data(np.float32(half1));
orig_mrc.voxel_size = 1.0;
orig_mrc.close();

half1 = np.expand_dims(half1, axis=0);
half1 = np.expand_dims(half1, axis=4);

#generate prediction
prediction = model.predict(half1);

prediction = np.squeeze(prediction)

#save the prediction
pred_mrc = mrcfile.new("pred.map", overwrite=True);
pred_mrc.set_data(np.float32(prediction));
pred_mrc.voxel_size = 1.0;
pred_mrc.close();
