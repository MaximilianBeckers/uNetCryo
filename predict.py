import numpy as np
from tensorflow import keras
import mrcfile

#***********************
#***** prediction ******
#***********************

# Recreate the exact same model, including its weights and the optimizer
model = keras.models.load_model('uNetCryo.hdf5');

#read the map, which is to be denoised
map_mrc = mrcfile.open("data/maps/test/" + "emd_2984" + ".map", mode='r');
map = np.copy(map_mrc.data);

apix = float(map_mrc.voxel_size.x);

size_map = map.shape;
size_window = min(np.min(size_map), 120);

map = np.pad(map, ((0 ,size_window - (size_map[0]%size_window)), (0, size_window - (size_map[1]%size_window)), (0, size_window - (size_map[2]%size_window))), 'constant');
size_map_new = map.shape;

denoised_map = np.zeros(size_map_new);

steps_x = int(size_map_new[0]/size_window);
steps_y = int(size_map_new[1]/size_window);
steps_z = int(size_map_new[2]/size_window);


#*************************************
#*** loop over the map and denoise ***
#*************************************

for x in range(steps_x):
	for y in range(steps_y):
		for z in range(steps_z):

			print(x);

			#extract local window
			tmp_window = map[x*size_window:(x+1)*size_window, y*size_window:(y+1)*size_window, z*size_window:(z+1)*size_window];

			# add nuisance dimensions
			tmp_window = np.expand_dims(tmp_window, axis=0);
			tmp_window = np.expand_dims(tmp_window, axis=4);

			# generate prediction
			prediction = model.predict(tmp_window);

			# delete nuisance dimensions
			prediction = np.squeeze(prediction);

			#save prediction
			denoised_map[x*size_window:(x+1)*size_window, y*size_window:(y+1)*size_window, z*size_window:(z+1)*size_window] = prediction;

#unpad denoised map
denoised_map = denoised_map[:size_map[0], :size_map[1], :size_map[2]];

#save the prediction
denoised_mrc = mrcfile.new("denoised.map", overwrite=True);
denoised_mrc.set_data(np.float32(denoised_map));
denoised_mrc.voxel_size = apix;
denoised_mrc.close();




