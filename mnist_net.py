import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import tensorflow as tf
import struct
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import sys

def main():
	args = sys.argv[1:]
	if len(args) == 2 and args[0] == '-dataset_dir':
		dataset_dir = str(args[1])	

	## Use CPU only
	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

	## Load MNIST dataset
	print("Loading dataset")
	images = []
	labels = []

	dims = (64,48) # dimensions of images to train/test with

	labels_map = ["Philips", "Flat", "Hole"]

	for i in (0, 1, 2):
		read_folder = dataset_dir + '/' + labels_map[i] + '/'

		for filename in os.listdir(read_folder):
			img = cv2.imread(os.path.join(read_folder,filename),0) # read img as grayscale
			img = cv2.resize(img, dims, interpolation = cv2.INTER_AREA)	# resize img to fit dims
			if img is not None:
				images.append(img / 255) # normalize pixel vals to be between 0 - 1
				labels.append(i)

	# ## Convert to numpy arrays, flatten images - change dimensions from Nx10x10 to Nx100
	images_np = np.asarray(images).astype('float32')
	labels_np = np.asarray(labels).astype('uint8')

	# ## Shuffle dataset
	# train_images, train_labels = shuffle(train_images, train_labels)
	# test_images, test_labels = shuffle(test_images, test_labels)

	#Perform the train-test split
	X_train, X_test, y_train, y_test = train_test_split(images_np, labels_np, test_size=0.33, random_state=69)

	## Define network structure
	model = Sequential([
		Flatten(input_shape=dims),		# reshape 10x10 to 100, layer 0
		Dense(32, activation='relu', use_bias=False),	# dense layer 1
		Dense(16, activation='relu', use_bias=False),	# dense layer 2
		Dense(3, activation='softmax', use_bias=False),	# dense layer 3
	])

	model.compile(optimizer='adam',
				  loss='sparse_categorical_crossentropy',
				  metrics=['accuracy'])


	## Train network  
	model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split = 0.1)

	model.summary()

	start_t = time.time()
	results = model.evaluate(X_test, y_test, verbose=0)
	totalt_t = time.time() - start_t
	print("Inference time for ", len(y_test), " test image: " , totalt_t, " seconds")


	print("test loss, test acc: ", results)

	#print(model.layers[1].weights[0].numpy().shape)
	#print(model.layers[2].weights[0].numpy().shape)
	#print(model.layers[3].weights[0].numpy().shape)

	## Retrieve network weights after training. Skip layer 0 (input layer)
	for w in range(1, len(model.layers)):
		weight_filename = "layer_" + str(w) + "_weights.txt" 
		open(weight_filename, 'w').close() # clear file
		file = open(weight_filename,"a") 
		file.write('{')
		for i in range(model.layers[w].weights[0].numpy().shape[0]):
			file.write('{')
			for j in range(model.layers[w].weights[0].numpy().shape[1]):
				file.write(str(model.layers[w].weights[0].numpy()[i][j]))
				if j != model.layers[w].weights[0].numpy().shape[1]-1:
					file.write(', ')
			file.write('}')
			if i != model.layers[w].weights[0].numpy().shape[0]-1:
				file.write(', \n')
		file.write('}')
		file.close()

	network_weights = model.layers[1].weights
	#print(network_weights)
	layer_1_W = network_weights[0].numpy()
	#print(layer_1_W)




	
	"""img_filename = "img_pixel_vals.txt" 
	open(img_filename, 'w').close() # clear file
	file = open(img_filename,"a") 
	file.write('{')
	for i in range(dims[1]):
		for j in range(dims[0]):
			file.write(str(test_images[0][i][j]))
			if j != dims[1]-1:
				file.write(', ')
		if i != dims[0]-1:
			file.write(', \n')
	file.write('}')
	file.close()"""


	"""img_filename = "img_pixel_vals_vhdl_array.txt" 
	open(img_filename, 'w').close() # clear file
	file = open(img_filename,"a") 
	file.write('(')
	for i in range(dims[1]):
		for j in range(dims[0]):
			file.write('"')
			wstr = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', test_images[0][i][j]))
			file.write(wstr)
			file.write('"')
			if j != dims[1]-1:
				file.write(', ')
		if i != dims[0]-1:
			file.write(', \n')
	file.write(')')
	file.close()"""


	"""img_filename = "img_pixel_vals.coe" 
	open(img_filename, 'w').close() # clear file
	file = open(img_filename,"a") 
	file.write('memory_initialization_radix=2;\n') # radix 2 = binary, radix 10 = decimal
	file.write('memory_initialization_vector=\n')
	for i in range(dims[1]):
		for j in range(dims[0]):
			wstr = ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', test_images[0][i][j]))
			file.write(wstr)
			if i == dims[0]-1 and j == dims[1]-1:
				file.write(';')
			else:
				file.write(',\n')
	file.close()"""

	print("test_image[0] label: ", y_test[0])

	x = X_test[0]
	x = np.expand_dims(x, axis=0)
	print("NN Prediction: ", np.argmax(model.predict(x)))


	print("Finished")

	# img = cv2.imread(r"/home/jakub/Desktop/SDU/masters/sem_1/embedded/MiniProject/401090579_1243895233150632_5627371333467907964_n.jpg",0) # read img as grayscale
	# img = cv2.resize(img, dims, interpolation = cv2.INTER_AREA)	# resize img to fit dims
	# if img is not None:
	# 	img = img / 255 # normalize pixel vals to be between 0 - 1

	# img = np.expand_dims(img, axis=0)
	
	# print("oUR Prediction: ", model.predict(img))
	
	
	
if __name__=="__main__":
    main()
