import serial
import os
import random
import cv2
import numpy as np
import struct
import time
import sys
def main():
	args = sys.argv[1:]
	if len(args) == 2 and args[0] == '-port':
		port = str(args[1])	

	port = "/dev/ttyUSB1"
	dims = (64,48) # dimensions of images to train/test with

	randomint = random.randrange(10)
	read_dir = '/home/jakub/Desktop/SDU/masters/sem_1/embedded/MiniProject/FPGA_AI/Dataset_new/Philips/'
	read_file = random.choice(os.listdir(read_dir)) # choose random test image
	img = cv2.imread(os.path.join(read_dir,read_file),0) # read img as grayscale
	img = cv2.resize(img, dims, interpolation = cv2.INTER_AREA)	# resize img to fit dims
	img = np.asarray((img / 255)).astype('float32')
	print("Label: ", str(randomint), " Filename: ", read_file, " Serialport: ", port) # print test image label
	# define serial connection
	ser = serial.Serial(port, 115200, serial.EIGHTBITS, serial.PARITY_NONE, serial.STOPBITS_ONE)

	counter = 0

	for i in range(dims[1]):
		for j in range(dims[0]):
			values = bytearray(struct.pack("f", img[i][j])) # turn pixel values into bytearray
			
			counter += 1
			print(counter)

			if len(values) != 4:
				print("STOP")

				raise(Exception)
			
			time.sleep(0.02)
			ser.write(values) # send bytearray over UART
			
			# nn_res = ser.readline().decode('UTF-8') # decode received bytes
			# print(nn_res) # can be used to display prints from U96
			
			# nn_res = ser.readline().decode('UTF-8') # decode received bytes
			# print(nn_res) # can be used to display prints from U96

	nn_res = ""
	while "output" not in nn_res: # check if nn output received
		nn_res = ser.readline().decode('UTF-8') # decode received bytes
		print(nn_res) # can be used to display prints from U96
	print(nn_res)
	
	
if __name__=="__main__":
    main()
