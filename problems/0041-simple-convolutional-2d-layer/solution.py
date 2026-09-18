import numpy as np
import math

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape

	# Your code here
	#create padded input image first
	padded_image = np.pad(input_matrix,pad_width=padding,mode = 'constant',constant_values =0)

	#calculate the output dimension
	output_height = math.floor((input_height + 2*padding- kernel_height)// (stride )+1)
	output_width = math.floor((input_width + 2* padding - kernel_width)//(stride) +1)

	# Now we have dim of output input_matrix
	#Create output matrix with all zeros to start with
	output_matrix = np.zeros((output_height,output_width))

	# Perform convolution
	# element wise multiplication
	for i in range(output_height):
		for j in range(output_width):
			row = i * stride
			col = j * stride
			region = padded_image[row:row + kernel_height,col:col +  kernel_width]
			output_matrix[i,j] = np.sum( region * kernel)


	return output_matrix
