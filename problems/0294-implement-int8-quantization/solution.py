import numpy as np

def int8_quantize(x: list[float]) -> dict:
	"""
	Perform symmetric INT8 quantization on a floating-point array.
	
	Args:
		x: Input list of floating-point values
		
	Returns:
		Dictionary with 'quantized', 'scale', and 'dequantized' keys
	"""
	#Convert the list into numpy array first
	x_numpy = np.asarray(x,dtype=np.float32)

	#convert all x value into absolute value
	x_abs =np.max(np.abs(x_numpy))
	#lets check the edge case
	if x_abs == 0:
		scale = 1.0
	else:
		scale = x_abs/127.0	
	
	quantized = np.round(x_numpy/scale)
	quantized = np.clip(quantized,-127,127).astype(np.int8)

	#Now dequantize
	dequantized = (quantized.astype(np.float32) * scale)

	return{
		'quantized':quantized.tolist(),
		'scale':round(float(scale),6),
		'dequantized':[round(v,4)for v in dequantized.tolist()],
		#'dequantize':dequantize.tolist()
	}