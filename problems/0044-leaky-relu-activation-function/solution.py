import numpy as np 
import math
def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	#alpha = 0.01
	if z > 0.0:
		relu = max(0.0, z)
	else:
		relu = alpha * (z)		
	return relu
