import numpy as np

def clip_gradients_by_value(gradients: np.ndarray, clip_value: float) -> np.ndarray:
    """
    Clip gradient values to be within [-clip_value, clip_value].
    
    Args:
        gradients: A numpy array representing gradients (any shape)
        clip_value: The maximum absolute value for any gradient element (non-negative)
    
    Returns:
        Clipped gradients with same shape as input
    """
    # Preserv the original gradients so make a copy
    new_grad = gradients.copy()
    flat = new_grad.ravel() # its gives 1D view 
    for i in range(flat.size):
        if flat[i] > clip_value:
            flat[i] = clip_value
        elif flat[i] < (-clip_value):
            flat[i] = (-clip_value)
        
           
    return new_grad