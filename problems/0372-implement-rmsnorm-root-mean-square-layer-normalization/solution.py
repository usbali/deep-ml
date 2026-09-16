import numpy as np

def rmsnorm(x: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """
    Apply RMSNorm to the input array.
    
    Parameters:
        x   : np.ndarray of shape (batch_size, features)
        g   : np.ndarray of shape (features,) - gain parameter
        eps : float - small constant for numerical stability
    
    Returns:
        np.ndarray of same shape as x
    """
    #compute the mean
    #rms_mean = np.mean(x**2, axis=1)
    #ValueError: operands could not be broadcast together with shapes (2,4) (2,)
    rms_mean = np.mean(x**2, axis =1,keepdims =True)
    rms_sq_root = np.sqrt(rms_mean +eps)
    RMSNorm = (x/(rms_sq_root)) *g 

    return RMSNorm