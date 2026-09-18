import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.
    
    Parameters:
    n_in: number of input units
    n_out: number of output units
    mode: 'fan_in' or 'fan_out'
    distribution: 'normal' or 'uniform'
    seed: random seed for reproducibility
    
    Returns:
    numpy array of shape (n_in, n_out) with He-initialized weights
    """
    if seed is not None:
        np.random.seed(seed)
    #its about weight initialization
    if mode == 'fan_in':
        fan = n_in
    elif mode == 'fan_out':
        fan = n_out
    else:
        raise ValueError(f" mode need to be either fan_in or fan_out, got '{mode}")
    #for He we take 2/fan    
    variance = 2/fan    
    # now depending on type of distribution 
    #Normal or unifrom find standard deviation and weights 
    if distribution == 'normal':
        std =np.sqrt(variance)
        weights = np.random.normal(0.0, std, size=(n_in,n_out))
    else: #distribution == 'uniform':
        # for unofrom variance = a**2/3
        bound = np.sqrt(3 * variance)
        weights = np.random.uniform(-bound, bound, size=(n_in, n_out))                 
    return weights