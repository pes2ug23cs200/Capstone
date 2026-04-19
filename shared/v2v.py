import numpy as np

FEATURES = 10  # adjust later if needed

def generate_message(benign=True):

    if benign:
        return np.random.normal(0,1,(1,20,17))
    else:
        return np.random.normal(5,2,(1,20,17))   