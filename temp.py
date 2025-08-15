#new plate assignment attempt
import numpy as np

def assign_origins(vertices, num_plates):
    plate_origins = np.random.choice(vertices, num_plates, replace=False)
    
        