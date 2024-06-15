import os
import cv2

def loadImages(dataPath):
    """
    load all Images in the folder and transfer a list of tuples. The first 
    element is the numpy array of shape (m, n) representing the image. 
    The second element is its classification (1 or 0)
      Parameters:
        dataPath: The folder path.
      Returns:
        dataset: The list of tuples.
    """
    # Begin your code (Part 1)
    """
    1. Create a list named "dataset" to store the tuples.
    2. For every image in the "face" folder, read it in grayscale so that 
      the resulting numpy array will have only two dimensions.
    3. If the image exists, append it to dataset with classification "1"
    4. As for non-face images, the procedure is similar, while their classification should be "0"
    """
    dataset = []

    root = str(dataPath) + "/face/"
    for image in os.listdir(root):
      img = cv2.imread(os.path.join(root,image),cv2.IMREAD_GRAYSCALE)
      if img is not None:
        dataset.append( (img,1) )
        
    root = str(dataPath) + "/non-face/"
    for image in os.listdir(root):
      img = cv2.imread(os.path.join(root,image),cv2.IMREAD_GRAYSCALE)
      if img is not None:
        dataset.append( (img,0) )   
    # End your code (Part 1)
    return dataset
