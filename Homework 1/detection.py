import os
import cv2
from matplotlib.backend_bases import namedtuple
import matplotlib.pyplot as plt

def detect(dataPath, clf):
    """
    Please read detectData.txt to understand the format. Load the image and get
    the face images. Transfer the face images to 19 x 19 and grayscale images.
    Use clf.classify() function to detect faces. Show face detection results.
    If the result is True, draw the green box on the image. Otherwise, draw
    the red box on the image.
      Parameters:
        dataPath: the path of detectData.txt
      Returns:
        No returns.
    """
    # Begin your code (Part 4)
    """
    1. Read every lines in the detectData.txt into the list "line".
    2. For each image, first get its name and the number of detected area.
    3. Read in the image, beside the original one, we also need to 
      read in another in grayscale so that it has the same format as training data.
    4. For each detected area, get its coordinates of the top-left corner, width, and height.
    5. Cut and resize the area in order to have the same format as training data.
    6. Call clf.classify() to detect the area, draw a green rectangle if it's a face,
      otherwise draw a red rectangle.
    7. Show the outcome.
    """
    with open(dataPath , 'r') as file:
      line = file.readlines() 
      count = 0
      while count < len(line):
        name, num = map(str , line[count].split())

        img_ori = cv2.imread("data/detect/" + name)
        img = cv2.imread("data/detect/" + name , cv2.IMREAD_GRAYSCALE)
        
        for i in range(int(num)):
          count += 1
          info = list(map(int , line[count].split()))
          x1, y1 = info[0], info[1]
          x2, y2 = x1 + info[2], y1 + info[3]
          face = img[y1:y2 , x1:x2]
          resized_face = cv2.resize(face , (19, 19))

          if clf.classify(resized_face) == 1:
            cv2.rectangle(img_ori, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
          else:
            cv2.rectangle(img_ori, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

        plt.imshow(cv2.cvtColor(img_ori , cv2.COLOR_BGR2RGB))
        plt.show()
        count += 1
    # End your code (Part 4)
