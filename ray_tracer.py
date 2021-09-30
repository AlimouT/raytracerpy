import cv2
import numpy as np




def main():
    screen = np.zeros([200,400,3])
    screen[:,:,0] = 200
    screen[:,:,1] = 0
    screen[:,:,2] = 255
    cv2.imwrite('output/testimage.jpg', screen)

if __name__ ==  "__main__":
    main()