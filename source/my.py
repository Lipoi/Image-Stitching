import cv2
import numpy
import Stitcher

imageA = cv2.imread("1.JPG")
imageB = cv2.imread("2.JPG")
imageC = cv2.imread("3.JPG")
imageD = cv2.imread("4.JPG")

image1 = cv2.resize(imageA,(600,450))
image2 = cv2.resize(imageB,(600,450))
image3 = cv2.resize(imageC,(600,450))
image4 = cv2.resize(imageD,(600,450))
'''
height = image1.shape[0]
width = image1.shape[1]

image11 = image1[0:height, 0:width//2]
image12 = image1[0:height, width//2:width]
image21 = image2[0:height, 0:width//2]
image22 = image2[0:height, width//2:width]
image31 = image3[0:height, 0:width//2]
image32 = image3[0:height, width//2:width]
image41 = image4[0:height, 0:width//2]
image42 = image4[0:height, width//2:width]


'''

'''
stitcher = Stitcher()
(result1, match1) = stitcher.stitch([image1, image2], showMatches=True)
(result2, match2) = stitcher.stitch([result1, image3], showMatches=True)
(result3, match3) = stitcher.stitch([result2, image4], showMatches=True)

cv2.imshow("match1", match1)
cv2.imshow("match2", match2)
cv2.imshow("match3", match3)
cv2.imshow("result1", result1)
cv2.imshow("result2", result2)
cv2.imshow("result3", result3)
'''

result1, match1 = Stitcher.Stitcher(image2, image1)
result2, match2 = Stitcher.Stitcher(image4, image3)
result1 = result1[0:450,0:700]
result2 = result2[0:450,0:700]
result3, match3 = Stitcher.Stitcher(result2, result1)
cv2.imshow("result", result3)
cv2.imshow("match", match3)
cv2.waitKey(0)

