import cv2
import numpy
def Stitcher(imageA, imageB):
    gray1 = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    descriptor = cv2.xfeatures2d.SIFT_create()
    keypoisA, featuresA = descriptor.detectAndCompute(gray1, None)
    keypoisB, featuresB = descriptor.detectAndCompute(gray2, None)
    keypoisA = numpy.float32([keypoi.pt for keypoi in keypoisA])
    keypoisB = numpy.float32([keypoi.pt for keypoi in keypoisB])
    matches = []
    for m in cv2.DescriptorMatcher_create("BruteForce").knnMatch(featuresA, featuresB, 2):
        if len(m) == 2 and m[0].distance < m[1].distance * 0.75:
            matches.append((m[0].trainIdx, m[0].queryIdx))
    poA = numpy.float32([keypoisA[i] for (_, i) in matches])
    poB = numpy.float32([keypoisB[i] for (i, _) in matches])
    H, status = cv2.findHomography(poA, poB, cv2.RANSAC, 4)
    Heighta = imageA.shape[0]
    Widtha = imageA.shape[1]
    Heightb = imageB.shape[0]
    Widthb = imageB.shape[1]
    result = cv2.warpPerspective(imageA, H, (Widtha + Widthb, Heighta))
    result[0:Heightb, 0:Widthb] = imageB
    match = numpy.zeros((max(Heighta, Heightb), Widtha + Widthb, 3), dtype="uint8")
    match[0:Heighta, 0:Widtha] = imageA
    match[0:Heightb, Widtha:] = imageB
    for ((tx, qx), s) in zip(matches, status):
        if s == 1:
            ptA = (int(keypoisA[qx][0]), int(keypoisA[qx][1]))
            ptB = (int(keypoisB[tx][0]) + Widtha, int(keypoisB[tx][1]))
            cv2.line(match, ptA, ptB, (0, 0, 255), 1)
    return (result, match)
