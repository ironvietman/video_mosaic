import cv2
import numpy as np
#from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10


def process_video():
    cap = cv2.VideoCapture('claire/claire0000.pgm')

    # Get the reference image
    dst = np.zeros((800, 800), np.float32)
    out = np.zeros((800, 800), np.float32)
    # M1 = np.eye(3)
    # dst[:ref.shape[0], :ref.shape[1]] = ref[:, :]
    rows, cols = dst.shape
    print dst

    # Get features to match
    orb = cv2.ORB_create()
    ret2, ref = cap.read()

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret is False:
            break
        kp1, des1 = orb.detectAndCompute(ref, None)
        kp2, des2 = orb.detectAndCompute(frame, None)
        print des2
        # create BFMatcher object
        bf = cv2.BFMatcher()

        # Match descriptors.
        matches = bf.match(des1, des2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches.
        img3 = cv2.drawMatches(ref, kp1, frame, kp2, matches[:10], None,
                               flags=2)

        cv2.imshow('Image', img3)

        # Get good points
        # store all the good matches as per Lowe's ratio test.
        good = matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good])
        print src_pts
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        print M
        #M1 = np.mat(M1) * np.mat(M).transpose

        out = cv2.warpPerspective(frame, M, (cols, rows))
        dst = cv2.addWeighted(out, 0.5, dst, 0.5, 0, 1, 0)

        cv2.imshow('Image mosaic', dst)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cap.release()

process_video()
cv2.waitKey(0)
