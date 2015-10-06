import cv2
import numpy as np
# from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10


def process_video():
    cap = cv2.VideoCapture('claire/claire0000.pgm')

    # Get the reference image
    dst = np.zeros((800, 800), np.uint8)
    out = np.zeros((800, 800), np.uint8)
    # M1 = np.eye(3)
    # dst[:ref.shape[0], :ref.shape[1]] = ref[:, :]
    rows, cols = dst.shape

    # Get features to match
    orb = cv2.ORB_create()
    first = False
    while(cap.isOpened()):
        if first is False:
            ret2, ref = cap.read()
            dst[100:100+ref.shape[0], 100:100+ref.shape[1]] = ref[:, :]
            first = True

        ret, frame = cap.read()
        if ret is False:
            break

        # Get keypoints and discriptor of mosaic and next frame
        kp1, des1 = orb.detectAndCompute(dst, None)
        kp2, des2 = orb.detectAndCompute(frame, None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        # make sure that we are matching the next frame to the ref
        # des2 = next frame descriptor
        # des1 = mosaic descriptor
        matches = bf.match(des2, des1)

        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        # Draw first 10 matches. frame to mosaic
        img3 = cv2.drawMatches(frame, kp2, dst, kp1, matches[:10], None,
                               flags=2)

        cv2.imshow('Matches', img3)

        # Get good points
        # Using all the points because we are using RANSAC to remove points
        # make sure that the dst points are for the mosaic (the target plane)
        good = matches
        src_pts = np.float32([kp2[m.queryIdx].pt for m in good])
        dst_pts = np.float32([kp1[m.trainIdx].pt for m in good])

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)
        print M
        # M1 = np.mat(M1) * np.mat(M).transpose

        out = cv2.warpPerspective(frame, M, (cols, rows))
        dst[out != 0] = np.uint8(out[out != 0])
        print dst
        cv2.imshow('Image mosaic', dst)

        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break

    cap.release()

process_video()
cv2.waitKey(0)
