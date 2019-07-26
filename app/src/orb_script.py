import cv2
import numpy as np

path_to_video = '../resources/find_chocolate.mp4'
path_to_template = '../resources/marker.jpg'

#path_to_template = '../resources/marker2.png'
#path_to_video = '../resources/second_chance.mp4'

def meth(img, frame, grayframe):
    print("ORB")
    # ORB Features Detector
    orb = cv2.ORB_create(nfeatures=2000)

    # features of picture
    kp_image, desc_image = orb.detectAndCompute(img, None)

    # features of grayframe
    kp_grayframe, desc_grayframe = orb.detectAndCompute(grayframe, None)

    # Brute Force Matching
    # вместе с orb детектором обычно используется cv2.NORM_HAMMING
    # crossCheck=True - означает, что будет меньше совпадений, но более качественных
    bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=False)
    matches = bf.match(desc_image, desc_grayframe)
    matches = sorted(matches, key=lambda x: x.distance)  # sort matches with distance

    # отбираем только хорошие совпадения
    good_matches = []
    for m in matches:
        if m.distance < 40:
            good_matches.append(m)

    # matches = sorted(matches, key=lambda x: x.distance)
    good_matches = sorted(good_matches, key=lambda x: x.distance)
    if (len(good_matches) > 20):
        good_matches = good_matches[:20]

    # создаём картинку, отображающую совпадения
    img_with_matching = cv2.drawMatches(img, kp_image, grayframe, kp_grayframe, good_matches, grayframe)

    # Homography(Гомография) - перспективная трансформация
    # если число совпадений> 3, ищем гомографию, иначе - отображаем просто grayframe
    if (len(good_matches) > 3):
        #good_matches_truncated = good_matches[:10]

        # get coordinates of picture(query) and grayframe(train) keypoints
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # находим гомографию, матрицу перспективной трансформации между двумя картинками(query и train)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        # Perspective transform
        h, w = img.shape  # размеры картинки(query)

        # создаём рамку для обозначения совпадающей картинки
        # дляя perspectiveTransform надо использовать [[[, ], [, ], [, ], [, ]]]
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        # define color of marking
        # red
        color = (0, 255, 0)


        # делаем перспективную трансформацию рамки параллельно картинке
        dst = cv2.perspectiveTransform(pts, matrix)
        #dst = cv2.perspectiveTransform(pts[None, :, :], matrix)
        homography = cv2.polylines(frame, [np.int32(dst)], True, color, 3)
        # homography = frame

        '''
        # Initialize lists
        list_kp_image = []
        list_kp_grayframe = []

        # For each match...
        for mat in good_points_truncated:
            # Get the matching keypoints for each of the images
            image_idx = mat.queryIdx
            grayframe_idx = mat.trainIdx

            # x - columns
            # y - rows
            # Get the coordinates
            (x1, y1) = kp_image[image_idx].pt
            (x2, y2) = kp_grayframe[grayframe_idx].pt

            # Append to each list
            list_kp_image.append((x1, y1))
            list_kp_grayframe.append((x2, y2))
        cx = 0
        cy = 0
        N = len(list_kp_grayframe)
        for i in range(N):
            cx += int(list_kp_grayframe[i][0])
            cy += int(list_kp_grayframe[i][1])
        cx = cx // N
        cy = cy // N
        '''

        # compute the center of the contour
        M = cv2.moments(dst)
        if M["m00"] == 0.0:
            cx = int(M["m10"] / (M["m00"] + 0.00000001))
            cy = int(M["m01"] / (M["m00"] + 0.00000001))
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        #cv2.circle(homography, (cx, cy), 20, color, 3)
        label='chocolate'
        cv2.putText(homography, label, (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("Homography", grayframe)

    return img_with_matching


def main():

    img = cv2.imread(path_to_template, cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture(path_to_video)

    while True:

        ret, frame = cap.read()

        if ret == True:
            # convert to gray
            grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            img_with_matching = meth(img, frame, grayframe)

            cv2.imshow("img_with_matching", img_with_matching)

            # if 'Esc' (k==27) is pressed then break
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
