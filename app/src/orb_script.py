import cv2
import numpy as np

path_to_video = '../resources/find_chocolate.mp4'
path_to_template = '../resources/marker.jpg'

path_to_video = '../resources/second_chance.mp4'
path_to_template = '../resources/marker2.png'

color_blue = (255, 0, 0)
color_green = (0, 255, 0)
color_red = (0, 0, 255)

label = 'chocolate'


def find_good_matches(matches):
    # отбираем только хорошие совпадения
    good_matches = []
    for m in matches:
        if m.distance < 30:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    if (len(good_matches) > 30):
        good_matches = good_matches[:20]
    return good_matches


def get_contour_center(dst):
    # compute the center of the contour
    M = cv2.moments(dst)
    if M["m00"] == 0.0:
        cx = int(M["m10"] / (M["m00"] + 0.00000001))
        cy = int(M["m01"] / (M["m00"] + 0.00000001))
    else:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    return cx, cy


def get_image_with_matching(template, frame, frame_gray):
    # ORB Features Detector
    #orb = cv2.ORB_create(nfeatures=1500)
    orb = cv2.ORB_create(nfeatures=2000, scoreType=cv2.ORB_FAST_SCORE)

    # features of template
    kp_template, desc_template = orb.detectAndCompute(template, None)

    # features of frame_gray
    kp_frame_gray, desc_frame_gray = orb.detectAndCompute(frame_gray, None)

    # Brute Force Matching
    # вместе с orb детектором обычно используется cv2.NORM_HAMMING
    # crossCheck=True - означает, что будет меньше совпадений, но более качественных
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_template, desc_frame_gray)
    matches = sorted(matches, key=lambda x: x.distance)  # sort matches with distance

    # отбираем только хорошие совпадения
    good_matches = find_good_matches(matches)

    # создаём картинку, отображающую совпадения
    image_with_matching = cv2.drawMatches(template, kp_template, frame_gray, kp_frame_gray, good_matches, frame_gray)

    # Homography(Гомография) - перспективная трансформация
    # если число совпадений> 3, ищем гомографию, иначе - отображаем просто grayframe
    if (len(good_matches) > 3):

        # get coordinates of template(query) and frame_gray(train) keypoints
        query_pts = np.float32([kp_template[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_frame_gray[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # находим гомографию, матрицу перспективной трансформации между двумя картинками(query и train)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()

        cv2.imshow('mask', mask)
        #cv2.imshow('matches_mask', matches_mask)

        # размеры картинки template(query)
        h, w = template.shape

        # создаём рамку для обозначения совпадающей картинки
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = pts
        if matrix is not None:
            # делаем перспективную трансформацию рамки параллельно картинке
            dst = cv2.perspectiveTransform(pts, matrix)

        homography = cv2.polylines(frame, [np.int32(dst)], True, color_green, 3)

        # compute the center of the contour
        cx, cy = get_contour_center(dst)

        cv2.putText(homography, label, (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_green, 2)

        cv2.imshow("Homography", homography)
    else:
        cv2.imshow("Homography", frame_gray)

    return image_with_matching


def main():
    template = cv2.imread(path_to_template, cv2.IMREAD_GRAYSCALE)
    cap = cv2.VideoCapture(path_to_video)

    while True:

        ret, frame = cap.read()
        if ret == True:

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            img_with_matching = get_image_with_matching(template, frame, frame_gray)

            cv2.imshow("img_with_matching", img_with_matching)

            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
