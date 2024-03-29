import cv2
import numpy as np

color_blue = (255, 0, 0)
color_green = (0, 255, 0)
color_red = (0, 0, 255)


def find_matches_using_orb(img_template, old_frame, old_frame_gray, old_dst, frames_count):
    # ORB Features Detector
    if frames_count == 0:
        nfeatures = 15000
    else:
        nfeatures = 1500

    orb = cv2.ORB_create(nfeatures)

    # features of picture
    kp_image, desc_image = orb.detectAndCompute(img_template, None)

    # features of grayframe
    kp_grayframe, desc_grayframe = orb.detectAndCompute(old_frame_gray, None)

    # Brute Force Matching
    # вместе с orb детектором обычно используется cv2.NORM_HAMMING
    # crossCheck=True - означает, что будет меньше совпадений, но более качественных
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc_image, desc_grayframe)
    matches = sorted(matches, key=lambda x: x.distance)  # sort matches with distance

    # отбираем только хорошие совпадения
    good_matches = []
    for m in matches:
        if m.distance < 50:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    if (len(good_matches) > 40):
        good_matches = good_matches[:40]





    train_pts = np.float32([])
    dst1 = old_dst

    # Homography(Гомография) - перспективная трансформация
    # если число совпадений> 3, ищем гомографию, иначе - отображаем просто grayframe


    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=2500,
                          qualityLevel=0.01,
                          minDistance=10,
                          blockSize=10)
    p0_templ = cv2.goodFeaturesToTrack(img_template, mask=None, **feature_params)
    p0_frame = cv2.goodFeaturesToTrack(old_frame_gray, mask=None, **feature_params)

    if (len(good_matches) > 10):

        # get coordinates of picture(query) and grayframe(train) keypoints
        query_pts = np.float32([kp_image[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_grayframe[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)


        # находим гомографию, матрицу перспективной трансформации между двумя картинками(query и train)
        matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        p0_frame = cv2.goodFeaturesToTrack(old_frame_gray, mask=None, **feature_params)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matches_mask,  # draw only inliers
                           flags=2)

        # создаём картинку, отображающую совпадения
        img_with_matching = cv2.drawMatches(img_template, kp_image, old_frame_gray, kp_grayframe, good_matches,
                                            None, **draw_params)
        cv2.imshow("img_with_matching", img_with_matching)

        # Perspective transform
        h, w = img_template.shape  # размеры картинки(query)

        # создаём рамку для обозначения совпадающей картинки
        pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)

        # делаем перспективную трансформацию рамки параллельно картинке
        dst1 = cv2.perspectiveTransform(pts, matrix)
        homography = cv2.polylines(old_frame, [np.int32(dst1)], True, color_red, 3)

        # compute the center of the contour
        M = cv2.moments(dst1)
        if M["m00"] == 0.0:
            cx = int(M["m10"] / (M["m00"] + 0.00000001))
            cy = int(M["m01"] / (M["m00"] + 0.00000001))
        else:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        # cv2.circle(homography, (cx, cy), 20, color, 3)
        label = 'chocolate'
        cv2.putText(homography, label, (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color_red, 2)

        #cv2.imshow("Homography_initial", homography)
    else:

        #cv2.imshow("Homography_initial", old_frame_gray)
        pass

    #return train_pts, dst1
    return p0_frame, dst1


def main():
    # Lucas-Kanade parameters
    lk_params = dict(winSize=(15, 15),
                     maxLevel=10,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Take template image
    path_to_template = '../resources/marker.jpg'
    #path_to_template = '../resources/marker2.png'
    img_template = cv2.imread(path_to_template, cv2.IMREAD_GRAYSCALE)

    # Take video from .mp4-file
    path_to_video = '../resources/find_chocolate.mp4'
    #path_to_video = '../resources/second_chance.mp4'
    cap = cv2.VideoCapture(path_to_video)

    # Take first frame
    ret, old_frame = cap.read()
    old_frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    #cv2.imshow("old_frame_gray", old_frame_gray)
    #cv2.imshow("img_template", img_template)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    old_dst = np.float32([[0, 0]]).reshape(-1, 1, 2)
    # old_points = train_pts (in method return)

    frames_count = 0
    old_points, old_dst = find_matches_using_orb(img_template, old_frame, old_frame_gray, old_dst, frames_count)

    while True:
        ret, frame = cap.read()

        if ret == True:
            frames_count += 1
            # print(frames_count)

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # calculate optical flow
            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_frame_gray, frame_gray, old_points, None,
                                                                 **lk_params)

            # compute the center of the contour
            M = cv2.moments(new_points)
            if M["m00"] == 0.0:
                cx = int(M["m10"] / (M["m00"] + 0.00000001))
                cy = int(M["m01"] / (M["m00"] + 0.00000001))
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

            # Select good points
            good_new = new_points[status == 1]
            good_old = old_points[status == 1]
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                mask = cv2.line(mask, (a, b), (c, d), color_green, 2)
                cv2.circle(frame, (a, b), 5, color_red, -1)

            frame_with_mask = cv2.add(frame, mask)

            # get coordinates of picture(query) and grayframe(train) keypoints
            query_pts = good_old.reshape(-1, 1, 2)
            train_pts = good_new.reshape(-1, 1, 2)

            matrix, mask1 = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 7.0)
            matches_mask = mask1.ravel().tolist()
            # print(len(good_new))

            # делаем перспективную трансформацию рамки параллельно картинке
            dst = cv2.perspectiveTransform(old_dst, matrix)

            homography1 = cv2.polylines(frame_with_mask, [np.int32(dst)], True, (255, 0, 0), 3)
            homography2 = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)

            # compute the center of the contour
            M = cv2.moments(dst)
            if M["m00"] == 0.0:
                cx = int(M["m10"] / (M["m00"] + 0.00000001))
                cy = int(M["m01"] / (M["m00"] + 0.00000001))
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

            cv2.imshow("Homography1", homography1)
            cv2.imshow("Homography2", homography2)

            # cv2.imshow('frame_with_mask', frame_with_mask)
            # cv2.imshow('frame', frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

            if frames_count % 15 == 0:
                old_points, old_dst = find_matches_using_orb(img_template, old_frame, old_frame_gray, old_dst,
                                                             frames_count)
                #cv2.imshow('frame%10', frame)

            else:
                old_dst = dst
            old_points = good_new.reshape(-1, 1, 2)
            old_frame_gray = frame_gray
        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
