import cv2
import numpy as np
import math
import pyautogui
import os


pyautogui.FAILSAFE = False
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    roi = frame[50:350, 50:350]
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 0)
    crop_image = roi

    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)

    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contour = max(contours, key=lambda x: cv2.contourArea(x))

        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        hull = cv2.convexHull(contour)

        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 2)

        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)

        if defects is not None:
            count_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])

                a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)

                angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180 / math.pi

                if angle <= 90:
                    count_defects += 1
                    cv2.circle(crop_image, far, 4, [0, 0, 255], -1)
                    cv2.line(crop_image, start, end, [0, 255, 0], 2)

            if count_defects >= 2:
                pyautogui.press('space')

               # os.system('xdotool search --onlyvisible --class "google-chrome" windowactivate --sync key space')
                

                cv2.putText(frame, "JUMP", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Window", frame)
    cv2.imshow("Countours", drawing)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
