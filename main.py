#!/usr/bin/env python3
# Python code for Multiple Color Detection


import numpy as np
import cv2

# Capturing video through webcam
webcam = cv2.VideoCapture(0)

# Start a while loop
while (1):

    # Reading the video from the
    # webcam in image frames
    _, imageFrame = webcam.read()
    output = imageFrame
    gray = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2GRAY)


    currentbalpos = []

    # apply GuassianBlur to reduce noise. medianBlur is also added for smoothening, reducing noise.
    # gray = cv2.GaussianBlur(gray,(3,3),0);
    gray = cv2.medianBlur(gray, 3)
    cv2.imshow("gray & blur",gray)

    # gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY,11,3.5)

    kernel = np.ones((2, 2), np.uint8)
    # gray = cv2.erode(gray,kernel,iterations = 1)

    # gray = cv2.dilate(gray,kernel,iterations = 1)

    # Convert the imageFrame in
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    cv2.imshow("saturation",hsvFrame)

    # Set range for red color and
    # define mask

    red_lower = np.array([150, 56, 111], np.uint8)
    red_upper = np.array([180, 255, 255], np.uint8)
    red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

    # Set range for green color and
    # define mask
    # green_lower = np.array([25, 52, 72], np.uint8)
    green_lower = np.array([70, 104, 144], np.uint8)
    green_upper = np.array([102, 255, 255], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Set range for blue color and
    # define mask

    blue_lower = np.array([94, 147, 65], np.uint8)
    blue_upper = np.array([179, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)

    lower_white = np.array([0, 0, 196])
    upper_white = np.array([179, 26, 255])
    white_mask = cv2.inRange(hsvFrame, lower_white, upper_white)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    kernal = np.ones((5, 5), "uint8")


    # For red color
    red_mask = cv2.dilate(red_mask, kernal)
    res_red = cv2.bitwise_and(imageFrame, imageFrame,
                              mask=red_mask)
    cv2.imshow("mask red",red_mask)


    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=green_mask)


    # For blue color
    blue_mask = cv2.dilate(blue_mask, kernal)
    res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=blue_mask)
    cv2.imshow("mask blue",res_blue)


    # yellow
    yellow_lower = np.array([22, 93, 0], np.uint8)
    yellow_upper = np.array([45, 255, 255], np.uint8)
    yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)
    yellow_mask = cv2.dilate(yellow_mask, kernal)
    res_white = cv2.bitwise_and(imageFrame, imageFrame,
                               mask=yellow_mask)

    cv2.imshow("mask yellow",yellow_mask)



    white_mask = cv2.dilate(white_mask, kernal)
    res_white = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=white_mask)

    cv2.imshow("mask white",white_mask)


    # Creating contour to track yellow color
    contours, hierarchy = cv2.findContours(red_mask,
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 150, param1=100, param2=45, minRadius=1, maxRadius=90)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

        # loop over the (x, y) coordinates and radius of the circles
        # for (x, y, r) in circles:
        for (x, y, r) in circles:

            # draw the circle in the output image, then draw a rectangle in the image
            # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            # cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # print(contours)
            for pic, contour in enumerate(contours):
                area = cv2.contourArea(contour)
                if (area > 300):
                    x2, y2, w2, h2 = cv2.boundingRect(contour)
                    if (x - (x2 + (w2 + h2) / 4) < 20):
                        cv2.putText(imageFrame, "red Colour", (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0))

                        currentbalpos.append([x2, y2, w2, h2])

    contours_white, hierarchy = cv2.findContours(white_mask,
                                                 cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours_white):
        area = cv2.contourArea(contour)
        if (area > 1000):
            x, y, w, h = cv2.boundingRect(contour)
            # print(y)
            imageFrame = cv2.rectangle(imageFrame, (x, y),
                                       (x + w, y + h),
                                       (0, 0, 0), 2)

            # cv2.putText(imageFrame, "white Colour", (x, y),
            #             cv2.FONT_HERSHEY_SIMPLEX,
            #             1.0, (255, 0, 0))
            for (xb, yb, wb, rb) in currentbalpos:
                if (x < xb < x + w and y < yb < y + h):
                    currentbalpos.clear()
                    cv2.putText(imageFrame, "BUT", (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.0, (0, 0, 255))
                    print("BUT")
            print("...................")

    # Program Termination
    cv2.imshow("LA BONNE BALLE", imageFrame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        webcam.release()
        cv2.destroyAllWindows()
        break