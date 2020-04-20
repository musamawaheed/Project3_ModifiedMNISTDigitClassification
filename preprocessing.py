import cv2
import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

with open('Data/train_images.pkl', 'rb') as d:
    train_images = pickle.load(d)

processed = []

for img in train_images:
    img2 = img.copy()
    retval, thresh_gray = cv2.threshold(img, thresh=254, maxval=255, type=cv2.THRESH_BINARY)
    thresh_gray2 = thresh_gray.copy()
    converted = cv2.convertScaleAbs(thresh_gray)
    converted2 = cv2.convertScaleAbs(thresh_gray)
    contours = cv2.findContours(converted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE, offset=(-1, -1))
    contours2 = cv2.findContours(converted2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mx = (0, 0, 0, 0)
    mx_area = 0
    for cont in contours[1]:
        x, y, w, h = cv2.boundingRect(cont)
        ls = max(w, h)
        area = ls ** 2
        #area = w * h
        if area > mx_area:
            mx = x, y, w, h
            mx_area = area
    x, y, w, h = mx
    roi = thresh_gray[y:y + h + 2, x:x + w + 2]

    mx2 = (0, 0, 0, 0)
    mx2_area = 0
    for cont2 in contours2[1]:
        x2, y2, w2, h2 = cv2.boundingRect(cont2)
        ls2 = max(w2, h2)
        area2 = ls2 ** 2
        #area = w * h
        if area2 > mx2_area:
            mx2 = x2, y2, w2, h2
            mx2_area = area2
    x2, y2, w2, h2 = mx2
    roi2 = thresh_gray[y2:y2 + h2, x2:x2 + w2]

    img_bb = cv2.rectangle(thresh_gray2, (x, y), (x + w, y + h), (255, 0, 255), 1)
    # plt.imshow(img_bb)
    # print(type(img_bb))

    # print(type(roi))
    #print(i, roi.shape)

    desired_size = 28
    old_size = roi.shape[:2]
    old_size2 = roi2.shape[:2]

    try:
        ratio = float(desired_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])
        im = cv2.resize(roi, (new_size[1], new_size[0]), interpolation=cv2.INTER_NEAREST)
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = (delta_h // 2), (delta_h - (delta_h // 2))
        left, right = (delta_w // 2), (delta_w - (delta_w // 2))
    except:
        ratio2 = float(desired_size) / max(old_size2)
        new_size2 = tuple([int(x * ratio2) for x in old_size2])
        im = cv2.resize(roi2, (new_size2[1], new_size2[0]), interpolation=cv2.INTER_NEAREST)
        delta_w = desired_size - new_size2[1]
        delta_h = desired_size - new_size2[0]
        top, bottom = (delta_h // 2), (delta_h - (delta_h // 2))
        left, right = (delta_w // 2), (delta_w - (delta_w // 2))

    color = [0, 0, 0]
    res = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    processed.append(res)

    # f, axarr = plt.subplots(1, 4)
    # axarr[0].set_ylabel('Example 2 \nLabel 5', fontsize=22)
    # axarr[0].imshow(img, cmap=cm.Greys_r)
    # axarr[0].set_title('Original image', fontsize=22)
    # axarr[1].imshow(thresh_gray, cmap=cm.Greys_r)
    # axarr[1].set_title('With threshold', fontsize=22)
    # axarr[2].imshow(img_bb, cmap=cm.Greys_r)
    # axarr[2].set_title('Largest bounding box', fontsize=22)
    # #axarr[3].imshow(roi, cmap=cm.Greys_r)
    # axarr[3].imshow(res, cmap=cm.Greys_r)
    # axarr[3].set_title('Cropped and resized', fontsize=22)
    # plt.show()

# print(len(processed))
processed_array = np.stack(processed)
print(processed_array.shape)
# print(type(processed[0]))


# plt.imshow(processed_array[100])
# plt.show()

# pickle_out = open("Data/train_data_processed3b_thresh.pkl", "wb")
# pickle.dump(processed_array, pickle_out)
# pickle_out.close()
