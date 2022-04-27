import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup


# if len(sys.argv) == 0:
    #     print("Имя файла не указано")
    #     return None
    # if os.path.basename(sys.argv[0]) == os.path.basename(current_ex_file):
    #     if len(sys.argv) > 1:
    #         path = sys.argv[1]
    #         print(path)
    #         return get_image_using_path(path)
    #     else:
    #         print("Имя файла не указано")
    #         return None
    # else:
    #     path = sys.argv[0]
    #     return get_image_using_path(path)


# def preprocessing_for_border_detection(img: np.ndarray,
#                                        gauss_kernel_size,
#                                        morph_kernel_size1,
#                                        morph_kernel_size2,
#                                        canny_thresholds=(30, 60)):
#
#     img_result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size1, morph_kernel_size1))
#     img_result = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel1, iterations=1)
#
#     x1 = 0
#     x2 = img_result.shape[1] - 1
#     for y in range(img_result.shape[0]):
#         img_result[y, x1] = 0
#         img_result[y, x2] = 0
#
#     img_result = cv2.GaussianBlur(img_result, (gauss_kernel_size, gauss_kernel_size), 0)
#     img_result = cv2.Canny(img_result, canny_thresholds[0], canny_thresholds[1])
#     kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size2, morph_kernel_size2))
#     img_result = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel2)
#
#     return img_result


def preprocessing_for_text_recognition(img, aug_values=(4, 8)):     # 5.5, 4 значение aug_value подобрано эмпирически
    results = []
    for aug_value in aug_values:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        augmented = cv2.resize(gray, None, fx=aug_value, fy=aug_value)
        thresholded_binary = cv2.threshold(augmented, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # thr = cv2.threshold(augmented, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0]
        # thresholded_binary = cv2.threshold(augmented, int(thr * 0.9), 255, cv2.THRESH_BINARY)[1]
        # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        # eroded = cv2.erode(thresholded_binary, kernel1, iterations=1)
        results.append(thresholded_binary)

    cv2.imshow('img', thresholded_binary)
    cv2.waitKey(100)
    return results

# if 1.3 <= w / h <= 2.1 and w * h > 64:
#     pass
#     cropped_img = img[y:y + h, x:x + w]
#
#     current_mean_color = get_mean_color(cropped_img)
#     if not check_color_proximity('red_price_mean_color', current_mean_color):  # перепроверить
#         if not check_color_proximity('green_price_mean_color', current_mean_color):
#             break
#
#     # if get_nearest_mean_color(current_mean_color) == 'blue_price_mean_color':
#     #     break
#
#     cropped_img1 = cropped_img[:cropped_img.shape[1] // 2, :]
#     cropped_img2 = cropped_img[cropped_img.shape[1] // 2:, :]
#
#     preprocessed_img_list1 = preprocessing_for_text_recognition(cropped_img1)
#     preprocessed_img_list2 = preprocessing_for_text_recognition(cropped_img2)
#     search_price = False
#     search_time = False
#
#     for preprocessed_img1 in preprocessed_img_list1:
#         chars = get_digit_only_text_data(preprocessed_img1)['text']
#         text = ''
#         for ch in chars:
#             text += ch
#         search_price = re.search(price_pattern, text)
#     for preprocessed_img2 in preprocessed_img_list2:
#         chars = get_digit_only_text_data(preprocessed_img2)['text']
#         text = ''
#         for ch in chars:
#             text += ch
#         search_time = re.search(time_pattern, text)
#     if search_price and search_time:
#         current_price_result = PriceResult(f"{search_time.group(0)}", x, y, w, h)  # разобраться со временем
#         text_cash.add(bbox, f"{search_time.group(0)}")


# def get_current_area(img, red_area_info, green_area_info):
#     """define in which area the current price is"""
#
#     red_filtered_img = get_filtered_by_colors_image(img, red_area_info.lower, red_area_info.upper)
#     red_contours = get_all_approx_contours(red_filtered_img, red_area_info.params)
#     max_red_contour = find_contour_with_the_biggest_area(red_contours)
#
#     green_filtered_img = get_filtered_by_colors_image(img, green_area_info.lower, green_area_info.upper)
#     green_contours = get_all_approx_contours(green_filtered_img, green_area_info.params)
#     max_green_contour = find_contour_with_the_biggest_area(green_contours)
#
#     red_bbox, green_bbox = get_bounding_boxes([max_red_contour, max_green_contour])
#     print(red_bbox)
#     print(green_bbox)
#
#     if red_bbox[1] < green_bbox[1]:
#         return "red"
#     else:
#         return "green"


def delete_black_borders(img):  # вроде не нужно
    mean = 0
    width = img.shape[1]
    height = img.shape[0]
    threshold = 30

    for x in range(width):
        for y in range(height):
            mean += img[y, x]
    mean /= width * height

    depth = 0
    black_borders_is_not_deleted = True
    while black_borders_is_not_deleted:
        black_borders_is_not_deleted = False
        for x in range(depth, depth + 1):
            for y in range(1, height - 1):
                if img[y, x] < threshold:
                    black_borders_is_not_deleted = True
                    for canal in range(3):
                        img[y, x] = mean

        for x in range(width - depth - 2, width - depth - 1):
            for y in range(1, height - 1):
                if img[y, x] < threshold:
                    black_borders_is_not_deleted = True
                    for canal in range(3):
                        img[y, x] = mean
        depth += 0

    depth = 0
    black_borders_is_not_deleted = True
    while black_borders_is_not_deleted:
        black_borders_is_not_deleted = False
        for x in range(0, width):
            for y in range(depth, depth + 1):
                if img[y, x] < threshold:
                    black_borders_is_not_deleted = True
                    for canal in range(3):
                        img[y, x] = mean

        for x in range(0, width):
            for y in range(height - depth - 2, height - depth - 1):
                if img[y, x] < threshold:
                    black_borders_is_not_deleted = True
                    for canal in range(3):
                        img[y, x] = mean
        depth += 0
    return img


def get_image_using_url(original_url: str) -> np.ndarray:
    """return the image from the standard page that the url points to"""

    response = requests.get(original_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    img_url = soup.find('img').get('src')

    img_response = requests.get(img_url)
    img_arr = np.asarray(bytearray(img_response.content), dtype=np.uint8)
    img = cv2.imdecode(img_arr, -1)

    return img

def trash():
    # cv2.imwrite("example.png", img)

    img = cv2.imread("example.png")

    img = cv2.resize(img, (640, 640))

    img = cv2.GaussianBlur(img, (1, 1), 0)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower = np.array([0, 195, 0])# for red
    # upper = np.array([23, 255, 255])

    lower = np.array([85, 0, 135])# for green
    upper = np.array([98, 255, 255])

    # lower = np.array([103, 0, 108]) # for gray
    # upper = np.array([122, 66, 148])

    mask = cv2.inRange(imgHSV, lower, upper)
    imgResult = cv2.bitwise_and(img, img, mask=mask)

    cv2.imshow('img', imgResult)
    cv2.waitKey(0)

    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # d = pytesseract.image_to_data(imgResult, output_type=Output.DICT)
    # print(d['text'])
    #
    # n_boxes = len(d['text'])
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #     img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    gray = cv2.cvtColor(imgResult, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(blur, 30, 60)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    cv2.imshow('img', closed)
    cv2.waitKey(0)

    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    con_poly = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            con_poly.append(approx)

    # black = np.zeros((640, 640, 3), np.uint8)
    poly = cv2.drawContours(img, con_poly, -1, (0, 255, 255), 2)

    cv2.imshow('img', poly)
    cv2.waitKey(0)

# img = cv2.imread("example.png")
    # filtered_img = get_filtered_by_colors_image(img, green_price_lower, green_price_upper)
    # filtered_img = preprocessing(filtered_img)
    # pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    # start = time.time()
    # d = pytesseract.image_to_boxes(filtered_img, output_type=Output.DICT)
    # print(d.keys())
    # print(time.time() - start)
    # n_boxes = len(d['char'])
    # print(n_boxes)
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['right'][i], d['bottom'][i])
    #     img = cv2.rectangle(img, (x, y), (w, h), (0, 255, 0), 2)
    # cv2.imshow('img', cv2.resize(img, (640, 640)))
    # cv2.waitKey(0)