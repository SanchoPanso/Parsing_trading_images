import numpy as np
import cv2
import time
import sys
import os

import config as cfg
# from config import *
from border_detection import get_ticker_borders, get_borders_of_vertical_scale
from input_output import get_image_using_path, write_into_json, get_image_using_url

from searching_for_definite_objects import PriceResult, mark_wrong_price_results
from searching_for_definite_objects import red_price_info, green_price_info, gray_price_info, white_price_info
from searching_for_definite_objects import prepare_image_for_price
from searching_for_definite_objects import get_price_data, define_direction, delete_intersecting_and_too_small_or_big
from searching_for_definite_objects import get_ticker


positive_samples_counter = 0


def highlight_prices(img, border, all_price_result: dict, ticker, direction, delay):
    global positive_samples_counter
    img = cv2.putText(img, 'Ticker: ' + ticker,
                      (int(img.shape[1] * 0.25), int(img.shape[0] * 0.25)),
                      cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    img = cv2.putText(img, 'Direction: ' + direction,
                      (int(img.shape[1] * 0.25), int(img.shape[0] * 0.3)),
                      cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    # draw vertical scale
    # img = cv2.line(img, (border, 0), (border, img.shape[0] - 1), (0, 255, 255))

    line_lenght = 900
    colors = {
        cfg.red_price_key: (0, 0, 255),
        cfg.green_price_key: (0, 255, 0),
        cfg.gray_price_key: (128, 128, 128),
        cfg.white_price_key: (255, 255, 255),
    }
    for i in all_price_result.keys():
        price_result = all_price_result[i]
        color = colors[i]
        for price in price_result:
            x = price.x
            y = price.y
            w = price.w
            h = price.h
            value = price.value
            reliability = price.reliability

            # get positive samples
            # sample_img = img[y: y + h, x + border - 7: x + border - 7 + w]
            # cv2.imwrite(f"positive_samples\\{positive_samples_counter}.jpg", sample_img)
            # positive_samples_counter += 1

            # get negative samples

            img = cv2.rectangle(img, (x + border - 7, y), (x + border - 7 + w, y + h), color, 2)
            img = cv2.line(img, (x + border - 7, y), (x + border - 7 - line_lenght, y), color)
            img = cv2.putText(img, value, (x + border - 7 - line_lenght, y - 5),
                                  cv2.FONT_HERSHEY_COMPLEX, 0.8, color, 2)

    cv2.imshow('result', cv2.resize(img, (1280, 768)))
    cv2.waitKey(delay)
    return img


def test_directory():
    img_paths = os.listdir("../original_images")
    start_time = time.time()
    for path in img_paths:
        # try:
            print("##################")
            print(path)
            img = cv2.imread(os.path.join("../original_images", path))
            ticker = get_ticker(img)
            borders_for_prices = get_borders_of_vertical_scale(img)
            for border in borders_for_prices:
                img_for_prices = prepare_image_for_price(img, border, 7)

                red_data = get_price_data(img_for_prices, red_price_info)
                print(f"Предварительная красная цена: {red_data}")

                gray_data = get_price_data(img_for_prices, gray_price_info)
                print(f"Предварительная серая цена: {gray_data}")

                green_data = get_price_data(img_for_prices, green_price_info)
                print(f"Предварительная зеленая цена: {green_data}")

                white_data = get_price_data(img_for_prices, white_price_info)
                print(f"Предварительная белая цена: {white_data}")

                all_price_results = {
                    cfg.red_price_key: red_data,
                    cfg.green_price_key: green_data,
                    cfg.gray_price_key: gray_data,
                    cfg.white_price_key: white_data,
                }
                all_price_results = delete_intersecting_and_too_small_or_big(all_price_results)
                # all_price_results = mark_wrong_price_results(all_price_results)
                direction = define_direction(all_price_results)
                img = highlight_prices(img, borders_for_prices[0], all_price_results, ticker, direction, 1)
                if not os.path.exists("results"):
                    os.mkdir("results")
                cv2.imwrite(os.path.join("results", path), img)
        # except Exception as e:
        #     print(e)
        #     print(f"There is a trouble with file{path}")
    print("Время работы: {:.2f} с".format(time.time() - start_time))
    if cv2.waitKey(0) == 27:
        sys.exit()


def test_one_file():
    # print(sys.argv)
    start_time = time.time()

    img = get_image_using_url("https://www.tradingview.com/x/JopwW6IR/")

    borders_for_prices = get_borders_of_vertical_scale(img)
    for border in borders_for_prices:
        img_for_prices = prepare_image_for_price(img, border, 9)

        ticker = get_ticker(img)
        print(f"Тикер: {get_ticker(img)}")

        red_data = get_price_data(img_for_prices, red_price_info)
        print(f"Предварительная красная цена: {red_data}")

        green_data = get_price_data(img_for_prices, green_price_info)
        print(f"Предварительная зеленая цена: {green_data}")

        gray_data = get_price_data(img_for_prices, gray_price_info)
        print(f"Предварительная серая цена: {gray_data}")

        white_data = get_price_data(img_for_prices, white_price_info)
        print(f"Предварительная белая цена: {white_data}")

        if len(red_data + green_data + gray_data + white_data) == 0:
            continue

        all_price_results = {
            cfg.red_price_key: red_data,
            cfg.green_price_key: green_data,
            cfg.gray_price_key: gray_data,
            cfg.white_price_key: white_data,
        }

        all_price_results = delete_intersecting_and_too_small_or_big(all_price_results)
        # all_price_results = mark_wrong_price_results(all_price_results)
        direction = define_direction(all_price_results)

        print("Время работы: {:.2f} с".format(time.time() - start_time))
        highlight_prices(img, border, all_price_results, ticker, direction, 0)


if __name__ == '__main__':
    test_directory()
