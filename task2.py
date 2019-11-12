"""
Template Matching

The goal of this task is to experiment with template matching techniques, i.e., normalized cross correlation (NCC).
"""


import argparse
import json
import os

import utils
from task1 import *


def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--img-path",
        type=str,
        default="./data/proj1-task2.jpg",
        help="path to the image")
    parser.add_argument(
        "--template-path",
        type=str,
        default="./data/proj1-task2-template.jpg",
        help="path to the template"
    )
    parser.add_argument(
        "--result-saving-path",
        dest="rs_path",
        type=str,
        default="./results/task2.json",
        help="path to file which results are saved (do not change this arg)"
    )
    args = parser.parse_args()
    return args

def elementwise_mul_mean_sum(a, b, mean_a, mean_b):
    """Elementwise multiplication."""
    value = 0.0
    for i, row in enumerate(a):
        for j, num in enumerate(row):
            value += (a[i][j] - mean_a)*(b[i][j] -mean_b)
    return value

def elementwise_square_sum(a, mean_a) :
    value = 0.0
    for i, row in enumerate(a):
        for j, num in enumerate(row):
            value += (a[i][j]- mean_a)**2
    return value

def norm_xcorr2d(patch, template):
    """Computes the NCC value between a image patch and a template.

    The image patch and the template are of the same size. The formula used to compute the NCC value is:
    sum_{i,j}(x_{i,j} - x^{m}_{i,j})(y_{i,j} - y^{m}_{i,j}) / (sum_{i,j}(x_{i,j} - x^{m}_{i,j}) ** 2 * sum_{i,j}(y_{i,j} - y^{m}_{i,j})) ** 0.5
    This equation is the one shown in Prof. Yuan's ppt.

    Args:
        patch: nested list (int), image patch.
        template: nested list (int), template.

    Returns:
        value (float): the NCC value between a image patch and a template.
    """

    patch_value = 0.0
    increment = 0
    for i, row in enumerate(patch):
        for j, column in enumerate(row):
            patch_value += patch[i][j]
            increment += 1
    patch_mean = (patch_value*1.0)/increment

    template_value = 0.0
    additive = 0
    for a, r in enumerate(template):
        for b, c in enumerate(r):
            template_value += template[a][b]
            additive += 1
    template_mean = (template_value*1.0)/additive

    elementwise_mul_mean_sum_value = elementwise_mul_mean_sum(patch, template, patch_mean, template_mean)
    elementwise_square_sum_patch_value = elementwise_square_sum(patch, patch_mean)
    elementwise_square_sum_template_value = elementwise_square_sum(template, template_mean)
    value = elementwise_mul_mean_sum_value/np.sqrt(elementwise_square_sum_patch_value * elementwise_square_sum_template_value)
    #print(value)
    return value
    
    
    #raise NotImplementedError

def match(img, template):
    """Locates the template, i.e., a image patch, in a large image using template matching techniques, i.e., NCC.

    Args:
        img: nested list (int), image that contains character to be detected.
        template: nested list (int), template image.

    Returns:
        x (int): row that the character appears (starts from 0).
        y (int): column that the character appears (starts from 0).
        max_value (float): maximum NCC value.
    """
    # TODO: implement this function.
    # raise NotImplementedError
    x = -1
    y = -1
    max_value = -1
    template_rows = int(len(template))
    template_columns = int(len(template[0]))
    image_rows = len(img)
    image_columns = len(img[0])
    #padded_img = utils.zero_pad(img, pad_x, pad_y)
    for i, row in enumerate(img):
        for j, num in enumerate(row):
            
            if i + template_rows > image_rows or j + template_columns > image_columns: 
                continue
            #print(i,j)
            cropped_img = utils.crop(img,i,i+ template_rows , j , j+template_columns)
            value = norm_xcorr2d(cropped_img, template)
            if value > max_value:
                x = i
                y = j
                max_value = value
    return x, y, round(max_value,3)
    
    raise NotImplementedError

def save_results(coordinates, template, template_name, rs_directory):
    results = {}
    results["coordinates"] = sorted(coordinates, key=lambda x: x[0])
    results["templat_size"] = (len(template), len(template[0]))
    with open(os.path.join(rs_directory, template_name), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()

    img = read_image(args.img_path)
    # template = utils.crop(img, xmin=10, xmax=30, ymin=10, ymax=30)
    # template = np.asarray(template, dtype=np.uint8)
    # cv2.imwrite("./data/proj1-task2-template.jpg", template)
    template = read_image(args.template_path)

    x, y, max_value = match(img, template)
    # The correct results are: x: 17, y: 129, max_value: 0.994
    with open(args.rs_path, "w") as file:
        json.dump({"x": x, "y": y, "value": max_value}, file)


if __name__ == "__main__":
    main()
