import json
import uuid

import cv2
import numpy as np
import sys

import progress
from doc_hierarchy import Article, ArticleImage, Region
from utils import show, Box, box_width, box_height
import utils

__version__ = '1.4.0.1'

import glob
from argparse import ArgumentParser
import os
import shutil
import scipy as sp
import scipy.ndimage


from tqdm import tqdm


def preprocess_image(img, bin_thresh, otsu=False,debug=False):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug:
        cv2.imshow('gray image', img)# andrey
        cv2.waitKey(0)
    img = cv2.bitwise_not(img)
    if debug:
        cv2.imshow('inverted gray image', img)  # andrey
        cv2.waitKey(0)
    mode = cv2.THRESH_BINARY
    if otsu:
        mode += cv2.THRESH_OTSU
    ret,img = cv2.threshold(img, bin_thresh, 255, mode)
    if debug:
        cv2.imshow('bw image', img)  # andrey
        cv2.waitKey(0)
        #img1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 0)
        #cv2.imshow('bw image adaptive', img1)  # andrey
        #cv2.waitKey(0)
    return img


def get_region_boxes(img, pad=10, erode_iters=3, min_box_width=50, min_aspect=0.7, min_box_width_rel=0.1,
                     approx_factor=0.06, bin_thresh=5, min_box_height=20, debug=False):


    norm_img_src = preprocess_image(img, bin_thresh,False,debug)
    # pad to make clear edges
    src_img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0,))
    if debug:
        cv2.imshow('orig image with black borders', src_img)  # andrey
        cv2.waitKey(0)
    norm_img = cv2.copyMakeBorder(norm_img_src, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0,))
    if debug:
        cv2.imshow('bw image with black borders', norm_img)  # andrey
        cv2.waitKey(0)

    kernel = np.ones(shape=(3, 3))
    norm_img = cv2.erode(norm_img, kernel, iterations=erode_iters)
    if debug:
        cv2.imshow('bw image after erosion', norm_img)  # andrey
        cv2.waitKey(0)
    norm_img = cv2.dilate(norm_img, kernel, iterations=erode_iters)
    if debug:
        cv2.imshow('bw image after closing', norm_img)  # andrey
        cv2.waitKey(0)

    #kernel = np.ones(shape=(3, 11))
    #norm_img = cv2.erode(norm_img, kernel, iterations=erode_iters)
    #if debug:
    #    cv2.imshow('bw image after erosion', norm_img)  # andrey
    #    cv2.waitKey(0)
    #norm_img = cv2.dilate(norm_img, kernel, iterations=erode_iters)
    #if debug:
    #    cv2.imshow('bw image after closing', norm_img)  # andrey
    #    cv2.waitKey(0)
#
    #kernel = np.ones(shape=(11, 3))
    #norm_img = cv2.erode(norm_img, kernel, iterations=erode_iters)
    #if debug:
    #    cv2.imshow('bw image after erosion', norm_img)  # andrey
    #    cv2.waitKey(0)
    #norm_img = cv2.dilate(norm_img, kernel, iterations=erode_iters)
    #if debug:
    #    cv2.imshow('bw image after closing', norm_img)  # andrey
    #    cv2.waitKey(0)


    regs = src_img.copy()
    cnts = cv2.findContours(norm_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #cnts = cv2.findContours(norm_img, cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

    # box_offset = 2 if erode_iters == 0 else 0
    box_offset = 0
    approx_factor_vec=np.array([0.02,0.07])


    boxes = []
    for approx_factor in approx_factor_vec:
        for c in cnts[1]:
            p = cv2.arcLength(c, True)
            if p >= (min_box_width+min_box_height):
                approx = cv2.approxPolyDP(c, approx_factor * p, True)
                len_check=len(approx)
                if 4 <= len(approx) <= 8:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    aspect = w / float(h)
                    hl_color = (255, 255, 0)
                    if aspect > min_aspect and w > min_box_width_rel * img.shape[1] and w > min_box_width and h > min_box_height:
                        box = Box(max(0, x - pad + box_offset), max(0, y - pad + box_offset),
                                  min(img.shape[1] - 1, x - pad + w - box_offset),
                                  min(img.shape[0] - 1, y - pad + h - box_offset))
                        box.img_data = img[box.y1:box.y2, box.x1:box.x2]
                        boxes.append(box)
                        hl_color = (255, 0, 0)
                        cv2.rectangle(regs, (x, y), (x + w, y + h), hl_color, 2) # andrey
                        if debug:
                            cv2.imshow('image with bounding box', regs)  # andrey
                            cv2.waitKey(0)
                else:
                    cv2.drawContours(regs, [approx], 0, (0, 255, 0),2) # andrey
                    if debug:
                        cv2.imshow('image with contours', regs)  # andrey
                        cv2.waitKey(0)
        #show(regs)


    return [src_img, norm_img, regs], boxes


def filter_boxes(boxes):
    filtered = []
    while len(boxes) > 0:
        b1 = boxes[0]
        # box overlaps itself, so hierarchy will have at least one element
        hierarchy = list(filter(lambda b: b.overlaps(b1), boxes))
        hierarchy = list(sorted(hierarchy, key=lambda b: b.area(), reverse=True))
        largest = hierarchy[0]
        for i in range(1, len(hierarchy)):
            if (hierarchy[i - 1].y2 - hierarchy[i - 1].y1) > 2 * (hierarchy[i].y2 - hierarchy[i].y1):
                ## doubtful case, add smaller box
                filtered.append(hierarchy[i])
        filtered.append(largest)
        for hb in hierarchy:
            boxes.remove(hb)
    return filtered

def filter_boxes_updated(boxes,src_img,debug):
    if debug:
        regs = src_img.copy()
        regs1 = src_img.copy()
    #1. Find unique boxes first
    repeated_box=np.zeros((len(boxes),),dtype=int)
    unique_boxes=[]
    # find equal boxes in the list
    coord_tolerance = 5 #pix
    for i in range(0, len(boxes)):
        if  repeated_box[i]==0:
            b_i=boxes[i]
            for j in range(0, len(boxes)):
                if i!=j and repeated_box[j]==0:
                    b_j = boxes[j]
                    box_eq=b_i.equal(b_j,coord_tolerance)
                    if box_eq:
                        repeated_box[j]=1

    for i in range(0, len(repeated_box)):
        if repeated_box[i]==0:
            cur_box=boxes[i]
            unique_boxes.append(cur_box)
            if debug:
                w=box_width(cur_box)
                h=box_height(cur_box)
                x=cur_box.x1
                y = cur_box.y1
                hl_color = (255, 0, 0)
                cv2.rectangle(regs1, (x, y), (x + w, y + h), hl_color, 2)  # andrey
                #cv2.namedWindow('image with unique boxes', cv2.WINDOW_NORMAL)
                cv2.imshow('image with unique boxes', regs1)  # andrey
                cv2.waitKey(0)
                ty=1

    # 2. Find intersection between boxes and exclude the bigger box
    exclude_box = np.zeros((len(unique_boxes),), dtype=int)
    left_boxes = []

    # find equal boxes in the list
    tolerance_for_inner_box = coord_tolerance  # pixels=margins for every side
    intersected_boxes_are_tolerance=0.7
    for i in range(0, len(unique_boxes)):
        if exclude_box[i] == 0:
            b_i = unique_boxes[i]
            for j in range(0, len(unique_boxes)):
                if i != j and exclude_box[j] == 0:
                    b_j = unique_boxes[j]
                    ## Check if box is inside the others
                    if not (b_i.inside_fixed(b_j,tolerance_for_inner_box)):
                        # If b_i is not inside the b_j check the intersection
                        box_intersection = b_i.intersection(b_j)
                        if box_intersection is not None:
                            # If the intersection between the boxes is equal to one of the boxes, then exclude the other box
                            box_eq_to_bi = box_intersection.equal(b_i, coord_tolerance)
                            box_eq_to_bj = box_intersection.equal(b_j, coord_tolerance)
                            b_i_area=b_i.area()
                            b_j_area = b_j.area()
                            num_common_sides=b_i.common_sides(b_j,coord_tolerance)
                            #if num_common_sides == 0:# one box is inside another
                            #    continue
                            if num_common_sides <= 2:
                                if box_eq_to_bi:
                                    exclude_box[i] = 1  # exclude the small box
                                if box_eq_to_bj:
                                    exclude_box[j] = 1  # exclude the small box
                            if num_common_sides >2:
                                if box_eq_to_bi :
                                    if b_i_area/b_j_area>=intersected_boxes_are_tolerance:
                                        exclude_box[i] = 1  # exclude the small box
                                    else :
                                        exclude_box[j] = 1  # exclude the big box
                                if box_eq_to_bj:
                                    if b_j_area/b_i_area>=intersected_boxes_are_tolerance:
                                        exclude_box[j] = 1  # exclude the small box
                                    else :
                                        exclude_box[i] = 1  # exclude the big box

    for i in range(0, len(exclude_box)):
        if exclude_box[i]==0:
            cur_box=unique_boxes[i]
            left_boxes.append(cur_box)
            if debug:
                w=box_width(cur_box)
                h=box_height(cur_box)
                x=cur_box.x1
                y = cur_box.y1
                hl_color = (255, 0, 0)
                cv2.rectangle(regs, (x, y), (x + w, y + h), hl_color, 2)  # andrey
                #cv2.namedWindow('image with left boxes', cv2.WINDOW_NORMAL)
                cv2.imshow('image with left boxes', regs)  # andrey
                cv2.waitKey(0)
                ty=1

    return left_boxes


def detect_border(arr, axis=0, border_threshold=0.9):
    sum_val=np.sum(arr, axis=axis)
    thr_val=border_threshold * arr.shape[axis] * 255
    b_loc = np.nonzero(np.sum(arr, axis=axis) > border_threshold * arr.shape[axis] * 255)[0]
    if len(b_loc) > 0:
        return b_loc[0], b_loc[-1]
    return None, None


def crop_border(img, debug_img=None, border_zone_h=0.05, border_zone_v=0.2, pad_if_border=1, debug=False):
    """
    :param img: grayscale image with pixel values 0 or 255
    :param border_zone:
    :return:
    """
    # invert black on white image
    # if np.sum(img == 255) > np.sum(img == 0):
    #     img = cv2.bitwise_not(img)

    if debug:
        cv2.imshow('image', img)  # andrey
        cv2.waitKey(0)
    b_zone_left, b_zone_top, b_zone_right, b_zone_bottom = int(img.shape[1] * border_zone_h), int(
        img.shape[0] * border_zone_v), int(img.shape[1] * (1 - border_zone_h)), int(img.shape[0] * (1 - border_zone_v))
    b_left = detect_border(img[:, :b_zone_left], axis=0)[-1]
    b_left = 0 if (b_left is None or b_left == b_zone_left - 1) else (b_left + 1 + pad_if_border)

    b_right = detect_border(img[:, b_zone_right:], axis=0)[0]
    b_right = img.shape[1] if (b_right is None or b_right == 0) else (b_zone_right + b_right - pad_if_border)

    b_top = detect_border(img[:b_zone_top, :], axis=1)[-1]
    b_top = 0 if (b_top is None or b_top == b_zone_top - 1) else (b_top + 1 + pad_if_border)

    b_bottom = detect_border(img[b_zone_bottom:, :], axis=1)[0]
    b_bottom = img.shape[0] if (b_bottom is None or b_bottom == 0) else (b_zone_bottom + b_bottom - pad_if_border)

    if debug:
        show((cv2.rectangle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), (b_left, b_top), (b_right - 1, b_bottom - 1),
                            color=(255, 0, 0)) / 2 + cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) / 2).astype(np.uint8),
             ref_imgs=[debug_img])

    return b_left, b_right, b_top, b_bottom


def annotate_boxes(start_id, boxes, ann_color):
    bid = start_id
    for b in boxes:
        b.meta = ann_color
        b.id = bid
        bid += 1
    return bid


def get_articles(input):
    articles = utils.get_metadata(input, utils.image_ext,
                                  lambda: [Article(id=-1, chain_id='', filename='', title='', images=[])],
                                  lambda x: x[0].images,
                                  lambda id, chain_id, filename: ArticleImage(id=id, chain_id=str(uuid.uuid4()),
                                                                              filename=filename, regions=[], title='',page=0,idx_on_page=0))
                                                                        #ItJim: ^this part didn't work because was lacking parameters.
    return articles


def get_file_mapping(articles):
    filelist = {}
    for a in articles:
        for i in a.images:
            filelist[i.filename] = i
    if len(filelist) == 0:
        print('None of input files exist')
        exit(-1)
    return filelist


def main(args):
    print(f'Source region extraction tool version {__version__}')
    a = ArgumentParser()
    a.add_argument('input',
                   help='figure image file or image folder path or semicolon separated list of files or meta json lines file')
    a.add_argument('--demo', help='highlight source region on input images and save separately',
                   default=False, action='store_true')
    a.add_argument('-out', help='source regions output dir',
                   default='out')
    a.add_argument('-meta_out', help='source regions output JSONL file',
                   default='regions.json')
    args = a.parse_args(args)
    demo_dir = args.out + '/demo'
    print('Building a list of source files...')
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir, ignore_errors=True)
    if args.demo:
        os.makedirs(demo_dir + '/', exist_ok=True)
    if args.out:
        if os.path.exists(args.out):
            shutil.rmtree(args.out, ignore_errors=True)
        try:
            os.makedirs(args.out, exist_ok=True)
            if args.demo:
                os.makedirs(demo_dir + '/', exist_ok=True)# andrey
        except:
            pass

    #if (args.input[-1]!='\\'):
    articles = get_articles(args.input)
    file_mapping = get_file_mapping(articles)
    #else:
    #    file_mapping = {}
    #    for (dirpath, dirnames, filenames) in os.walk(args.input):
    #        for fnam in filenames:
    #            file_mapping[args.input+fnam] = ArticleImage(id=1,chain_id='0',filename=args.input+fnam,title='',page=1,idx_on_page=1,regions=[])
    #        break
    if len(file_mapping) == 0:
        print('No files found in specified sources')
        exit(-1)
    reg_id = 0
    for i, filename in enumerate(file_mapping.keys()):
        src_img = cv2.imread(filename)
        if src_img is None:
            print(f'Error reading file, skipping: {filename}')
            continue
        cur_id = 1
        # extract with default settings
        boxes, boxes2, boxes3,boxes3a, boxes4 = [], [], [], [], []
        PerformDebug=False
        annotations, boxes = get_region_boxes(src_img, bin_thresh=5,debug=PerformDebug)
        cur_id = annotate_boxes(cur_id, boxes, (255, 0, 0))
        # preset 2 to extract bordered regions
        annotations2, boxes2 = get_region_boxes(src_img, erode_iters=0, bin_thresh=50,debug=PerformDebug)
        cur_id = annotate_boxes(cur_id, boxes2, (0, 255, 0))
        # preset 3 with another threshold
        annotations3, boxes3 = get_region_boxes(src_img, erode_iters=0, bin_thresh=100,debug=PerformDebug)
        cur_id = annotate_boxes(cur_id, boxes3, (0, 0, 255))
        # preset 3a with another threshold
        annotations3a, boxes3a = get_region_boxes(src_img, erode_iters=0, bin_thresh=150, debug=PerformDebug)
        cur_id = annotate_boxes(cur_id, boxes3a, (0, 0, 255))
        # preset 4 - low threshold no morph
        annotations4, boxes4 = get_region_boxes(src_img, erode_iters=0,debug=PerformDebug)
        cur_id = annotate_boxes(cur_id, boxes4, (255, 0, 255))
        #boxes = filter_boxes(boxes + boxes2 + boxes3 + boxes3a + boxes4)
        all_boxes=boxes + boxes2 + boxes3 + boxes3a + boxes4
        # Check the number of all boxes
        if len(all_boxes) == 1:  # add the entire image as a box
            box = Box(0, 0, src_img.shape[1] - 1, src_img.shape[0] - 1)
            box.img_data = src_img
            all_boxes.append(box)
        PerformDebug = False
        updated_boxes = filter_boxes_updated(all_boxes,src_img,debug=PerformDebug)
        # boxes = boxes + boxes2 + boxes3
        if len(updated_boxes) == 0:
            print(f'Warning: {filename} no source regions detected, assuming single source region')
            box = Box(0, 0, src_img.shape[1]-1, src_img.shape[0]-1, id=cur_id)
            box.img_data = src_img
            updated_boxes.append(box)

        # remove borders from regions
        print(f'{filename} number of source regions: {len(updated_boxes)}')
        for b in updated_boxes:
            src_reg = b.img_data
            assert src_reg.shape[:2] == b.img_data.shape[:2]
            # crop borders on normalized reg (white/black borders)

            PerformDebug = False
            x1, x2, y1, y2 = crop_border(preprocess_image(b.img_data, 150, otsu=True), src_reg,debug=PerformDebug)
            sr = src_reg[y1:y2, x1:x2]
            # sr = src_img[b.y1:b.y2, b.x1:b.x2]
            reg_path = os.path.join(args.out,
                                    '.'.join(os.path.basename(filename).split('.')[:-1]) + '_' + 'R' + str(
                                        b.id) + '.png')
            article_image = file_mapping[filename]
            article_image.regions.append(
                Region(id=reg_id, chain_id=article_image.chain_id, filename=os.path.abspath(reg_path),
                       box=(b.x1, b.y1, b.x2, b.y2), matches=[],
                       title=os.path.basename(reg_path)))
            reg_id += 1
            #cv2.imshow('subimage', sr)  # andrey
            #cv2.waitKey(0)
            cv2.imwrite(reg_path, sr)

        if args.demo:
            ref = src_img.copy()
            for b in updated_boxes:
                cv2.rectangle(ref, (b.x1, b.y1), (b.x2, b.y2),color=(b.meta if b.meta is not None else (255, 128, 100)),
                                  thickness=2)
            ResultImagePathToSave=os.path.join(demo_dir, '.'.join(os.path.basename(filename).split('.')[:-1]) + '.png')
            cv2.imwrite(ResultImagePathToSave, ref)
            cv2.imshow('image with bounding box', ref)  # andrey
            cv2.waitKey(0)
        progress.report_progress(i + 1, len(file_mapping), 'Extracting regions')
        #show(ref_imgs=annotations, title=f'{filename}')
        #show(ref_imgs=annotations2, title=f'{filename} pass 2')
        #show(ref_imgs=annotations3, title=f'{filename} pass 3')
        #show(ref_imgs=annotations4, title=f'{filename} pass 4')
    with open(args.meta_out, mode='w') as meta_file:
        for a in articles:
            meta_file.write(a.toJSON() + os.linesep)


if __name__ == '__main__':
    main(sys.argv[1:])
