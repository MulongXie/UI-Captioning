from utils.ocr import ocr as ocr
from utils.ocr.Text import Text
import cv2
import json
import time
import os
from os.path import join as pjoin


def wrap_up_texts(texts):
    texts_json = []
    for i, text in enumerate(texts):
        loc = text.location
        t = {'id': i,
             'bounds': [loc['left'], loc['top'], loc['right'], loc['bottom']],
             'content': text.content}
        texts_json.append(t)
    return texts_json


def save_detection_json(file_path, texts, img_shape):
    f_out = open(file_path, 'w')
    output = {'img_shape': img_shape, 'texts': []}
    for text in texts:
        c = {'id': text.id, 'content': text.content}
        loc = text.location
        c['column_min'], c['row_min'], c['column_max'], c['row_max'] = loc['left'], loc['top'], loc['right'], loc['bottom']
        c['width'] = text.width
        c['height'] = text.height
        output['texts'].append(c)
    json.dump(output, f_out, indent=4)


def visualize_texts(org_img, texts, shown_resize_height=None, show=False, write_path=None):
    img = org_img.copy()
    for text in texts:
        text.visualize_element(img, line=2)

    img_resize = img
    if shown_resize_height is not None:
        img_resize = cv2.resize(img, (int(shown_resize_height * (img.shape[1]/img.shape[0])), shown_resize_height))

    if show:
        cv2.imshow('texts', img_resize)
        cv2.waitKey(0)
        cv2.destroyWindow('texts')
    if write_path is not None:
        cv2.imwrite(write_path, img_resize)
    return img_resize


def text_sentences_recognition(texts):
    '''
    Merge separate words detected by Google ocr into a sentence
    '''
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_on_same_line(text_b, 'h', bias_justify=0.2 * min(text_a.height, text_b.height), bias_gap=1.3 * max(text_a.word_width, text_b.word_width)):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()

    for i, text in enumerate(texts):
        text.id = i
    return texts


def merge_intersected_texts(texts):
    '''
    Merge intersected texts (sentences or words)
    '''
    changed = True
    while changed:
        changed = False
        temp_set = []
        for text_a in texts:
            merged = False
            for text_b in temp_set:
                if text_a.is_intersected(text_b, bias=2):
                    text_b.merge_text(text_a)
                    merged = True
                    changed = True
                    break
            if not merged:
                temp_set.append(text_a)
        texts = temp_set.copy()
    return texts


def text_cvt_orc_format(ocr_result):
    texts = []
    if ocr_result is not None:
        for i, result in enumerate(ocr_result):
            error = False
            x_coordinates = []
            y_coordinates = []
            text_location = result['boundingPoly']['vertices']
            content = result['description']
            for loc in text_location:
                if 'x' not in loc or 'y' not in loc:
                    error = True
                    break
                x_coordinates.append(loc['x'])
                y_coordinates.append(loc['y'])
            if error: continue
            location = {'left': min(x_coordinates), 'top': min(y_coordinates),
                        'right': max(x_coordinates), 'bottom': max(y_coordinates)}
            texts.append(Text(i, content, location))
    return texts


def text_filter_noise(texts):
    valid_texts = []
    for text in texts:
        if len(text.content) <= 1 and text.content.lower() not in ['a', ',', '.', '!', '?', '$', '%', ':', '&', '+']:
            continue
        valid_texts.append(text)
    return valid_texts


def text_detection(input_file='data/0.png', ocr_root='data/output', show=False, save=False):
    start = time.time()
    name = input_file.replace('\\', '/').split('/')[-1][:-4]
    img = cv2.imread(input_file)

    ocr_result = ocr.ocr_detection_google(input_file)
    texts = text_cvt_orc_format(ocr_result)
    texts = merge_intersected_texts(texts)
    texts = text_filter_noise(texts)
    texts = text_sentences_recognition(texts)
    if save:
        os.makedirs(ocr_root, exist_ok=True)
        visualize_texts(img, texts, shown_resize_height=800, show=show, write_path=pjoin(ocr_root, name+'.jpg'))
        save_detection_json(pjoin(ocr_root, name+'.json'), texts, img.shape)
        print("[Text Detection Completed in %.3f s] Input: %s Output: %s" % (time.time() - start, input_file, pjoin(ocr_root, name+'.json')))
    return wrap_up_texts(texts)


if __name__ == '__main__':
    print(text_detection())

