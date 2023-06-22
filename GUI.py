import os
import cv2
import json
from os.path import join as pjoin
import copy

from utils.classification.IconClassifier import IconClassifier
from utils.classification.IconCaption import IconCaption
from utils.ocr.text_detection import text_detection

import sys
import warnings
sys.path.append('utils/classification')
warnings.filterwarnings("ignore", category=Warning)


class GUI:
    def __init__(self, gui_img_file, gui_json_file, output_file_root='data/twitter/testcase1',
                 resize=(1080, 2280), model_icon_caption=None, model_icon_classification=None):
        self.img_file = gui_img_file
        self.json_file = gui_json_file
        self.gui_no = gui_img_file.replace('/', '\\').split('\\')[-1].split('.')[0]

        self.resize = resize
        self.img = cv2.resize(cv2.imread(gui_img_file), resize)      # resize the image to be consistent with the vh
        self.json = json.load(open(gui_json_file, 'r', encoding='utf-8'))  # json data, the view hierarchy of the GUI

        self.element_id = 0
        self.elements = []          # list of element in dictionary {'id':, 'class':...}
        self.elements_leaves = []   # leaf nodes that does not have children
        self.element_tree = None    # structural element tree, dict type
        self.blocks = []            # list of blocks from element tree
        self.removed_node_no = 0    # for the record of the number of removed nodes

        self.ocr_text = []               # GUI ocr detection result, list of texts {}
        self.model_icon_caption = model_icon_caption   # IconCaption
        self.model_icon_classification = model_icon_classification  # IconClassification

        # output file paths
        self.output_dir = pjoin(output_file_root, 'guidata')
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file_path_elements = pjoin(self.output_dir, self.gui_no + '_elements.json')
        self.output_file_path_element_tree = pjoin(self.output_dir, self.gui_no + '_tree.json')

    def load_elements(self, file_path_elements=None, file_path_element_tree=None):
        if not file_path_elements: file_path_elements = self.output_file_path_elements
        if not file_path_element_tree: file_path_element_tree = self.output_file_path_element_tree

        if not os.path.exists(file_path_elements) or not os.path.exists(file_path_element_tree):
            print('Loading FAILED, No such file:', file_path_elements, file_path_element_tree)

        print('Load elements from', file_path_elements)
        self.elements = json.load(open(file_path_elements, 'r', encoding='utf-8'))           # => self.elements
        self.gather_leaf_elements()                     # => self.elements_leaves
        self.element_id = self.elements[-1]['id'] + 1         # => self.element_id
        print('Load element tree from', file_path_element_tree)
        self.element_tree = json.load(open(file_path_element_tree, 'r', encoding='utf-8'))   # => self.element_tree

    '''
    **************************
    *** UI Info Extraction ***
    **************************
    '''
    def ui_info_extraction(self):
        '''
        Extract elements from raw view hierarchy Json file and store them as dictionaries
        => self.elements; self.elements_leaves
        '''
        # print('--- Extract elements from VH ---')
        json_cp = copy.deepcopy(self.json)
        element_root = json_cp['activity']['root']
        element_root['class'] = 'root'
        # clean up the json tree to remove redundant layout node
        self.prone_invalid_children(element_root)
        self.remove_redundant_nesting(element_root)
        self.merge_element_with_single_leaf_child(element_root)
        self.extract_children_elements(element_root, 0)
        self.revise_elements_attrs()
        self.gather_leaf_elements()
        # json.dump(self.elements, open(self.output_file_path_elements, 'w', encoding='utf-8'), indent=4)
        # print('Save elements to', self.output_file_path_elements)

    def prone_invalid_children(self, element):
        '''
        Prone invalid children elements
        Leave valid children and prone their children recursively
        Take invalid children's children as its own directly
        '''
        def check_if_element_valid(ele, min_length=5):
            '''
            Check if the element is valid and should be kept
            '''
            if (ele['bounds'][0] >= ele['bounds'][2] - min_length or ele['bounds'][1] >= ele['bounds'][
                3] - min_length) or \
                    ('layout' in ele['class'].lower() and not ele['clickable']):
                return False
            return True

        valid_children = []
        if 'children' in element:
            for child in element['children']:
                if check_if_element_valid(child):
                    valid_children.append(child)
                    self.prone_invalid_children(child)
                else:
                    valid_children += self.prone_invalid_children(child)
                    self.removed_node_no += 1
            element['children'] = valid_children
        return valid_children

    def remove_redundant_nesting(self, element):
        '''
        Remove redundant parent node whose bounds are same
        '''
        if 'children' in element and len(element['children']) > 0:
            redundant = False
            new_children = []
            for child in element['children']:
                # inherit clickability
                if element['clickable']:
                    child['clickable'] = True
                # recursively inspect child node
                new_children += self.remove_redundant_nesting(child)
                if child['bounds'] == element['bounds']:
                    redundant = True
            # only return the children if the node is redundany
            if redundant:
                self.removed_node_no += 1
                return new_children
            else:
                element['children'] = new_children
        return [element]

    def merge_element_with_single_leaf_child(self, element):
        '''
        Keep the resource-id and class and clickable of the child element
        '''
        if 'children' in element:
            if len(element['children']) == 1 and 'children' not in element['children'][0]:
                child = element['children'][0]
                element['resource-id'] = child['resource-id'] if 'resource-id' in child else ''
                element['class'] = child['class']
                element['clickable'] = child['clickable']
                self.removed_node_no += 1
                del element['children']
            else:
                new_children = []
                for child in element['children']:
                    new_children.append(self.merge_element_with_single_leaf_child(child))
                element['children'] = new_children
        return element

    def extract_children_elements(self, element, layer):
        '''
        Recursively extract children from an element
        '''
        element['id'] = self.element_id
        element['layer'] = layer
        self.elements.append(element)
        children_depth = layer  # record the depth of the children
        if 'children' in element and len(element['children']) > 0:
            element['children-id'] = []
            for child in element['children']:
                self.element_id += 1
                element['children-id'].append(self.element_id)
                children_depth = max(children_depth, self.extract_children_elements(child, layer+1))
            element['children-depth'] = children_depth
            # replace wordy 'children' with 'children-id'
            del element['children']
        if 'ancestors' in element:
            del element['ancestors']
        return children_depth

    def revise_elements_attrs(self):
        '''
        Revise some attributes in elements
        1. add "area"
        2. keep "content-desc" as a string
        '''
        for ele in self.elements:
            bounds = ele['bounds']
            ele['area'] = {'height': int(bounds[2] - bounds[0]), 'length': int(bounds[3] - bounds[1])}
            if 'content-desc' in ele and type(ele['content-desc']) == list:
                if not ele['content-desc'][0]:
                    ele['content-desc'] = ''
                else:
                    ele['content-desc'] = ','.join(ele['content-desc'])

    def gather_leaf_elements(self):
        i = 0
        for ele in self.elements:
            if 'children-id' not in ele:
                ele['leaf-id'] = i
                self.elements_leaves.append(ele)
                i += 1

    '''
    *******************
    *** UI Analysis ***
    *******************
    '''
    def ui_analysis_elements_description(self, ocr=True, caption=True, cls=True):
        '''
        Extract description for UI elements through 'text', 'content-desc', 'classification' and 'caption'
        => element['description']
        '''
        # print('--- Analyze UI elements ---')
        # use ocr to detect text
        if ocr: self.ocr_detect_gui_text()
        # generate caption for non-text elements
        if caption: self.caption_elements()
        # classify non-text elements
        if cls: self.classify_elements()
        # extract element description from 'text', 'content-desc', 'icon-cls' and 'caption'
        for ele in self.elements_leaves:
            description = ''
            # check text
            if 'text' in ele and len(ele['text']) > 0:
                description += ele['text']
            # check content description
            if 'content-desc' in ele and len(ele['content-desc']) > 0 and ele['content-desc'] != ele['text']:
                description = ele['content-desc'] if len(description) == 0 else description + ' / ' + ele['content-desc']
            # if no text and content description, check caption
            if len(description) == 0:
                if 'icon-cls' in ele and ele['icon-cls']:
                    description = ele['icon-cls']
                elif 'caption' in ele and '<unk>' not in ele['caption']:
                    description = ele['caption']
                else:
                    description = None
            ele['description'] = description
        # save the elements with 'description' attribute
        json.dump(self.elements, open(self.output_file_path_elements, 'w', encoding='utf-8'), indent=4)
        # print('Save elements to', self.output_file_path_elements)

    def ocr_detect_gui_text(self):
        scale_w = self.resize[0] / self.img.shape[1]
        scale_h = self.resize[1] / self.img.shape[0]

        def scale_text_bounds(bounds):
            return [bounds[0] * scale_w, bounds[1] * scale_h,
                    bounds[2] * scale_w, bounds[3] * scale_h]

        def match_text_and_element(ele):
            '''
            Match ocr text and element through iou
            '''
            for text in self.ocr_text:
                t_b, e_b = text['bounds'], ele['bounds']
                # calculate intersected area between text and element
                intersected = max(0, min(t_b[2], e_b[2]) - max(t_b[0], e_b[0])) * max(0, min(t_b[3], e_b[3]) - max(t_b[1], e_b[1]))
                if intersected > 0:
                    ele['ocr'] = ' '.join([ele['ocr'], text['content']])
                    ele['text'] = ' '.join([ele['text'], text['content']])

        # google ocr detection for the GUI image
        self.ocr_text = text_detection(self.img_file)
        for t in self.ocr_text:
            t['bounds'] = scale_text_bounds(t['bounds'])

        # merge text to elements according to position
        for element in self.elements_leaves:
            if 'text' not in element or element['text'] == '':
                element['ocr'] = ''
                element['text'] = ''
                match_text_and_element(element)

    def caption_elements(self, elements=None):
        if self.model_icon_caption is None:
            self.model_icon_caption = IconCaption(vocab_path='utils/classification/model_results/vocab_idx2word.json',
                                                  model_path='utils/classification/model_results/labeldroid.pt')
        elements = self.elements_leaves if elements is None else elements
        clips = []
        for ele in elements:
            bound = ele['bounds']
            clips.append(self.img[bound[1]: bound[3], bound[0]:bound[2]])
        captions = self.model_icon_caption.predict_images(clips)
        for i, ele in enumerate(elements):
            ele['caption'] = captions[i]

    def classify_elements(self, elements=None):
        if self.model_icon_classification is None:
            self.model_icon_classification = IconClassifier(model_path='utils/classification/model_results/best-0.93.pt',
                                                            class_path='utils/classification/model_results/iconModel_labels.json')
        elements = self.elements_leaves if elements is None else elements
        clips = []
        for ele in elements:
            bound = ele['bounds']
            clips.append(self.img[bound[1]: bound[3], bound[0]:bound[2]])
        classes = self.model_icon_classification.predict_images(clips)
        for i, ele in enumerate(elements):
            if classes[i][1] > 0.95:
                ele['icon-cls'] = classes[i][0]
            else:
                ele['icon-cls'] = None

    '''
    ***********************
    *** Structural Tree ***
    ***********************
    '''
    def ui_element_tree(self):
        '''
        Form a hierarchical element tree with a few key attributes to represent the vh
        => self.element_tree
        => self.blocks
        '''
        # print('--- Generate element tree ---')
        self.element_tree = self.combine_children_to_tree(self.elements[0])
        json.dump(self.element_tree, open(self.output_file_path_element_tree, 'w'), indent=4)
        # print('Save element tree to', self.output_file_path_element_tree)

    def combine_children_to_tree(self, element):
        element_cp = copy.deepcopy(element)
        if 'children-id' in element_cp:
            element_cp['children'] = []
            for c_id in element_cp['children-id']:
                element_cp['children'].append(self.combine_children_to_tree(self.elements[c_id]))
            self.select_ele_attr(element_cp, ['scrollable', 'id', 'resource-id', 'class', 'clickable', 'children', 'description', 'area'])
        else:
            self.select_ele_attr(element_cp, ['id', 'resource-id', 'class', 'clickable', 'children', 'description', 'area'])
        self.simplify_ele_attr(element_cp)
        return element_cp

    def select_ele_attr(self, element, selected_attrs):
        element_cp = copy.deepcopy(element)
        for key in element_cp.keys():
            if key == 'selected' and element[key]:
                continue
            if key not in selected_attrs or element[key] is None or element[key] == '':
                del(element[key])

    def simplify_ele_attr(self, element):
        if 'resource-id' in element:
            element['resource-id'] = element['resource-id'].replace('com', '')
            element['resource-id'] = element['resource-id'].replace('android', '')
            element['resource-id'] = element['resource-id'].replace('..', '.')
            element['resource-id'] = element['resource-id'].replace('.:', ':')
        if 'class' in element:
            element['class'] = element['class'].replace('android', '')
            element['class'] = element['class'].replace('..', '.')
            element['class'] = element['class'].replace('.:', ':')

    def get_ui_element_node_by_id(self, ele_id):
        ele_id = int(ele_id)
        if ele_id >= len(self.elements):
            print('No element with id', ele_id, 'is found')
            return None
        return self.search_node_by_id(self.element_tree, ele_id)

    def search_node_by_id(self, node, ele_id):
        if node['id'] == ele_id:
            return node
        if node['id'] > ele_id:
            return None
        if 'children' in node:
            last_child = None
            for child in node['children']:
                if child['id'] == ele_id:
                    return child
                if child['id'] > ele_id:
                    break
                last_child = child
            return self.search_node_by_id(last_child, ele_id)

    '''
    *********************
    *** Visualization ***
    *********************
    '''
    def show_each_element(self, only_leaves=False):
        board = self.img.copy()
        if only_leaves:
            elements = self.elements_leaves
            print(len(elements))
        else:
            elements = self.elements
        for ele in elements:
            print(ele['class'])
            print(ele, '\n')
            bounds = ele['bounds']
            clip = self.img[bounds[1]: bounds[3], bounds[0]: bounds[2]]
            color = (0,255,0) if not ele['clickable'] else (0,0,255)
            cv2.rectangle(board, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color, 3)
            cv2.imshow('clip', cv2.resize(clip, (clip.shape[1] // 3, clip.shape[0] // 3)))
            cv2.imshow('ele', cv2.resize(board, (board.shape[1] // 3, board.shape[0] // 3)))
            if cv2.waitKey() == ord('q'):
                break
        cv2.destroyAllWindows()

    def show_all_elements(self, only_leaves=False):
        board = self.img.copy()
        if only_leaves:
            elements = self.elements_leaves
        else:
            elements = self.elements
        for ele in elements:
            bounds = ele['bounds']
            color = (0,255,0) if not ele['clickable'] else (0,0,255)
            cv2.rectangle(board, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color, 3)
        cv2.imshow('elements', cv2.resize(board, (board.shape[1] // 3, board.shape[0] // 3)))
        cv2.waitKey()
        cv2.destroyWindow('elements')

    def show_element(self, element, show_children=True):
        board = self.img.copy()
        color = (0,255,0) if not element['clickable'] else (0,0,255)
        bounds = element['bounds']
        cv2.rectangle(board, (bounds[0], bounds[1]), (bounds[2], bounds[3]), color, 3)
        if show_children and 'children-id' in element:
            for c_id in element['children-id']:
                bounds = self.elements[c_id]['bounds']
                cv2.rectangle(board, (bounds[0], bounds[1]), (bounds[2], bounds[3]), (255,0,255), 3)
        cv2.imshow('element', cv2.resize(board, (board.shape[1] // 3, board.shape[0] // 3)))
        cv2.waitKey()
        cv2.destroyWindow('element')

    def show_element_by_id(self, ele_id, show_children=True):
        element = self.elements[ele_id]
        self.show_element(element, show_children)

    def show_screen(self):
        cv2.imshow('screen', self.img)
        cv2.waitKey()
        cv2.destroyWindow('screen')


if __name__ == '__main__':
    load = False
    gui = GUI(gui_img_file='data/rico/raw/0.png',
              gui_json_file='data/rico/raw/0.json',
              output_file_root='data/rico/guidata',
              resize=(1440, 2560))
    # load previous result
    if load:
        gui.load_elements()
    # process from scratch
    else:
        gui.ui_info_extraction()
        gui.ui_analysis_elements_description()
        gui.ui_element_tree()
    gui.show_all_elements(only_leaves=True)
