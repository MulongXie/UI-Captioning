import time
from os.path import join as pjoin
from glob import glob
import os
import json
import sys
import shutil
import warnings
import cv2
import numpy as np

from utils.classification.IconClassifier import IconClassifier
from utils.classification.IconCaption import IconCaption
from utils.llm.Openai import OpenAI
from utils.llm.Summarizer import Summarizer
from module.GUI import GUI
sys.path.append('utils/classification')
warnings.filterwarnings("ignore", category=Warning)


def rico_sca_data_generation(rico_sca_dir='C:/Mulong/Data/rico/rico_sca', rico_data_dir='D:/Mulong/Datasets/gui/rico/combined/all'):
    ui_no = open(pjoin(rico_sca_dir, 'rico_sca.txt'), 'r')
    for line in ui_no.readlines():
        ui_name = line.split('.')[0]
        shutil.copy(pjoin(rico_data_dir, ui_name + '.jpg'), pjoin(rico_sca_dir, ui_name + '.jpg'))
        shutil.copy(pjoin(rico_data_dir, ui_name + '.json'), pjoin(rico_sca_dir, ui_name + '.json'))


def check_annotations(start_gui_no, end_gui_no,
                      rico_dir='C:/Mulong/Data/rico/rico_sca',
                      gui_dir='C:/Mulong/Data/ui captioning',
                      annotation_dir='C:/Mulong/Data/ui captioning/annotation',
                      revision_dir='C:/Mulong/Data/ui captioning/annotation-revision'):
    annotation_files = glob(pjoin(annotation_dir, '*'))
    annotation_files = sorted(annotation_files, key=lambda x: int(os.path.basename(x).split('_')[0]))
    for file in annotation_files[start_gui_no: end_gui_no]:
        # load annotation
        annotation = json.load(open(file, 'r', encoding='utf-8'))
        print('[File]:', file)
        print('[Factor]:', annotation['factor'])
        print('[Caption]', annotation['annotation'])
        print('*** Press "q" to quit, "r" to revise, anything else to continue ***')
        # show gui
        gui_img_file = pjoin(rico_dir, annotation['gui-no'] + '.jpg')
        gui_vh_file = pjoin(rico_dir, annotation['gui-no'] + '.json')
        gui = GUI(gui_img_file, gui_vh_file, gui_dir, resize=(1440, 2560))
        gui.load_elements()
        key = gui.show_all_elements()
        if key == ord('q'):
            cv2.destroyWindow('elements')
            break
        elif key == ord('r'):
            annotation['revised'] = True
            annotation['annotation'] = input('Revision:')
            revision_file = pjoin(revision_dir, annotation['gui-no'] + '_' + annotation['factor'] + '.json')
            json.dump(annotation, open(revision_file, 'w', encoding='utf-8'), indent=4)
            print('*** Revision save to %s ***' % revision_file)
            cv2.destroyWindow('elements')
        print('\n')


def check_annotation_by_ui_id(gui_id,
                              rico_dir='C:/Mulong/Data/rico/rico_sca',
                              gui_dir='C:/Mulong/Data/ui captioning',
                              annotation_dir='C:/Mulong/Data/ui captioning/annotation'):
    # load annotation
    annotation_files = glob(pjoin(annotation_dir, str(gui_id) + '*.json'))
    for file in annotation_files:
        annotation = json.load(open(file, 'r', encoding='utf-8'))
        print('***********')
        print('[Factor]:', annotation['factor'])
        print('[Caption]', annotation['annotation'])
    # show gui
    gui_id = str(gui_id)
    gui_img_file = pjoin(rico_dir, gui_id + '.jpg')
    gui_vh_file = pjoin(rico_dir, gui_id + '.json')
    gui = GUI(gui_img_file, gui_vh_file, gui_dir, resize=(1440, 2560))
    gui.load_elements()
    gui.show_all_elements()
    cv2.destroyWindow('elements')
    return gui


class DataCollector:
    def __init__(self, input_dir, output_dir, gui_img_resize=(1440, 2560), engine_model='gpt-3.5-turbo-16k'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_annotation_dir = pjoin(self.output_dir, 'annotation')
        os.makedirs(self.output_annotation_dir, exist_ok=True)
        self.img_files = sorted(glob(pjoin(input_dir, '*.jpg')), key=lambda x: int(os.path.basename(x).split('.')[0]))
        self.vh_files = sorted(glob(pjoin(input_dir, '*.json')), key=lambda x: int(os.path.basename(x).split('.')[0]))

        self.gui_img_resize = gui_img_resize
        self.gui_detection_models = {'classification':IconClassifier(model_path='./utils/classification/model_results/best-0.93.pt', class_path='./utils/classification/model_results/iconModel_labels.json'),
                                     'caption':IconCaption(vocab_path='./utils/classification/model_results/vocab_idx2word.json',  model_path='./utils/classification/model_results/labeldroid.pt')}  # {IconClassification, IconCaption}

        self.llm_engine = OpenAI(model=engine_model)
        self.llm_summarizer = Summarizer(self.llm_engine)

        self.turn_on_revision = True   # True to turn on revision
        self.revise_stop_point = 1     # the UI number where the revision stops
        self.annotations = []
        self.revise_suggestions = ''
        self.annotation_factors = ['Key Element', 'Functionality', 'Layout', "Accessibility"]

    '''
    ********************
    *** GUI Analysis ***
    ********************
    '''
    def analyze_gui(self, gui_img_file, gui_json_file, show=False):
        gui = GUI(gui_img_file=gui_img_file, gui_json_file=gui_json_file, output_file_root=self.output_dir,
                  resize=self.gui_img_resize,
                  model_icon_caption=self.gui_detection_models['caption'],
                  model_icon_classification=self.gui_detection_models['classification'])
        gui.ui_info_extraction()
        gui.ui_analysis_elements_description()
        gui.ui_element_tree()
        if show:
            gui.show_all_elements()
        return gui

    def load_gui_analysis(self, gui_img_file, gui_json_file, show=False):
        gui = GUI(gui_img_file=gui_img_file, gui_json_file=gui_json_file, output_file_root=self.output_dir, resize=self.gui_img_resize)
        gui.load_elements()
        if show:
            gui.show_all_elements()

    '''
    **********************************************
    *** GUI Annotation V1 - Human Ground Truth ***
    **********************************************
    '''
    def annotate_gui_human_gt(self, gui_img_file, gui_json_file, factor, load_gui=True, show_gui=False):
        # 1. analyze GUI
        if not load_gui:
            print('*** GUI Analysis ***')
            gui = self.analyze_gui(gui_img_file, gui_json_file, show=False)
        else:
            print('*** Load GUI Info ***')
            gui = GUI(gui_img_file=gui_img_file, gui_json_file=gui_json_file, output_file_root=self.output_dir, resize=self.gui_img_resize)
            gui.load_elements()
        ann_result = {'gui-no': gui.gui_no, 'factor': factor, 'element-tree': str(gui.element_tree)}

        # 2. generate summarization by llm
        self.llm_summarizer.wrap_previous_annotations_as_examples(self.annotations[-3:])
        summarization = self.llm_summarizer.summarize_gui(gui, factor=factor)

        # 3. annotation revision
        print('*** Summarization [' + factor + '] ***\n', summarization)
        if show_gui:
            key = gui.show_all_elements()
            if key == ord('q'):
                cv2.destroyWindow('elements')
                return None
        if self.turn_on_revision:
            revise = input('Do you want to revise the summarization? ("y" or "n"): ')
            if revise.lower() == 'y':
                print('*** Revision ***')
                ground_truth = input('Input your ground truth: ')
                revision_suggestion = input('Input revision points: ')
                ann_result['revised'] = True
                ann_result['annotation'] = ground_truth
                ann_result['revision-suggestion'] = revision_suggestion
            else:
                ann_result['revised'] = False
                ann_result['annotation'] = summarization
                turn_off = input('Do you want to turn off revision from now? ("y" or "n"): ')
                if turn_off.lower() == 'y':
                    self.turn_on_revision = False
        else:
            ann_result['revised'] = False
            ann_result['annotation'] = summarization
        json.dump(ann_result, open(pjoin(self.output_annotation_dir, str(gui.gui_no) + '_' + factor + '.json'), 'w', encoding='utf-8'), indent=4)
        self.annotations.append(ann_result)
        return ann_result

    def annotate_all_guis_human_gt(self, start_gui_no, end_gui_no, factor_id, load_gui=False, show_gui=False, turn_on_revision=True, wait_time=2):
        '''
        :param start_gui_no: int, start gui file name
        :param end_gui_no: int, end gui file name
        :param factor_id: factor to annotate, ['Key Element', 'Functionality', 'Layout', "Accessibility"]
        :param load_gui: whether to load an existing GUI analysis result
        :param show_gui: whether to show the GUI while annotating
        :param turn_on_revision: whether to offer the chance to revise the summarization at all
        :param wait_time: time to wait in each iteration to avoid over-frequent request
        '''
        self.turn_on_revision = turn_on_revision
        for i, gui_img_file in enumerate(self.img_files[start_gui_no: end_gui_no]):
            gui_vh_file = self.vh_files[start_gui_no + i]
            print('\n\n=== Annotating (press "q" to quit) === [%d / %d] %s' % (i+start_gui_no, end_gui_no, gui_img_file))
            if not self.annotate_gui_human_gt(gui_img_file, gui_vh_file, factor=self.annotation_factors[factor_id], load_gui=load_gui, show_gui=show_gui):
                break
            time.sleep(wait_time)

    '''
    ****************************************
    *** GUI Annotation V2 - GPT Revision ***
    ****************************************
    '''
    def annotate_gui_gpt_revision(self, gui_img_file, gui_json_file, factor, load_gui=True):
        '''
        Annotate gui with recursive gui revision
        '''
        # 1. analyze GUI
        if not load_gui:
            print('\n*** GUI Analysis ***')
            gui = self.analyze_gui(gui_img_file, gui_json_file, show=False)
        else:
            print('\n*** Load GUI Info ***')
            gui = GUI(gui_img_file=gui_img_file, gui_json_file=gui_json_file, output_file_root=self.output_dir, resize=self.gui_img_resize)
            gui.load_elements()

        # 2. iterative annotation revision
        annotation = {'gui-no': gui.gui_no, 'factor': factor, 'element-tree': str(gui.element_tree),
                      'annotation-history': [], 'revision-suggestion-history': [],
                      'annotation': '', 'revision-suggestion': ''}
        prev_ann = [self.annotations[np.random.randint(0, max(1, self.revise_stop_point))]] if len(self.annotations) > 0 else []
        self.llm_summarizer.wrap_previous_annotations_as_examples(prev_ann)
        while True:
            summarization = self.llm_summarizer.summarize_gui_with_revise_suggestion(gui, factor, annotation)
            annotation['annotation-history'].append(summarization)
            print(summarization)
            if self.turn_on_revision:
                gui.show_all_elements()
                revise = input('Do you want to revise the summarization? ("y" or "n"): ')
                if revise.lower() == 'y':
                    print('\n*** Revision Suggestions ***')
                    revision_suggestion = input('-- Input revision points: ')
                    annotation['revision-suggestion-history'].append(revision_suggestion)
                    cv2.destroyWindow('elements')
                else:
                    cv2.destroyWindow('elements')
                    turn_off = input('Do you want to turn off revision from now? ("y" or "n"): ')
                    if turn_off.lower() == 'y':
                        self.turn_on_revision = False
                    self.revise_stop_point += 1
                    break
            else:
                break
        annotation['annotation'] = annotation['annotation-history'][-1]
        annotation['revision-suggestion'] = self.revise_suggestions + ' - '.join(annotation['revision-suggestion-history'])
        self.revise_suggestions = annotation['revision-suggestion']
        json.dump(annotation, open(pjoin(self.output_annotation_dir, str(gui.gui_no) + '_' + factor + '.json'), 'w', encoding='utf-8'), indent=4)
        self.annotations.append(annotation)
        return annotation

    def annotate_all_guis_gpt_revision(self, start_gui_no, end_gui_no, factor_id, load_gui=False, turn_on_revision=True, wait_time=2):
        self.turn_on_revision = turn_on_revision
        for i, gui_img_file in enumerate(self.img_files[start_gui_no: end_gui_no]):
            gui_vh_file = self.vh_files[start_gui_no + i]
            print('\n\n+++ Annotating +++ [%d / %d] %s' % (i+start_gui_no, end_gui_no, gui_img_file))
            self.annotate_gui_gpt_revision(gui_img_file, gui_json_file=gui_vh_file, factor=self.annotation_factors[factor_id], load_gui=load_gui)
            time.sleep(wait_time)

    def load_annotations(self, start_gui_no, end_gui_no, factor_id, revise_stop_point):
        self.revise_stop_point = revise_stop_point
        factor = self.annotation_factors[factor_id]
        for i in range(start_gui_no, end_gui_no):
            file_name = pjoin(self.output_annotation_dir, str(i) + '_' + factor + '.json')
            if os.path.exists(file_name):
                print('Load', file_name)
                annotation = json.load(open(file_name, 'r', encoding='utf-8'))
                self.annotations.append(annotation)
                self.revise_suggestions = annotation['revision-suggestion']


if __name__ == '__main__':
    data = DataCollector(input_dir='C:/Mulong/Data/rico/rico_sca',
                         output_dir='C:/Mulong/Data/ui captioning - rs',
                         engine_model='gpt-3.5-turbo-16k')

    # Option 1. single annotate_gui_gpt_revision
    # data.annotate_gui_gpt_revision(gui_img_file='data/rico/raw/2.jpg',
    #                                gui_json_file='data/rico/raw/2.json',
    #                                factor='key elements', load_gui=False)

    # Option 2. multiple annotate_all_guis_gpt_revision
    data.load_annotations(start_gui_no=0, end_gui_no=5, factor_id=0, revise_stop_point=1)
    data.annotate_all_guis_gpt_revision(start_gui_no=0, end_gui_no=10,
                                        factor_id=0,
                                        load_gui=False,
                                        turn_on_revision=True)
