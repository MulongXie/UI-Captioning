from os.path import join as pjoin
from glob import glob
import os
import json

from utils.classification.IconClassifier import IconClassifier
from utils.classification.IconCaption import IconCaption
from utils.llm.Openai import OpenAI
from utils.llm.Summarizer import Summarizer
from GUI import GUI


class DataCollector:
    def __init__(self, input_dir, output_dir, gui_img_resize=(1440, 2560), engine_model='gpt-3.5-turbo'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.output_annotation_dir = pjoin(self.output_dir, 'annotation')
        os.makedirs(self.output_annotation_dir, exist_ok=True)
        self.img_files = sorted(glob(pjoin(input_dir, '*.jpg')))
        self.vh_files = sorted(glob(pjoin(input_dir, '*.json')))

        self.gui_img_resize = gui_img_resize
        self.gui_detection_models = {'classification':IconClassifier(model_path='./utils/classification/model_results/best-0.93.pt', class_path='./utils/classification/model_results/iconModel_labels.json'),
                                     'caption':IconCaption(vocab_path='./utils/classification/model_results/vocab_idx2word.json',  model_path='./utils/classification/model_results/labeldroid.pt')}  # {IconClassification, IconCaption}

        self.llm_engine = OpenAI(model=engine_model)
        self.llm_summarizer = Summarizer(self.llm_engine)

        self.annotations = []
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
    **********************
    *** GUI Annotation ***
    **********************
    '''
    def annotate_gui(self, gui_img_file, gui_json_file, factor, load_gui=True, show_gui=False):
        # 1. analyze GUI
        if not load_gui:
            print('*** GUI Analysis ***')
            gui = self.analyze_gui(gui_img_file, gui_json_file, show=show_gui)
        else:
            print('*** Load GUI Info ***')
            gui = GUI(gui_img_file=gui_img_file, gui_json_file=gui_json_file, output_file_root=self.output_dir, resize=self.gui_img_resize)
            gui.load_elements()

        ann_result = {'gui-no': gui.gui_no, 'factor':factor, 'element-tree': str(gui.element_tree)}

        # 2. generate summarization by llm
        self.llm_summarizer.wrap_previous_annotations_as_examples(self.annotations[-3:])
        summarization = self.llm_summarizer.summarize_gui(gui, factor=factor)

        # 3. annotation revision
        print('*** Summarization [' + factor + '] ***\n', summarization)
        if show_gui:
            key = gui.show_all_elements()
            if key == ord('q'):
                return None          
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
        json.dump(ann_result, open(pjoin(self.output_annotation_dir, str(gui.gui_no) + '_ann.json'), 'w', encoding='utf-8'), indent=4)
        self.annotations.append(ann_result)
        return ann_result

    def annotate_all_guis(self, start_gui_no, end_gui_no, factor_id, load_gui=False, show_gui=False):
        '''
        :param start_gui_no: int, start gui file name
        :param end_gui_no: int, end gui file name
        :param factor_id: factor to annotate, ['Key Element', 'Functionality', 'Layout', "Accessibility"]
        :param load_gui: whether to load an existing GUI analysis result
        :param show_gui: whether to show the GUI while annotating
        '''
        for i, gui_img_file in enumerate(self.img_files[start_gui_no: end_gui_no]):
            gui_vh_file = self.vh_files[i]
            print('\n=== Annotating (press "q" to quit) ===', gui_img_file)
            if not self.annotate_gui(gui_img_file, gui_vh_file, factor=self.annotation_factors[factor_id], load_gui=load_gui, show_gui=show_gui):
                break
