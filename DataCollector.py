from os.path import join as pjoin
from glob import glob

from utils.classification.IconClassifier import IconClassifier
from utils.classification.IconCaption import IconCaption
from utils.llm.Openai import OpenAI
from utils.llm.Summarizer import Summarizer
from GUI import GUI


class DataCollector:
    def __init__(self, input_dir, output_dir, gui_img_resize=(1440, 2560), engine_model='gpt-3.5-turbo'):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.img_files = glob(pjoin(input_dir, '*.jpg'))
        self.vh_files = glob(pjoin(input_dir, '*.json'))

        self.gui_img_resize = gui_img_resize
        self.gui_detection_models = {'classification':IconClassifier(model_path='./utils/classification/model_results/best-0.93.pt', class_path='./utils/classification/model_results/iconModel_labels.json'),
                                     'caption':IconCaption(vocab_path='./utils/classification/model_results/vocab_idx2word.json',  model_path='./utils/classification/model_results/labeldroid.pt')}  # {IconClassification, IconCaption}

        self.llm_engine = OpenAI(model=engine_model)
        self.llm_summarizer = Summarizer(self.llm_engine)

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
    *************************
    *** LLM Summarization ***
    *************************
    '''

