from GUI import GUI
import cv2
import time


class GUIDataManager:
    def __init__(self, app_name, test_case_no, output_file_root='data'):
        self.app_name = app_name
        self.test_case_no = test_case_no
        self.output_file_root = output_file_root

        self.gui_detection_models = {'classification':None, 'caption':None}  # {IconClassification, IconCaption}
        self.gui = None
        self.device = None
        self.init_device()

    def get_current_gui_from_device(self, set_gui_no=0, gui_detection=True, show_gui_ele=False):
        if not self.device:
            print('No Device')
            return
        self.device.gui_no = set_gui_no
        self.device.reset_file_path_by_gui_no(gui_no=set_gui_no)
        self.collect_gui()
        self.analyze_gui(gui_detection, show_gui_ele)
        self.add_system_info_in_tree()
        return self.gui

    def load_gui(self, gui_no, show_gui_ele=False):
        self.device.gui_no = gui_no
        self.device.reset_file_path_by_gui_no(gui_no=gui_no)
        self.load_gui_data(gui_no, show_gui_ele)
        return self.gui

    '''
    **********************
    *** Assets Loading ***
    **********************
    '''
    def init_device(self):
        from ppadb.client import Client as AdbClient
        from Device import Device
        client = AdbClient(host="127.0.0.1", port=5037)
        if len(client.devices()) > 0:
            print('Load Device')
            self.device = Device(client.devices()[0], app_name=self.app_name, test_case_no=self.test_case_no, output_file_root=self.output_file_root)
        else:
            print('No Connected Device')

    def load_gui_detection_models(self):
        if 'classification' in self.gui_detection_models and self.gui_detection_models['classification'] is not None:
            return
        from utils.classification.IconClassifier import IconClassifier
        from utils.classification.IconCaption import IconCaption
        # print('- Load GUI Detection Model -')
        self.gui_detection_models['classification'] = IconClassifier(model_path='./utils/classification/model_results/best-0.93.pt', class_path='./utils/classification/model_results/iconModel_labels.json')
        self.gui_detection_models['caption'] = IconCaption(vocab_path='./utils/classification/model_results/vocab_idx2word.json',  model_path='./utils/classification/model_results/labeldroid.pt')

    '''
    ********************
    *** Get GUI Info ***
    ********************
    '''
    def load_gui_data(self, gui_no, show=False):
        '''
        Load analyzed result gui data
        '''
        # print('\n=== Load and analyzed UI info ===')
        print('--- Load and analyzed UI info ---')
        self.device.reset_file_path_by_gui_no(gui_no)
        gui = GUI(gui_img_file=self.device.output_file_path_screenshot,
                  gui_json_file=self.device.output_file_path_json,
                  output_file_root=self.device.testcase_save_dir)
        gui.load_elements()
        if show:
            gui.show_all_elements(only_leaves=False)
        self.gui = gui

    def collect_gui(self):
        '''
        collect the raw GUI data [raw xml, VH Json, Screenshot] on current screen and save to 'data/app_name/test_case_no/device'
        => ui_no.xml, ui_no.json, ui_no.png
        '''
        print('=== Collect UI metadata from the device ===')
        self.device.cap_screenshot()
        self.device.cap_vh()
        self.device.reformat_vh_json()

    def analyze_gui(self, gui_detection=True, show=False):
        '''
        Clean up the VH tree and extract [elements, element_tree] and save to "data/app_name/test_case_no/guidata"
        :param gui_detection: True to enable icon classification and captioning
        => ui_no_elements.json, ui_no_tree.json
        '''
        print('=== Extract and analyze UI info ===')
        if gui_detection:
            self.load_gui_detection_models()
        gui = GUI(gui_img_file=self.device.output_file_path_screenshot,
                  gui_json_file=self.device.output_file_path_json,
                  output_file_root=self.device.testcase_save_dir,
                  model_icon_caption=self.gui_detection_models['caption'],
                  model_icon_classification=self.gui_detection_models['classification'])
        gui.ui_info_extraction()
        gui.ui_analysis_elements_description(ocr=gui_detection, caption=gui_detection, cls=gui_detection)
        gui.ui_element_tree()
        if show:
            gui.show_all_elements(only_leaves=False)
        self.gui = gui

    def add_system_info_in_tree(self):
        self.gui.element_tree['system_info'] = self.device.get_package_and_activity_name()
        self.gui.element_tree['system_info']['keyboard_active'] = self.device.check_keyboard_active()

    '''
    **********************
    *** Execute Action ***
    **********************
    '''
    def execute_action(self, action, gui, show=False, waiting_time=2):
        '''
        @action: {"Action": "Click", "Element": 14, "Reason":}
            => 'click', 'scroll'
        @device: ppadb device
        '''
        print('--- Execute the action on the GUI ---')
        ele = gui.elements[action['Element']]
        bounds = ele['bounds']
        if action['Action'].lower() == 'click':
            centroid = ((bounds[2] + bounds[0]) // 2, (bounds[3] + bounds[1]) // 2)
            if show:
                board = gui.img.copy()
                cv2.circle(board, (centroid[0], centroid[1]), 20, (255, 0, 255), 8)
                cv2.imshow('click', cv2.resize(board, (board.shape[1] // 3, board.shape[0] // 3)))
                cv2.waitKey()
                cv2.destroyWindow('click')
            self.device.adb_device.input_tap(centroid[0], centroid[1])
        elif action['Action'].lower() == 'scroll':
            bias = 5
            if show:
                board = gui.img.copy()
                cv2.circle(board, (bounds[2]-bias, bounds[3]+bias), 20, (255, 0, 255), 8)
                cv2.circle(board, (bounds[0]-bias, bounds[1]+bias), 20, (255, 0, 255), 8)
                cv2.imshow('scroll', cv2.resize(board, (board.shape[1] // 3, board.shape[0] // 3)))
                cv2.waitKey()
                cv2.destroyWindow('scroll')
            self.device.adb_device.input_swipe(bounds[2]-bias, bounds[3]+bias, bounds[0]-bias, bounds[1]+bias, 500)
        elif action['Action'].lower() == 'input':
            print(action)
            self.device.adb_device.input_text(action['Input Text'])
        # wait a few second to be refreshed
        time.sleep(waiting_time)

    def device_go_back_button(self, waiting_time=2):
        self.device.adb_device.shell('input keyevent KEYCODE_BACK')
        # wait a few second to be refreshed
        time.sleep(waiting_time)

    def device_launch_app(self, package_name, waiting_time=2):
        self.device.adb_device.shell(f'monkey -p {package_name} -c android.intent.category.LAUNCHER 1')
        time.sleep(waiting_time)


if __name__ == '__main__':
    gui_manager = GUIDataManager('twitter', 1, output_file_root='../data')
    gui_manager.get_current_gui_from_device(gui_detection=True, show_gui_ele=True)
