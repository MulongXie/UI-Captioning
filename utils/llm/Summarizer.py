class Summarizer:
    def __init__(self, engine, conversation=None):
        self.engine = engine    # Openai llm engine, based on gpt-3.5-turbo or gpt-4
        self.conversation = conversation
        self.gui_summary = None

    def reset_conversation(self):
        self.conversation = [
            {'role': 'system', 'content': self.engine.role},
            {'role': 'user', 'content': 'You will be given some view hierarchies of UIs containing various UI blocks and elements.'}
        ]

    def summarize_gui(self, gui, factor='functionality', word_limit=50, printlog=False):
        self.conversation.append(
            {'role': 'user', 'content': 'Please summarize this UI in terms of ' + factor + ' within ' + str(word_limit) + ' words.' + str(gui.element_tree)}
        )
        self.conversation.append(self.engine.ask_openai_conversation(self.conversation, printlog))
        self.gui_summary = self.conversation[-1]['content']
        return self.gui_summary

    def wrap_previous_annotations_as_examples(self, annotations):
        self.reset_conversation()
        # append previous annotations as examples
        if len(annotations) > 0:
            self.conversation.append(
                {'role': 'user', 'content': 'Here are some examples with appropriate summarization.'},
            )
            for i, ann in enumerate(annotations):
                if ann['revised']:
                    # prev_ann = 'The summarization is not perfectly correct, here is the revision by human and the reasons for revision.' \
                    #            'Learn from them for your future summarization generation.\n' \
                    #            '#Revised Ground Truth Summarization:\n' + annotations['annotation'] + '.\n' \
                    #            '#Reasons for Revision:\n' + annotations['revision-suggestion'] + '.\n'
                    self.conversation += [
                        {'role': 'user', 'content': 'Example ' + str(i) + ': #GUI:' + str(ann['element-tree'])},
                        {'role': 'user', 'content': '#Ground Truth Summarization: ' + ann['annotation']},
                        {'role': 'user', 'content': '#Revision Suggestions: ' + ann['revision-suggestion']},
                    ]

