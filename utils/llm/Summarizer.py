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
            # {'role': 'user', 'content': 'Summarize this UI in terms of ' + factor + ' within ' + str(word_limit) + ' words.' + str(gui.element_tree)}
            {'role': 'user', 'content': 'Summarize this UI in terms of ' + factor + '.' + str(gui.element_tree)}
        )
        self.conversation.append(self.engine.ask_openai_conversation(self.conversation, printlog))
        self.gui_summary = self.conversation[-1]['content']
        return self.gui_summary

    def wrap_previous_annotations_as_examples(self, annotations):
        self.reset_conversation()
        # append previous annotations as examples
        if len(annotations) > 0:
            self.conversation.append(
                {'role': 'user', 'content': 'Here are some examples with appropriate UI summarization.'},
            )
            for i, ann in enumerate(annotations):
                # prev_ann = 'The summarization is not perfectly correct, here is the revision by human and the reasons for revision.' \
                #            'Learn from them for your future summarization generation.\n' \
                #            '#Revised Ground Truth Summarization:\n' + annotations['annotation'] + '.\n' \
                #            '#Reasons for Revision:\n' + annotations['revision-suggestion'] + '.\n'
                self.conversation += [
                    # {'role': 'user', 'content': 'Example ' + str(i) + ': #GUI:' + str(ann['element-tree'])},
                    {'role': 'user', 'content': '#Example Summarization: ' + ann['annotation'] + '.\n #Summarization Rules: ' + ann['revision-suggestion']},
                ]
            self.conversation.append(
                {'role': 'user', 'content': 'Learn from the #Example Summarization and exactly follow the #Summarization Rules in future summarization.'},
            )

    def summarize_gui_with_revise_suggestion(self, gui, factor, annotation, printlog=False):
        print('\n==============================')
        print('\n*** Summarization [' + factor + '] ***')
        # self.reset_conversation()
        if len(annotation['revision-suggestion-history']) == 0:
            self.conversation.append(
                {'role': 'user', 'content': 'Summarize this UI in terms of ' + factor + '. UI: ' + str(gui.element_tree)}
            )
        else:
            self.conversation.append(
                {'role': 'user', 'content': 'Your summarization does not perfectly meet our expectation, consider revise it with the following revision suggestions: ' + annotation['revision-suggestion-history'][-1]}
            )
        self.conversation.append(self.engine.ask_openai_conversation(self.conversation, printlog))
        self.gui_summary = self.conversation[-1]['content'].replace('\n\n', '\n')
        return self.gui_summary
