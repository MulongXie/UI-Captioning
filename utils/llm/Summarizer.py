class Summarizer:
    def __init__(self, engine, conversation=None):
        self.engine = engine    # Openai llm engine, based on gpt-3.5-turbo or gpt-4
        self.conversation = conversation
        self.gui_summary = None

    def reset_conversation_with_gui(self, gui):
        self.conversation = [
            {'role': 'system', 'content': self.engine.role},
            {'role': 'user', 'content': 'This is a view hierarchy of a UI containing various UI blocks and elements.'},
            {'role': 'user', 'content': str(gui.element_tree)},
        ]

    def summarize_gui(self, gui, printlog=False):
        self.reset_conversation_with_gui(gui)
        self.conversation.append(
            {'role': 'user', 'content': 'Please summarize this UI.'}
        )
        self.conversation.append(self.engine.ask_openai_conversation(self.conversation, printlog))
        self.gui_summary = self.conversation[-1]['content']
        return self.gui_summary

