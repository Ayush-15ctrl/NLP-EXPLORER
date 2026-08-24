import gradio as gr
from modules.tokenization import (
    spacy_tokenization,
    modern_tokenization,
)

from modules.preprocessing import (
    remove_stopwords,
    lemmatize

)

from modules.bow import (
    bag_of_words
)



from modules.ner import (
    named_entity_recognition
)

DEFAULT_TEXT = """

Barack Obama visited New York in 2020.

He gave an interesting speech about technology.

"""
with gr.Blocks(
    title="NLP Explorer"
) as app:
    gr.Markdown(
        """
        # 🧠 NLP Explorer
        ### Natural Language Processing Laboratory
        Enter a paragraph below and explore different NLP techniques using the tabs.
        """
    )

    text_input = gr.Textbox(
        label="Enter Your Text",
        placeholder="Enter a paragraph here...",
        value=DEFAULT_TEXT,
        lines=8
    )

    gr.Markdown(
        """
        **Enter the text above and then select any experiment tab.**
        """
    )

    with gr.Tabs():
        with gr.Tab("🔤 Tokenization"):
            gr.Markdown(
                """
                ## spaCy Tokenization
                Tokenization divides text into individual tokens such as words, numbers and punctuation.
                """
            )

            tokenize_button = gr.Button(
                "Perform Tokenization",
                variant="primary"
            )

            token_output = gr.Dataframe(
                headers=[
                    "Token"
                ],
                label="spaCy Tokens"
            )

            tokenize_button.click(
                fn=spacy_tokenization,
                inputs=text_input,
                outputs=token_output
            )

        with gr.Tab("🤖 Modern Tokenizers"):
            gr.Markdown(
                """
                ## Modern NLP Tokenization
                Compare tokenization performed by popular Transformer models.
                """
            )

            model_dropdown = gr.Dropdown(
                choices=[
                    "BERT",
                    "GPT-2",
                    "T5",
                    "DeepSeek"
                ],
                value="BERT",
                label="Select Tokenizer"
            )

            modern_button = gr.Button(
                "Perform Modern Tokenization",
                variant="primary"
            )

            modern_output = gr.Dataframe(
                headers=[
                    "Token Number",
                    "Token"
                ],
                label="Model Tokens"
            )

            modern_button.click(
                fn=modern_tokenization,
                inputs=[
                    text_input,
                    model_dropdown
                ],
                outputs=modern_output
            )

        with gr.Tab("🚫 Stop Words"):
            gr.Markdown(
                """
                ## Stop Word Removal
                Stopwords are common words that spaCy identifies as having relatively little information for many NLP tasks.
                Example: **the, is, a, an, and**
                """
            )

            stop_button = gr.Button(
                "Remove Stop Words",
                variant="primary"
            )

            cleaned_output = gr.Textbox(
                label="Text After Removing Stop Words",
                lines=6
            )

            removed_output = gr.Textbox(
                label="Removed Stop Words",
                lines=3
            )

            stop_button.click(
                fn=remove_stopwords,
                inputs=text_input,
                outputs=[
                    cleaned_output,
                    removed_output
                ]
            )

        with gr.Tab("🌱 Lemmatization"):
            gr.Markdown(
                """
                ## Lemmatization
                Lemmatization converts words into their base or dictionary form.
                Example:
                **running → run**
                """
            )

            lemma_button = gr.Button(
                "Perform Lemmatization",
                variant="primary"
            )

            lemma_output = gr.Dataframe(
                headers=[
                    "Original Word",
                    "Lemma",
                    "POS"
                ],
                label="Lemmatization Result"
            )

            lemma_button.click(
                fn=lemmatize,
                inputs=text_input,
                outputs=lemma_output
            )

        with gr.Tab("📊 Bag of Words"):
            gr.Markdown(
                """
                ## Bag of Words
                Bag of Words represents text based on the frequency of words.
                """
            )
            
            bow_button = gr.Button(
                "Create Bag of Words",
                variant="primary"
            )

            bow_output = gr.Dataframe(
                headers=[
                    "Word",
                    "Frequency"
                ],
                label="Word Frequency"
            )

            vocabulary_output = gr.Textbox(
                label="Vocabulary"
            )

            bow_button.click(
                fn=bag_of_words,
                inputs=text_input,
                outputs=[
                    bow_output,
                    vocabulary_output
                ]
            )

        with gr.Tab("🏷️ Named Entity Recognition"):
            gr.Markdown(
                """
                ## Named Entity Recognition (NER)
                NER identifies named entities such as:
                - PERSON
                - ORG
                - GPE
                - DATE
                - MONEY
                - LOC
                """
            )

            ner_button = gr.Button(
                "Find Named Entities",
                variant="primary"
            )

            ner_output = gr.Dataframe(
                headers=[
                    "Entity",
                    "Label",
                    "Description"
                ],
                label="Named Entities"
            )
            ner_button.click(
                fn=named_entity_recognition,
                inputs=text_input,
                outputs=ner_output
            )

app.launch()