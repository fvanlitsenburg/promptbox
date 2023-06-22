import os
import sys
import logging
from pathlib import Path
from json import JSONDecodeError

import pandas as pd
import streamlit as st
from markdown import markdown
from ui.utils import haystack_is_ready, query, send_feedback, upload_doc, haystack_version, get_backlink, load_models, fetch_docs, query_each_document

model_path = "../../../hf/"

# Adjust to a question that you would like users to see in the search bar when they load the UI:
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "Does the document concern misuse of client money?")
DEFAULT_ANSWER_AT_STARTUP = os.getenv("DEFAULT_ANSWER_AT_STARTUP", "")
DEFAULT_PROMPT_AT_STARTUP = os.getenv("DEFAULT_PROMPT_AT_STARTUP","""Synthesize a comprehensive answer from the following given question and relevant paragraphs.
Provide a clear and concise response that summarizes the key points and information presented in the paragraphs.
Your answer should be in your own words and be no longer than necessary.
\n\n Question: {query}
\n\n Paragraphs: {join(documents)}  \n\n Answer:""")
DEFAULT_MODEL_AT_STARTUP = os.getenv("DEFAULT_MODEL_AT_STARTUP","flan-t5-base")



# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", "3"))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", "3"))

# Labels for the evaluation
EVAL_LABELS = os.getenv("EVAL_FILE", str(Path(__file__).parent / "eval_labels_example.csv"))

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def main():

    st.set_page_config(page_title="Promptmaster", page_icon="https://haystack.deepset.ai/img/HaystackIcon.png")

    # Persistent state
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent("prompt", DEFAULT_PROMPT_AT_STARTUP)
    set_state_if_absent("answer", DEFAULT_ANSWER_AT_STARTUP)
    set_state_if_absent("model",DEFAULT_MODEL_AT_STARTUP)
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)

    #load_models(DEFAULT_MODEL_AT_STARTUP,DEFAULT_PROMPT_AT_STARTUP)
    modelbase = load_models()
    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Title
    st.write("# PrompMaster")
    st.markdown(
        """
This demo allows you to upload documents and query them using Generative AI.

The pipeline goes through each document and answers the query for each document. It embeds the query in a 'prompt template' that gives the LLM further instructions. You can expand the textbox below that defines the prompt template.

In the sidebar you can select which LLM to use.

*Note: do not use keywords, but full-fledged questions.* The demo is not optimized to deal with keyword queries and might misunderstand you.
""",
        unsafe_allow_html=True,
    )

    # Sidebar
    st.sidebar.header("Options")
    model = st.sidebar.selectbox('What LLM would you like to use?',
    ('flan-t5-base','fastchat-t5-3b-v1.0'))
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=10,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        on_change=reset_results,
    )

    eval_mode = st.sidebar.checkbox("Evaluation mode")
    debug = st.sidebar.checkbox("Show debug info")

    # File upload block
    if not DISABLE_FILE_UPLOAD:
        st.sidebar.write("## File Upload:")
        data_files = st.sidebar.file_uploader("", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        for data_file in data_files:
            # Upload file
            if data_file:
                raw_json = upload_doc(data_file)
                st.sidebar.write(str(data_file.name) + " &nbsp;&nbsp; ✅ ")
                if debug:
                    st.subheader("REST API JSON response")
                    st.sidebar.write(raw_json)

    hs_version = ""
    try:
        hs_version = f" <small>(v{haystack_version()})</small>"
    except Exception:
        pass

    st.sidebar.markdown(
        f"""
    <style>
        a {{
            text-decoration: none;
        }}
        .haystack-footer {{
            text-align: center;
        }}
        .haystack-footer h4 {{
            margin: 0.1rem;
            padding:0;
        }}
        footer {{
            opacity: 0;
        }}
    </style>

    """,
        unsafe_allow_html=True,
    )

    # Search bar
    question = st.text_input("", value=st.session_state.question, max_chars=100, on_change=reset_results)

    with st.expander("See and edit prompt template"):
            prompt = st.text_area("", value=st.session_state.prompt, max_chars=1000, on_change=reset_results)

    col1, = st.columns(1)
    col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)




    # Run button
    run_pressed = col1.button("Run")

    run_query = (
        run_pressed or question != st.session_state.question
    ) and not st.session_state.random_question_requested

    # Check the connection
    with st.spinner("⌛️ &nbsp;&nbsp; Haystack is starting..."):
        if not haystack_is_ready():
            st.error("🚫 &nbsp;&nbsp; Connection Error. Is Haystack running?")
            run_query = False
            reset_results()

    if not run_pressed:

        st.table(dict.fromkeys(fetch_docs(),"No input, ask a question!"))

    elif run_query and question:
        reset_results()
        st.session_state.question = question

        try:
            print("Running "+model)
            print(modelbase[model])
            print(type(modelbase[model]))
            output = query_each_document(question,modelbase[model],prompt)
            print("Table")
            st.table(output[0])
            print("oOutput:")
            expander = st.expander("See detailed prompt info")
            expander.write(output[1])
            print(output[0])
            #print(output[1])
        except Exception as e:
            print(e)

main()
