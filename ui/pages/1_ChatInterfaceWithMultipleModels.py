import os
import sys
import logging
from pathlib import Path
from json import JSONDecodeError

import pandas as pd
import streamlit as st
from markdown import markdown
from utils import haystack_is_ready, query, send_feedback, upload_doc, haystack_version, get_backlink, load_models, fetch_docs, query_listed_documents, check_sentiment

model_path = "../../../hf/"

# Adjust to a question that you would like users to see in the search bar when they load the UI:
# Questions
DEFAULT_QUESTION_AT_STARTUP_P1 = os.getenv("DEFAULT_QUESTION_AT_STARTUP_P1", "What are HSBC’s restrictive policies on the Oil and Gas sectors?")


# Prompts
DEFAULT_PROMPT_AT_STARTUP_P1 = os.getenv("DEFAULT_PROMPT_AT_STARTUP_P1","""Synthesize a comprehensive answer from the following given question and relevant paragraphs.
Provide a clear and concise response that summarizes the key points and information presented in the paragraphs.
Your answer should be in your own words and be no longer than necessary.
\n\n Question: {query}
\n\n Paragraphs: {join(documents)}  \n\n Answer:""")


# Sliders
DEFAULT_DOCS_FROM_RETRIEVER = int(os.getenv("DEFAULT_DOCS_FROM_RETRIEVER", "3"))
DEFAULT_NUMBER_OF_ANSWERS = int(os.getenv("DEFAULT_NUMBER_OF_ANSWERS", "3"))

# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def main():

    st.set_page_config(page_title="Promptbox", layout="wide")

    # Persistent state
    set_state_if_absent("question_p1", DEFAULT_QUESTION_AT_STARTUP_P1)
    set_state_if_absent("prompt_p1", DEFAULT_PROMPT_AT_STARTUP_P1)
    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)
    set_state_if_absent("random_question_requested", False)

    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Title
    st.write("# PromptBox")
    st.write("---")

    # Sidebar
    st.sidebar.header("Options")
    st.sidebar.write("Available models: \n ")
    for i in st.session_state.models:
        st.sidebar.markdown("-- " + i)
    top_k_retriever = st.sidebar.slider(
        "Max. number of documents from retriever",
        min_value=1,
        max_value=10,
        value=DEFAULT_DOCS_FROM_RETRIEVER,
        step=1,
        on_change=reset_results,
    )

    # File upload block
    if not DISABLE_FILE_UPLOAD:
        st.sidebar.write("## File Upload:")
        data_files = st.sidebar.file_uploader("", type=["pdf", "txt", "docx"], accept_multiple_files=True)
        for data_file in data_files:
            # Upload file
            if data_file:
                raw_json = upload_doc(data_file)
                st.sidebar.write(str(data_file.name) + " &nbsp;&nbsp; ✅ ")


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

    part1,part2 = st.columns(2)

    with part1:
        st.header("Input")
        tab1,tab2,tab3 = st.tabs(["Front","Detailed instructions","Detailed output"])

        with tab1:
            st.markdown(
                """
        This demo allows you to upload documents and query them using Generative AI.

        On this page, we will render output from the same question, for the same document, using different models so you can compare.

        """,
                unsafe_allow_html=True,
            )




            st.subheader("Select a document")

            document = st.selectbox('',
            (fetch_docs()))
            st.subheader("Write your question")

            # Search bar
            question_p1 = st.text_input("", value=st.session_state.question_p1, max_chars=100, on_change=reset_results,key=1)

            with tab2:
                with st.expander("See and edit first question prompt template"):
                        prompt_p1 = st.text_area("", value=st.session_state.prompt_p1, max_chars=1000, on_change=reset_results,key=2)

            col1, = st.columns(1)
            col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)



            col1, = st.columns(1)
            col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)


            part2.header("Output & Documents")
            part2.write("---")

            # Run button
            run_pressed = col1.button("Run")

            run_query = (
                run_pressed or question_p1 != st.session_state.question_p1
            ) and not st.session_state.random_question_requested

            # Check the connection
            with st.spinner("⌛️ &nbsp;&nbsp; Haystack is starting..."):
                if not haystack_is_ready():
                    st.error("🚫 &nbsp;&nbsp; Connection Error. Is Haystack running?")
                    run_query = False
                    reset_results()

            if not run_pressed:

                with part2:
                    st.table(dict.fromkeys(st.session_state.models,"No input, ask a question!"))

            elif run_query and question_p1:
                reset_results()
                st.session_state.question_p1 = question_p1

                try:
                    modelbase = st.session_state.models
                    output = {}
                    detailed_output = {}
                    for model in modelbase:
                        print("Running "+model)
                        print(model)
                        print(modelbase[model])
                        print(type(modelbase[model]))
                        print(document)
                        response = query_listed_documents(question_p1,[document],modelbase[model],prompt_p1)
                        output[model] = response[0][0]['Answer']
                        detailed_output[model] = response[1]
                    print("Table")
                    with part2:
                        st.table(output)

                    with tab3:
                        expander = st.expander("See detailed prompt info for question 1")
                        expander.write(response[1])


                except Exception as e:
                    print(e)

main()
