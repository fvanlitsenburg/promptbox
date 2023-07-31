import os
import sys
import logging
from pathlib import Path
from json import JSONDecodeError

import pandas as pd
import streamlit as st
from markdown import markdown
from utils.utils import haystack_is_ready, upload_doc, haystack_version, load_models, fetch_docs, query_listed_documents, check_sentiment

# Adjust to questions for demo:
DEFAULT_QUESTION_AT_STARTUP_P1 = os.getenv("DEFAULT_QUESTION_AT_STARTUP_P1", "What are HSBC’s restrictive policies on the Oil and Gas sectors?")
DEFAULT_QUESTION_AT_STARTUP_P2 = os.getenv("DEFAULT_QUESTION_AT_STARTUP_P2", "What kind of new financing will be prohibited by HSBC based on this policy?")
DEFAULT_QUESTION_AT_STARTUP_P3 = os.getenv("DEFAULT_QUESTION_AT_STARTUP_P3", "What kind of new clients will be restricted by HSBC based on this policy?")
DEFAULT_QUESTION_AT_STARTUP_P4 = os.getenv("DEFAULT_QUESTION_AT_STARTUP_P4", "What kind of transaction and/or client will be subject to enhanced due-diligence based on this policy?")
DEFAULT_QUESTION_AT_STARTUP_P5 = os.getenv("DEFAULT_QUESTION_AT_STARTUP_P5", "  How will Client Transition Plan affect HSBC’s decision on financing Oil and Gas clients?")


# Standard prompt:
DEFAULT_PROMPT_AT_STARTUP_P1 = os.getenv("DEFAULT_PROMPT_AT_STARTUP_P1","""Synthesize a comprehensive answer from the following given question and relevant paragraphs.
Provide a clear and concise response that summarizes the key points and information presented in the paragraphs.
Your answer should be in your own words and be no longer than necessary.
\n\n Question: {query}
\n\n Paragraphs: {join(documents)}  \n\n Answer:""")


# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def main():

    st.set_page_config(page_title="Promptbox", layout="wide",page_icon="ui/pages/promptbox_logo.png",)

    # Persistent state
    set_state_if_absent("question_p1", DEFAULT_QUESTION_AT_STARTUP_P1)
    set_state_if_absent("question_p2", DEFAULT_QUESTION_AT_STARTUP_P2)
    set_state_if_absent("question_p3", DEFAULT_QUESTION_AT_STARTUP_P3)
    set_state_if_absent("question_p4", DEFAULT_QUESTION_AT_STARTUP_P4)
    set_state_if_absent("question_p5", DEFAULT_QUESTION_AT_STARTUP_P5)
    set_state_if_absent("prompt_p1", DEFAULT_PROMPT_AT_STARTUP_P1)

    set_state_if_absent("results", None)
    set_state_if_absent("raw_json", None)


    # Small callback to reset the interface in case the text of the question changes
    def reset_results(*args):
        st.session_state.answer = None
        st.session_state.results = None
        st.session_state.raw_json = None

    # Title
    st.image("ui/pages/promptbox_banner.png")
    st.write("---")

    # Sidebar
    st.sidebar.header("Options")
    st.sidebar.write("Available models: \n ")
    for i in st.session_state.modellist:
        st.sidebar.markdown("-- " + i)


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
                with st.expander("See and edit question prompt template"):
                        prompt_p1 = st.text_area("", value=st.session_state.prompt_p1, max_chars=1000, on_change=reset_results,key=2)

            question_p2 = st.text_input("", value=st.session_state.question_p2, max_chars=100, on_change=reset_results,key=3)
            question_p3 = st.text_input("", value=st.session_state.question_p3, max_chars=100, on_change=reset_results,key=4)
            question_p4 = st.text_input("", value=st.session_state.question_p4, max_chars=100, on_change=reset_results,key=5)
            question_p5 = st.text_input("", value=st.session_state.question_p5, max_chars=100, on_change=reset_results,key=6)

            col1, = st.columns(1)
            col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)


            part2.header("Output & Documents")
            part2.write("---")

            # Run button
            run_pressed = col1.button("Run")

            run_query = (
                run_pressed or question_p1 != st.session_state.question_p1
            )

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
                    modellist = st.session_state.modellist
                    output = {}
                    detailed_output = {}

                    # Load each model and answer the queries for it
                    for model in modellist:


                        # Check if the requested model(s) has already been loaded
                        if model in modelbase.keys():
                            print("No loading needed")
                        else:
                            print("Loading next model")
                            modelbase = load_models([model])

                        output[model] = {}
                        detailed_output[model] = {}

                        # Run each model
                        for question in [question_p1,question_p2,question_p3,question_p4,question_p5]:
                            response = query_listed_documents(question,[document],modelbase[model],prompt_p1)
                            output[model][question] = response[0][0]['Answer']
                            detailed_output[model][question] = response[1]

                    with part2:
                        st.table(output)

                    with tab3:
                        expander = st.expander("See detailed prompt info for question 1")
                        expander.write(detailed_output)


                except Exception as e:
                    print(e)

main()
