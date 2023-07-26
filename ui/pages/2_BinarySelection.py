import os
import sys
import logging
from pathlib import Path
from json import JSONDecodeError

import pandas as pd
import streamlit as st
from markdown import markdown
from utils import haystack_is_ready, upload_doc, haystack_version, load_models, fetch_docs, query_listed_documents, check_sentiment

model_path = "../../../hf/"

# Adjust to a question that you would like users to see in the search bar when they load the UI:
# Questions
DEFAULT_QUESTION_AT_STARTUP = os.getenv("DEFAULT_QUESTION_AT_STARTUP", "Was the applicant deemed 'fit and proper' or not?")
DEFAULT_QUESTION2_AT_STARTUP = os.getenv("DEFAULT_QUESTION2_AT_STARTUP", "Was the application 'fit and proper'?")
DEFAULT_QUESTION3_AT_STARTUP = os.getenv("DEFAULT_QUESTION3_AT_STARTUP", "What was the applicant's defense?")

# Prompts
DEFAULT_PROMPT_AT_STARTUP = os.getenv("DEFAULT_PROMPT_AT_STARTUP","""Synthesize a comprehensive answer from the following given question and relevant paragraphs.
Provide a clear and concise response that summarizes the key points and information presented in the paragraphs.
Your answer should be in your own words and be no longer than necessary.
\n\n Question: {query}
\n\n Paragraphs: {join(documents)}  \n\n Answer:""")
DEFAULT_PROMPT2_AT_STARTUP = os.getenv("DEFAULT_PROMPT2_AT_STARTUP","""Answer the following question based on the provided paragraph. Answer it with 'yes' or 'no', then provide your reasoning. Answer 'na' if the answer is unclear. For example: 'yes'. Because...
 Question: {query}
 Paragraph: {join(documents)}
 Answer:""")
DEFAULT_PROMPT3_AT_STARTUP = os.getenv("DEFAULT_PROMPT3_AT_STARTUP","""Synthesize a comprehensive answer from the following given question and relevant paragraphs.
Provide a clear and concise response that summarizes the key points and information presented in the paragraphs.
Your answer should be in your own words and be no longer than necessary.
\n\n Question: {query}
\n\n Paragraphs: {join(documents)}  \n\n Answer:""")

# Models
DEFAULT_MODEL_AT_STARTUP = os.getenv("DEFAULT_MODEL_AT_STARTUP","flan-t5-base")
DEFAULT_MODEL2_AT_STARTUP = os.getenv("DEFAULT_MODEL2_AT_STARTUP","flan-t5-base")
DEFAULT_MODEL3_AT_STARTUP = os.getenv("DEFAULT_MODEL3_AT_STARTUP","flan-t5-base")



# Whether the file upload should be enabled or not
DISABLE_FILE_UPLOAD = bool(os.getenv("DISABLE_FILE_UPLOAD"))


def set_state_if_absent(key, value):
    if key not in st.session_state:
        st.session_state[key] = value


def main():

    st.set_page_config(page_title="Promptbox", layout="wide",page_icon="ui/pages/promptbox_logo.png",)

    # Persistent state
    set_state_if_absent("question", DEFAULT_QUESTION_AT_STARTUP)
    set_state_if_absent("question2", DEFAULT_QUESTION2_AT_STARTUP)
    set_state_if_absent("question3", DEFAULT_QUESTION3_AT_STARTUP)
    set_state_if_absent("prompt", DEFAULT_PROMPT_AT_STARTUP)
    set_state_if_absent("prompt2", DEFAULT_PROMPT2_AT_STARTUP)
    set_state_if_absent("prompt3", DEFAULT_PROMPT3_AT_STARTUP)
    set_state_if_absent("model",DEFAULT_MODEL_AT_STARTUP)
    set_state_if_absent("model2",DEFAULT_MODEL2_AT_STARTUP)
    set_state_if_absent("model3",DEFAULT_MODEL3_AT_STARTUP)
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
    st.image("ui/pages/promptbox_banner.png")
    st.write("---")

    # Sidebar
    st.sidebar.header("Options")
    model = st.sidebar.selectbox('What would you like to use for the first LLM?',
    ('flan-t5-base','fastchat-t5-3b-v1.0'))
    model2 = st.sidebar.selectbox('What would you like to use for the second LLM?',
    ('flan-t5-base','fastchat-t5-3b-v1.0'))
    model3 = st.sidebar.selectbox('What would you like to use for the third LLM?',
    ('flan-t5-base','fastchat-t5-3b-v1.0'))


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

        Several LLM agents are tied together. Your second question will only be applied to documents where the first question came back with 'yes'.

        Using the tab "detailed instructions" you can edit the prompt templates used.

        """,
                unsafe_allow_html=True,
            )





            st.subheader("First question - binary for each document")

            # Search bar
            question = st.text_input("", value=st.session_state.question, max_chars=100, on_change=reset_results,key=1)

            with tab2:
                with st.expander("See and edit first question prompt template"):
                        prompt = st.text_area("", value=st.session_state.prompt, max_chars=1000, on_change=reset_results,key=2)

            col1, = st.columns(1)
            col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)

            st.subheader("Second question - checking sentiment for documents")

            # Search bar
            question2 = st.text_input("", value=st.session_state.question2, max_chars=100, on_change=reset_results,key=3)

            st.subheader("Third question - checking content of relevant documents")

            # Search bar
            question3 = st.text_input("", value=st.session_state.question3, max_chars=100, on_change=reset_results,key=4)

            with tab2:
                with st.expander("See and edit second question prompt template"):
                        prompt2 = st.text_area("", value=st.session_state.prompt2, max_chars=1000, on_change=reset_results,key=5)

            with tab2:
                with st.expander("See and edit third question prompt template"):
                        prompt3 = st.text_area("", value=st.session_state.prompt3, max_chars=1000, on_change=reset_results,key=6)

            col1, = st.columns(1)
            col1.markdown("<style>.stButton button {width:100%;}</style>", unsafe_allow_html=True)


            part2.header("Output & Documents")
            part2.write("---")

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

                with part2:
                    st.table(dict.fromkeys(fetch_docs(),"No input, ask a question!"))

            elif run_query and question:
                reset_results()
                st.session_state.question = question

                try:
                    # Running first model
                    if model in modelbase.keys():
                            print("No loading needed")
                    else:
                            print("Loading next model")
                            modelbase = load_models([model])
                    print("Running "+model)
                    print(modelbase[model])
                    
                    output = query_listed_documents(question,fetch_docs(),modelbase[model],prompt)
                    
                    with part2:
                        st.table(output[0])
                    print("Check sentiment")

                    # Running second model

                    if model2 in modelbase.keys():
                            print("No loading needed")
                    else:
                            print("Loading next model")
                            modelbase = load_models([model2])

                    output2 = check_sentiment(question2,modelbase[model2],output[0],prompt2)
                    with part2:
                        st.write(question2)
                        st.table(output2[0])
                    print("Output:")
                    with tab3:
                        expander = st.expander("See detailed prompt info for question 1")
                        expander.write(output[1])
                        expander = st.expander("See detailed prompt info for question 2")
                        expander.write(output2[1])

                    # Running third model

                    if model3 in modelbase.keys():
                            print("No loading needed")
                    else:
                            print("Loading next model")
                            modelbase = load_models([model3])

                    output3 = query_listed_documents(question3,output2[2],modelbase[model3],prompt3)
                    with part2:
                        st.write(question3)
                        st.table(output3[0])
                except Exception as e:
                    print(e)

main()
