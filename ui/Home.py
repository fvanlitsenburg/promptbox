import streamlit as st
from utils import haystack_is_ready, upload_doc, haystack_version, load_models, fetch_docs, query_listed_documents, check_sentiment
import os


st.set_page_config(
    page_title="Promptbox",
    page_icon="ui/pages/promptbox_logo.png",
)




st.session_state['modellist'] = ['flan-t5-base','fastchat-t5-3b-v1.0']


modelbase = load_models(st.session_state.modellist) # Comment this line to run with only one model in cache:

# Uncomment this line to run with only one model in cache:
# modelbase = load_models(['flan-t5-base'])

st.session_state['models'] = modelbase
print(st.session_state.modellist)
print(modelbase)

st.image("ui/pages/promptbox_banner.png")

st.sidebar.success("Select a use case above.")

st.markdown(
    """
    Promptbox brings Generative AI in a more intuitive, workflow-manner to your organisation. \n
    **👈 Select a use case from the sidebar** to see some examples

"""
)
