import streamlit as st
from utils import haystack_is_ready, query, send_feedback, upload_doc, haystack_version, get_backlink, load_models, fetch_docs, query_listed_documents, check_sentiment

model_path = "../../../hf/"


st.set_page_config(
    page_title="Promptbox",
    page_icon="pages/promptbox_logo.png",
)

modelbase = load_models()

st.session_state['models'] = modelbase
#st.session_state['models'] = "flan-t5-base"

st.image("pages/promptbox_banner.png")

st.sidebar.success("Select a use case above.")

st.markdown(
    """
    Promptbox brings Generative AI in a more intuitive, workflow-manner to your organisation. \n
    **👈 Select a use case from the sidebar** to see some examples

"""
)
