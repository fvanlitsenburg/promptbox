import streamlit as st
from utils import haystack_is_ready, query, send_feedback, upload_doc, haystack_version, get_backlink, load_models, fetch_docs, query_listed_documents, check_sentiment

model_path = "../../../hf/"


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

modelbase = load_models()

st.session_state['models'] = modelbase
#st.session_state['models'] = "flan-t5-base"

st.write("# Welcome to Streamlit! 👋")

st.sidebar.success("Select a demo above.")

st.markdown(
    """
    Streamlit is an open-source app framework built specifically for
    Machine Learning and Data Science projects.
    **👈 Select a demo from the sidebar** to see some examples
    of what Streamlit can do!
    ### Want to learn more?
    - Check out [streamlit.io](https://streamlit.io)
    - Jump into our [documentation](https://docs.streamlit.io)
    - Ask a question in our [community
        forums](https://discuss.streamlit.io)
    ### See more complex demos
    - Use a neural net to [analyze the Udacity Self-driving Car Image
        Dataset](https://github.com/streamlit/demo-self-driving)
    - Explore a [New York City rideshare dataset](https://github.com/streamlit/demo-uber-nyc-pickups)
"""
)
