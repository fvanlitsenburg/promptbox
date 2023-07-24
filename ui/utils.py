# pylint: disable=missing-timeout

from typing import List, Dict, Any, Tuple, Optional

import os
import logging
from time import sleep

import requests
import streamlit as st

from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever
from haystack.nodes import  PromptNode, PromptTemplate,AnswerParser,PromptModel
from elasticsearch import Elasticsearch

model_path = "../hf/"

from haystack.schema import Document



@st.cache_resource
def load_models(models=['flan-t5-base']):
    """ Load model and store it in cache. If a different model is loaded, we overwrite the old model.

    Takes a list of models as an input, *however*, until this code has been refactored the list should contain only one model.
    """
    models = dict.fromkeys(models,'')
    for model in models:
        models[model] = PromptModel(model_name_or_path=model_path+model,model_kwargs={'task_name':'text2text-generation','trust_remote_code':True})
        print(models[model])
        print("Successfully loaded " + model)
    p = Pipeline()
    print(p)
    print(models)
    return models

def build_ES_pipeline(promptmodel,prompt_text):
    """
    This function takes a promptmodel - preloaded from the load_models function and cached by Streamlit -
    and uses it to build a Pipeline that connects to an ElasticSearch DocumentStore,
    """

    ESdocument_store = ElasticsearchDocumentStore(index="document",host='localhost')

    BM25retriever = BM25Retriever(
        document_store=ESdocument_store
    )


    print("Loading node")
    prompt_node = PromptNode(promptmodel, default_prompt_template=prompt_text)
    prompt_node.debug = True

    print("Loading pipeline")
    ES_p = Pipeline()
    ES_p.add_node(component=BM25retriever, name="Retriever1", inputs=["Query"])
    ES_p.add_node(component=prompt_node, name="QA", inputs=["Retriever1"])
    return ES_p

def check_sentiment(query,model,in_docs,prompt_text):
    """
    This function takes a query and assesses whether the output suggests 'yes', 'no', or 'n/a'. The prompt_text should be geared towards giving 'yes.', 'no.', or'na.' as an output.

    It outputs a dictionary of document names, and whether the answer was 'yes' or 'no'; a list of outputs from running the model;
    and a list of document names where the answer was 'no'.

    Parameters
    - query: the query to give a yes/no answer to
    - model: the model to use, as loaded in load_models above
    - in_docs: a dictionary of document names and corresponding text snippets to check for a yes/no answer
    - prompt_text: the prompt template text to use

    """
    prompt_node = PromptNode(model, default_prompt_template=prompt_text)

    p2 = Pipeline()
    p2.add_node(component=prompt_node, name="QA", inputs=["Query"])

    documents = {}
    through_docs = []
    answers=[]
    for j in in_docs:
        answer = p2.run(query=query,params={"QA":{"documents":[Document(j['Answer'])]}})
        answers.append(answer)
        if r"no." in answer['results'][0].lower():
            documents[j['Document']] = 'No'
            through_docs.append(j['Document'])
        elif r"na." in answer['results'][0].lower():
            documents[j['Document']] = 'N/a'
        else:
            documents[j['Document']] = 'Yes'
    print(documents)
    return documents,answers,through_docs



API_ENDPOINT = os.getenv("API_ENDPOINT", "http://localhost:8000")
STATUS = "initialized"
HS_VERSION = "hs_version"
DOC_REQUEST = "query"
DOC_FEEDBACK = "feedback"
DOC_UPLOAD = "file-upload"


def haystack_is_ready():
    """
    Used to show the "Haystack is loading..." message
    """
    url = f"{API_ENDPOINT}/{STATUS}"
    try:
        if requests.get(url).status_code < 400:
            return True
    except Exception as e:
        logging.exception(e)
        sleep(1)  # To avoid spamming a non-existing endpoint at startup
    return False


@st.cache_data
def haystack_version():
    """
    Get the Haystack version from the REST API
    """
    url = f"{API_ENDPOINT}/{HS_VERSION}"
    return requests.get(url, timeout=0.1).json()["hs_version"]

def fetch_docs(store="ES"):
    """
    Fetches the names of files in the document store.
    """
    if store == "Weaviate":
        client = weaviate.Client(
        url = "http://localhost:8080",  # Replace with your endpoint
        )

        result = (
        client.query
        .aggregate('Document')
        .with_group_by_filter(['name'])
        .with_fields('groupedBy { value }')
        .do()
            )

        docs = [x['groupedBy']['value'] for x in result['data']['Aggregate']['Document']]

    if store == "ES":
        es = Elasticsearch()

        body = {
            "size": 0,
            "aggs": {
                "docs": {
                    "terms": {
                        "size": 100,
                        "field": "name"
                    }
                }
            }
        }

        result = es.search(index="document", body=body)

        docs = [x['key'] for x in result['aggregations']['docs']['buckets']]

    return docs

def query_listed_documents(query,documents,model,prompt_text):
    """ Takes a query, list of documents, and a pipeline to run retrieval on the specified documents.

    """

    output = []
    det_output = []

    p = build_ES_pipeline(model,prompt_text)

    for j in documents:
        res = p.run(query=query,params={"Retriever1": {"top_k": 10,"filters":{'name':[j]}},"debug": True})
        print(res)
        out = {'Document':j,'Answer':res['results'][0].replace('<pad>',"")}
        output.append(out)
        det_output.append(res)
    print(output)
    return output,det_output

def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("files", file)]
    response = requests.post(url, files=files,data={'split_length':'50'}).json()
    return response
