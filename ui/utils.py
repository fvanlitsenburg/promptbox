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

model_path = "../../hf/"

from haystack.schema import Document



@st.cache_resource
def load_models(models=['flan-t5-base']):
    ''' Load models and store them in cache. On a first trial run, it's recommended to use flan-t5-base, which can be ran relatively easily on CPU. ,'fastchat-t5-3b-v1.0'
    '''
    #models = {"flan-t5-base":''}
    models = dict.fromkeys(models,'')
    for model in models:
        models[model] = PromptModel(model_name_or_path=model_path+model,model_kwargs={'task_name':'text2text-generation'})
        print(models[model])
        print("Successfully loaded " + model)
    p = Pipeline()
    print(p)
    return models

def load_prompt(prompt_text):
    lfqa_prompt = PromptTemplate(name="lfqa",
                             prompt_text=prompt_text,
                             output_parser=AnswerParser(),)
    return lfqa_prompt

def build_ES_pipeline(promptmodel,prompt_text):

    ESdocument_store = ElasticsearchDocumentStore(index="document",host='localhost')

    BM25retriever = BM25Retriever(
        document_store=ESdocument_store
    )


    #model = PromptModel(model_name_or_path="lmsys/fastchat-t5-3b-v1.0",model_kwargs={'task_name':'text2text-generation'})
    #model = PromptModel(model_name_or_path=model_path+model,model_kwargs={'task_name':'text2text-generation'})
    #prompt_node = PromptNode(models[model], default_prompt_template='question-answering-per-document',)

    #othermodel = PromptModel(model_name_or_path=model_path+'flan-t5-base',model_kwargs={'task_name':'text2text-generation'})
    #print(othermodel)

    print("Loading node")
    print(promptmodel)
    print(type(promptmodel))
    prompt_node = PromptNode(promptmodel, default_prompt_template=load_prompt(prompt_text))


    print(prompt_node)
    #prompt_node = model
    prompt_node.debug = True

    print("Loading pipeline")
    ES_p = Pipeline()
    ES_p.add_node(component=BM25retriever, name="Retriever1", inputs=["Query"])
    ES_p.add_node(component=prompt_node, name="QA", inputs=["Retriever1"])
    return ES_p

def check_sentiment(model,in_docs,prompt_text):

    prompt_node = PromptNode(model, default_prompt_template=load_prompt(prompt_text))

    p2 = Pipeline()
    p2.add_node(component=prompt_node, name="QA", inputs=["Query"])

    documents = {}
    through_docs = []
    answers=[]
    for j in in_docs:
        answer = p2.run(query="Does the answer indicate yes or no?",params={"QA":{"documents":[Document(j['Answer'])]}})
        answers.append(answer)
        if answer['answers'][0].answer == 'no':
            documents[j['Document']] = 'No'
            through_docs.append(j['Document'])
        else:
            documents[j['Document']] = 'Yes'
    print(documents)
    print(answers)
    return documents,answers,through_docs


'''@st.cache_resource
def load_models(model,prompt_text):
    models = {"flan-t5-base":'','fastchat-t5-3b-v1.0':''}
    global use_model
    use_model = build_ES_pipeline(model,prompt_text)
    print("Successfully loaded " + model)
    return use_model'''

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
    '''
    Fetches the names of files in the document store.
    '''
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
    ''' Takes a query, list of documents, and a pipeline to run retrieval on the specified documents.

    '''

    output = []
    det_output = []

    p = build_ES_pipeline(model,prompt_text)

    for j in documents:
        res = p.run(query=query,params={"Retriever1": {"top_k": 5,"filters":{'name':[j]}},"debug": True})
        out = {'Document':j,'Answer':res['answers'][0].answer}
        output.append(out)
        det_output.append(res)
    print(output)
    return output,det_output

def query_each_document(query,model,prompt_text):

    #p = load_models(model,prompt_text)

    p = build_ES_pipeline(model,prompt_text)

    print(model)
    print(query)
    print(prompt_text)
    print(p)

    print("fetching docs")
    docs = fetch_docs()

    output = []
    det_output = []

    for j in docs:
        try:
            res = p.run(query=query,params={"Retriever1": {"top_k": 5,"filters":{'name':[j]}},"debug": True})
        except:
            print(res["_debug"])
        out = {'Document':j,'Answer':res['answers'][0].answer}
        output.append(out)
        det_output.append(res["_debug"])
    print(output)
    print(p.components)
    return output,det_output

def query(query, filters={}, top_k_retriever=3) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Send a query to the REST API and parse the answer.
    Returns both a ready-to-use representation of the results and the raw JSON.
    """

    url = f"{API_ENDPOINT}/{DOC_REQUEST}"
    params = {"filters": filters, "Retriever": {"top_k": top_k_retriever}, "Reader": {"top_k": top_k_reader}}
    req = {"query": query, "params": params}
    response_raw = requests.post(url, json=req)

    if response_raw.status_code >= 400 and response_raw.status_code != 503:
        raise Exception(f"{vars(response_raw)}")

    response = response_raw.json()
    if "errors" in response:
        raise Exception(", ".join(response["errors"]))

    # Format response
    results = []
    answers = response["answers"]
    for answer in answers:
        if answer.get("answer", None):
            results.append(
                {
                    "context": "..." + answer["context"] + "...",
                    "answer": answer.get("answer", None),
                    "source": answer["meta"]["name"],
                    "relevance": round(answer["score"] * 100, 2),
                    "document": [doc for doc in response["documents"] if doc["id"] == answer["document_id"]][0],
                    "offset_start_in_doc": answer["offsets_in_document"][0]["start"],
                    "_raw": answer,
                }
            )
        else:
            results.append(
                {
                    "context": None,
                    "answer": None,
                    "document": None,
                    "relevance": round(answer["score"] * 100, 2),
                    "_raw": answer,
                }
            )
    return results, response


def send_feedback(query, answer_obj, is_correct_answer, is_correct_document, document) -> None:
    """
    Send a feedback (label) to the REST API
    """
    url = f"{API_ENDPOINT}/{DOC_FEEDBACK}"
    req = {
        "query": query,
        "document": document,
        "is_correct_answer": is_correct_answer,
        "is_correct_document": is_correct_document,
        "origin": "user-feedback",
        "answer": answer_obj,
    }
    response_raw = requests.post(url, json=req)
    if response_raw.status_code >= 400:
        raise ValueError(f"An error was returned [code {response_raw.status_code}]: {response_raw.json()}")


def upload_doc(file):
    url = f"{API_ENDPOINT}/{DOC_UPLOAD}"
    files = [("files", file)]
    response = requests.post(url, files=files,data={'split_length':'50'}).json()
    return response


def get_backlink(result) -> Tuple[Optional[str], Optional[str]]:
    if result.get("document", None):
        doc = result["document"]
        if isinstance(doc, dict):
            if doc.get("meta", None):
                if isinstance(doc["meta"], dict):
                    if doc["meta"].get("url", None) and doc["meta"].get("title", None):
                        return doc["meta"]["url"], doc["meta"]["title"]
    return None, None
