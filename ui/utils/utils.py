# pylint: disable=missing-timeout

from typing import List, Dict, Any, Tuple, Optional
import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
print(file_dir)


# The class. Relative imports mucking up so adding here
from haystack.nodes import PromptModelInvocationLayer
from llama_cpp import Llama
import os
from typing import Dict, List, Union, Type, Optional

import logging 

logger = logging.getLogger(__name__)

class LlamaCPPInvocationLayer(PromptModelInvocationLayer):
    def __init__(self, model_name_or_path: Union[str,os.PathLike],
        max_length: Optional[int] = 128,
        max_context: Optional[int] = 512,
        n_parts: Optional[int] = -1,
        seed: Optional[int]= 1337,
        f16_kv: Optional[bool] = True,
        logits_all: Optional[bool] = False,
        vocab_only: Optional[bool] = False,
        use_mmap: Optional[bool] = True,
        use_mlock: Optional[bool] = False,
        embedding: Optional[bool] = False,
        n_threads: Optional[int]= None,
        n_batch: Optional[int] = 512,
        last_n_tokens_size: Optional[int] = 64,
        lora_base: Optional[str] = None,
        lora_path: Optional[str] = None,
        verbose: Optional[bool] = True,
        **kwargs):

        """
        Creates a new Llama CPP InvocationLayer instance.

        :param model_name_or_path: The name or path of the underlying model.
        :param kwargs: See `https://abetlen.github.io/llama-cpp-python/#llama_cpp.llama.Llama.__init__`. For max_length, we use the 128 'max_tokens' setting.
        """
        if model_name_or_path is None or len(model_name_or_path) == 0:
            raise ValueError("model_name_or_path cannot be None or empty string")

        self.model_name_or_path = model_name_or_path
        self.max_context = max_context
        self.max_length = max_length
        self.n_parts = n_parts
        self.seed = seed
        self.f16_kv = f16_kv
        self.logits_all = logits_all
        self.vocab_only = vocab_only
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.embedding = embedding
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.last_n_tokens_size = last_n_tokens_size
        self.lora_base = lora_base
        self.lora_path = lora_path
        self.verbose = verbose
        self.model:Model = Llama(model_path = model_name_or_path,
            n_ctx = max_context,
            n_parts = n_parts,
            seed = seed,
            f16_kv = f16_kv,
            logits_all = logits_all,
            vocab_only = vocab_only,
            use_mmap = use_mmap,
            use_mlock = use_mlock,
            embedding = embedding,
            n_threads = n_threads,
            n_batch = n_batch,
            last_n_tokens_size = last_n_tokens_size,
            lora_base = lora_base,
            lora_path = lora_path,
            verbose = verbose)
    

    def _ensure_token_limit(self, prompt: Union[str, List[Dict[str, str]]]) -> Union[str, List[Dict[str, str]]]:
        """Ensure that length of the prompt and answer is within the maximum token length of the PromptModel.

        :param prompt: Prompt text to be sent to the generative model.
        """
        if not isinstance(prompt, str):
            raise ValueError(f"Prompt must be of type str but got {type(prompt)}")
        
        context_length = self.model.n_ctx()
        tokenized_prompt = self.model.tokenize(bytes(prompt,'utf-8'))
        if len(tokenized_prompt) + self.max_length > context_length:
            logger.warning(
            "The prompt has been truncated from %s tokens to %s tokens so that the prompt length and "
            "answer length (%s tokens) fit within the max token limit (%s tokens). "
            "Shorten the prompt to prevent it from being cut off",
            len(tokenized_prompt),
            max(0, context_length -  self.max_length),
            self.max_length,
            context_length,
            )
            print(bytes.decode(self.model.detokenize(tokenized_prompt[:max(0, context_length -  self.max_length)]),'utf-8'))
            print(type(bytes.decode(self.model.detokenize(tokenized_prompt[:max(0, context_length -  self.max_length)]),'utf-8')))
            return bytes.decode(self.model.detokenize(tokenized_prompt[:max(0, context_length -  self.max_length)]),'utf-8')

        return prompt

    def invoke(self, *args, **kwargs):
        """
        It takes a prompt and returns a list of generated text using the underlying model.
        :return: A list of generated text.
        """
        output: List[Dict[str, str]] = []
        stream = kwargs.pop("stream",False)

        generated_texts = []
        
        if kwargs and "prompt" in kwargs:
            prompt = kwargs.pop("prompt")

            # For more details refer to call documentation for Llama CPP https://abetlen.github.io/llama-cpp-python/#llama_cpp.llama.Llama.__call__
            model_input_kwargs = {
                key: kwargs[key]
                for key in [
                    "suffix",
                    "max_tokens",
                    "temperature",
                    "top_p",
                    "logprobs",
                    "echo",
                    "repeat_penalty",
                    "top_k",
                ]
                if key in kwargs
            }
            
        if stream:
            for token in self.model(prompt,stream=True,**model_input_kwargs):
                generated_texts.append(token['choices'][0]['text'])
        else:
            output = self.model(prompt,**model_input_kwargs)
            generated_texts = [o['text'] for o in output['choices']]
        return generated_texts

    def supports(cls, model_name_or_path: str, **kwargs) -> bool:
        """
        Checks if the given model is supported by this invocation layer.

        :param model_name_or_path: The name or path of the model.
        :param kwargs: Additional keyword arguments passed to the underlying model which might be used to determine
        if the model is supported.
        :return: True if this invocation layer supports the model, False otherwise.
        """
        #I guess there is not much to validate here ¯\_(ツ)_/¯
        return model_name_or_path is not None and len(model_name_or_path) > 0

cwd = os.getcwd()
print(cwd)

#from utils.llamalayer import LlamaCPPInvocationLayer

import logging
from time import sleep

import requests
import streamlit as st

from haystack import Pipeline
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.nodes import BM25Retriever, EmbeddingRetriever
from haystack.nodes import  PromptNode, PromptTemplate,AnswerParser,PromptModel
from elasticsearch import Elasticsearch

model_path = "../hf/"

from haystack.schema import Document



@st.cache_resource
def load_models(models=['flan-t5-base']):
    """ Load model and store it in cache. If a different model is loaded, we overwrite the old model.

    Takes a list of models as an input, *however*, until this code has been refactored the input list should contain only one model.
    """
    models = dict.fromkeys(models,'')
    cpp_models = {'llama-cpp':'llama2-7b/llama-2-7b-chat.ggmlv3.q4_1.bin','nous':'nous/nous-hermes-llama-2-7b.ggmlv3.q3_K_M.bin'}
    for model in models:
        if model in cpp_models.keys():
            models[model] = PromptModel(model_name_or_path="../llama.cpp/models/"+cpp_models[model],invocation_layer_class=LlamaCPPInvocationLayer,model_kwargs={'max_context':4096})
        else:
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

    ESdocument_store = ElasticsearchDocumentStore(index="document",host='localhost') # Comment to change to embedding retrieval

    # ESdocument_store = ElasticsearchDocumentStore(index="document",host='localhost',embedding_dim=384) # Uncomment for embedding retrieval

    Retriever = BM25Retriever(document_store=ESdocument_store) # Comment to change to embedding retrieval

    #Retriever = EmbeddingRetriever(document_store=ESdocument_store, embedding_model="sentence-transformers/all-MiniLM-L6-v2", model_format="sentence_transformers", top_k=5) # Uncomment for embedding retrieval

    print("Loading node")
    prompt_node = PromptNode(promptmodel, default_prompt_template=prompt_text,model_kwargs={"max_tokens":512})
    prompt_node.debug = True

    print("Loading pipeline")
    ES_p = Pipeline()
    ES_p.add_node(component=Retriever, name="Retriever1", inputs=["Query"])
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
        res = p.run(query=query,params={"Retriever1": {"top_k": 5,"filters":{'name':[j]}},"debug": True})
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
