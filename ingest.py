"""Load html from files, clean up, split, ingest into Weaviate."""
import logging
import os
import re
from parser import langchain_docs_extractor

#import weaviate
from bs4 import BeautifulSoup, SoupStrainer
from langchain.document_loaders import RecursiveUrlLoader, SitemapLoader
from langchain.indexes import SQLRecordManager
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)
#from langchain.vectorstores.weaviate import Weaviate
from langchain_community.vectorstores.azuresearch import AzureSearch


from _index import index
#from chain import get_embedding_model
#from constants import AZURE_SEARCH_INDEX #WEAVIATE_DOCS_INDEX_NAME

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__f3153da19a294dbda5dc6b5da3265156"
os.environ["LANGCHAIN_PROJECT"] = "chatagent"
#os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangSmith API Key:")

from langchain.globals import set_debug
from langchain.globals import set_verbose

set_debug(True)
set_verbose(True)


from .helpers.LLMHelper import LLMHelper
from .helpers.EnvHelper import EnvHelper

llm_helper = LLMHelper()
env_helper = EnvHelper()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# WEAVIATE_URL = os.environ["WEAVIATE_URL"]
# WEAVIATE_API_KEY = os.environ["WEAVIATE_API_KEY"]
RECORD_MANAGER_DB_URL = os.environ["RECORD_MANAGER_DB_URL"]

get_embedding_model = llm_helper.get_embedding_model

def metadata_extractor(meta: dict, soup: BeautifulSoup) -> dict:
    title = soup.find("title")
    description = soup.find("meta", attrs={"name": "description"})
    html = soup.find("html")
    return {
        "source": meta["loc"],
        "title": title.get_text() if title else "",
        "description": description.get("content", "") if description else "",
        "language": html.get("lang", "") if html else "",
        **meta,
    }


def load_langchain_docs():
    return SitemapLoader(
        "https://python.langchain.com/sitemap.xml",
        filter_urls=["https://python.langchain.com/"],
        parsing_function=langchain_docs_extractor,
        default_parser="lxml",
        bs_kwargs={
            "parse_only": SoupStrainer(
                name=("article", "title", "html", "lang", "content")
            ),
        },
        meta_function=metadata_extractor,
    ).load()


def load_langsmith_docs():
    return RecursiveUrlLoader(
        url="https://docs.smith.langchain.com/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
    ).load()


def simple_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return re.sub(r"\n\n+", "\n\n", soup.text).strip()


def load_api_docs():
    return RecursiveUrlLoader(
        url="https://api.python.langchain.com/en/latest/",
        max_depth=8,
        extractor=simple_extractor,
        prevent_outside=True,
        use_async=True,
        timeout=600,
        # Drop trailing / to avoid duplicate pages.
        link_regex=(
            f"href=[\"']{PREFIXES_TO_IGNORE_REGEX}((?:{SUFFIXES_TO_IGNORE_REGEX}.)*?)"
            r"(?:[\#'\"]|\/[\#'\"])"
        ),
        check_response_status=True,
        exclude_dirs=(
            "https://api.python.langchain.com/en/latest/_sources",
            "https://api.python.langchain.com/en/latest/_modules",
        ),
    ).load()


def ingest_docs():
    docs_from_documentation = load_langchain_docs()
    logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
    docs_from_api = load_api_docs()
    logger.info(f"Loaded {len(docs_from_api)} docs from API")
    docs_from_langsmith = load_langsmith_docs()
    logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    docs_transformed = text_splitter.split_documents(
        docs_from_documentation + docs_from_api + docs_from_langsmith
    )

    # We try to return 'source' and 'title' metadata when querying vector store and
    # Azure AI Search will error at query time if one of the attributes is missing from a
    # retrieved document.
    for doc in docs_transformed:
        if "source" not in doc.metadata:
            doc.metadata["source"] = ""
        if "title" not in doc.metadata:
            doc.metadata["title"] = ""

    vector_store_address=env_helper.AZURE_SEARCH_SERVICE
    vector_store_password=env_helper.AZURE_SEARCH_KEY if env_helper.AZURE_AUTH_TYPE == "keys" else None
    index_name=env_helper.AZURE_SEARCH_INDEX,

    embeddings = get_embedding_model()
    vector_store = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )

    record_manager = SQLRecordManager(
        f"azuresearch/{index_name}", db_url=RECORD_MANAGER_DB_URL
    )
    record_manager.create_schema()

    indexing_stats = index(
        docs_transformed,
        record_manager,
        vector_store,
        cleanup="full",
        source_id_key="source",
        force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
    )

    logger.info(f"Indexing stats: {indexing_stats}")
    num_vecs = vector_store.get_num_vectors()
    logger.info(
        f"LangChain now has this many vectors: {num_vecs}",
    )

# def ingest_docs():
#     docs_from_documentation = load_langchain_docs()
#     logger.info(f"Loaded {len(docs_from_documentation)} docs from documentation")
#     docs_from_api = load_api_docs()
#     logger.info(f"Loaded {len(docs_from_api)} docs from API")
#     docs_from_langsmith = load_langsmith_docs()
#     logger.info(f"Loaded {len(docs_from_langsmith)} docs from Langsmith")

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
#     docs_transformed = text_splitter.split_documents(
#         docs_from_documentation + docs_from_api + docs_from_langsmith
#     )

#     # We try to return 'source' and 'title' metadata when querying vector store and
#     # Weaviate will error at query time if one of the attributes is missing from a
#     # retrieved document.
#     for doc in docs_transformed:
#         if "source" not in doc.metadata:
#             doc.metadata["source"] = ""
#         if "title" not in doc.metadata:
#             doc.metadata["title"] = ""

#     client = weaviate.Client(
#         url=WEAVIATE_URL,
#         auth_client_secret=weaviate.AuthApiKey(api_key=WEAVIATE_API_KEY),
#     )
#     embedding = get_embedding_model()
#     vectorstore = Weaviate(
#         client=client,
#         index_name=WEAVIATE_DOCS_INDEX_NAME,
#         text_key="text",
#         embedding=embedding,
#         by_text=False,
#         attributes=["source", "title"],
#     )

#     record_manager = SQLRecordManager(
#         f"weaviate/{WEAVIATE_DOCS_INDEX_NAME}", db_url=RECORD_MANAGER_DB_URL
#     )
#     record_manager.create_schema()

#     indexing_stats = index(
#         docs_transformed,
#         record_manager,
#         vectorstore,
#         cleanup="full",
#         source_id_key="source",
#         force_update=(os.environ.get("FORCE_UPDATE") or "false").lower() == "true",
#     )

#     logger.info(f"Indexing stats: {indexing_stats}")
#     num_vecs = client.query.aggregate(WEAVIATE_DOCS_INDEX_NAME).with_meta_count().do()
#     logger.info(
#         f"LangChain now has this many vectors: {num_vecs}",
#     )


# if __name__ == "__main__":
#     ingest_docs()
