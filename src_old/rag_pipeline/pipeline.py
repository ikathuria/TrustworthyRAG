import json

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from utils.logger import Logger


class CyberSecurityRAG:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self.logger = Logger("RAG_Pipeline")

    def load_cve_data(self, cve_file_path):
        try:
            with open(cve_file_path, 'r') as f:
                cve_data = json.load(f)

            documents = []
            for cve in cve_data.get('CVE_Items', []):
                cve_id = cve['cve']['CVE_data_meta']['ID']
                description = cve['cve']['description']['description_data'][0]['value']
                severity = cve.get('impact', {}).get('baseMetricV3', {}).get(
                    'cvssV3', {}).get('baseScore', 'N/A')

                doc_content = f"CVE ID: {cve_id}\nDescription: {description}\nSeverity: {severity}"
                documents.append({'content': doc_content, 'metadata': {
                                 'cve_id': cve_id, 'severity': severity, 'source': 'CVE', 'type': 'vulnerability'}})

            self.logger.info(f"Loaded {len(documents)} CVE documents")
            return documents
        except Exception as e:
            self.logger.error(f"Error loading CVE data: {e}")
            return []

    def build_vector_store(self, documents):
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200)
            texts, metadatas = [], []
            for doc in documents:
                chunks = text_splitter.split_text(doc['content'])
                for chunk in chunks:
                    texts.append(chunk)
                    metadatas.append(doc['metadata'])
            self.vector_store = FAISS.from_texts(
                texts, self.embeddings, metadatas=metadatas)

            self.retriever = self.vector_store.as_retriever(
                search_kwargs={'k': 5, 'include_metadata': True})
            self.logger.info(f"Built vector store with {len(texts)} chunks")
            return True
        except Exception as e:
            self.logger.error(f"Error building vector store: {e}")
            return False

    def setup_qa_chain(self, llm):
        if not self.retriever:
            raise ValueError("Vector store and retriever must be built first")

        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            verbose=True,
        )
        self.logger.info("QA chain setup complete")

    def query(self, question):
        if not self.qa_chain:
            raise ValueError("QA chain must be setup first")
        try:
            result = self.qa_chain({"query": question})
            self.logger.info(f"Query: {question}")
            for i, doc in enumerate(result['source_documents']):
                self.logger.info(f"Source {i+1}: {doc.metadata}")
            return {'answer': result['result'], 'source_documents': result['source_documents'], 'query': question}
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return None
