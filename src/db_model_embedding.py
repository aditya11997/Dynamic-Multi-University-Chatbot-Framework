import torch
from langchain import HuggingFacePipeline  # Importing LangChain's HuggingFacePipeline
from langchain.chains import RetrievalQA  # For setting up a retrieval-based QA system
from langchain.document_loaders import HuggingFaceDatasetLoader, UnstructuredURLLoader  # For loading datasets
from langchain.embeddings import HuggingFaceEmbeddings  # For working with HuggingFace embeddings
from langchain.llms import HuggingFacePipeline  # LangChain's wrapper for HuggingFace's pipelines
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter  # For splitting text
from langchain.vectorstores import FAISS, Pinecone, Weaviate  # For using different vector storage solutions
from langchain_core.output_parsers import StrOutputParser  # For parsing outputs in LangChain
from langchain_core.runnables import RunnablePassthrough  # For creating runnable actions in LangChain
from langchain.prompts import PromptTemplate

from transformers import (AutoModelForCausalLM, AutoModelForQuestionAnswering,
                          AutoTokenizer, BitsAndBytesConfig, pipeline) 
import os

from src.config import Args
from src.data_load import DataManager  

class TextProcessor:
    def __init__(self, args):
        self.args = args
        
        self.embedding_model = self.args.embedding_model
        
        
        # Model and encoding configurations
        self.model_kwargs = self.args.model_kwargs
        self.encode_kwargs = self.args.encode_kwargs
        
        # Initialize embeddings processor
        self.embeddings_processor = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs=self.model_kwargs,
            encode_kwargs=self.encode_kwargs
        )

    def process_documents(self, data):
        """ Split documents and generate embeddings """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        docs = text_splitter.split_documents(data)
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model,
                                            model_kwargs=self.model_kwargs,
                                            encode_kwargs=self.encode_kwargs
                                            )
        return docs, embeddings

    def setup_faiss_retriever(self, docs, embeddings):
        """ Setup FAISS as a retriever """
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        return retriever

    def init_text_generation_pipeline(self):
        """ Initialize the text generation pipeline with model quantization """
        # model_name = self.args.quantised_model
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        # )
        # model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # ## Save the model and load it
        # print("="*30)
        # print("save model")
        
        # # Save the model
        # model.save_pretrained(os.path.join(self.args.output_dir,"saved_model"))

        # # Save the tokenizer
        # tokenizer.save_pretrained(os.path.join(self.args.output_dir,"saved_tokenizer"))
        
        
        # Load the model
        model = AutoModelForCausalLM.from_pretrained(os.path.join(self.args.output_dir,"saved_model"))

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.args.output_dir,"saved_tokenizer"))
        
        print("load model")
        print("="*60)
        
        

        text_generation_pipeline = pipeline(
            model=model,
            tokenizer=tokenizer,
            task="text-generation",
            temperature=0.7,  # Set to a more conservative value
            top_k=50,  # Limit to top-k predictions to avoid low probability tokens
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=True,
            max_new_tokens=400,
            num_return_sequences=1,  # Generate one sequence
        )
        
        llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

        prompt_template = """
                                <|system|>
                                Answer the question based on your knowledge. Use the following context to help:

                                {context}

                                </s>
                                <|user|>
                                {question}
                                </s>
                                <|assistant|>

                            """
        
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template,
        )

        llm_chain = prompt | llm | StrOutputParser()
        return llm_chain

    
    def setup_rag_chain(self, retriever, llm_chain):
        """ Set up the RAG chain combining the retriever and LLM chain """
        rag_chain = {"context": retriever, "question": RunnablePassthrough()} | llm_chain
        return rag_chain

# if __name__ == "__main__":
#     args = Args()
#     text_processor = TextProcessor(args)
#     data_manager = DataManager(args)
#     data = data_manager.load_data_from_file()
#     docs, embeddings = text_processor.process_documents(data)
#     retriever = text_processor.setup_faiss_retriever(docs, embeddings)
#     llm_chain = text_processor.init_text_generation_pipeline()
#     rag_chain = text_processor.setup_rag_chain(retriever, llm_chain)
#     print("run successfully")
#     # Here you would use `retriever` and `llm_chain` as needed in your application.
