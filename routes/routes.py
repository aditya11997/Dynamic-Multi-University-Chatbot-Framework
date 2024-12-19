import aiohttp
import os
import pickle
import logging
import asyncio

from flask import Blueprint, jsonify, request

from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langdetect import detect_langs
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator

from src.config import BotConfig
from src.data_load import KnowledgeBaseManager 

config = BotConfig()

# Initialize SERP API Key
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Define the Blueprint for routes
routes = Blueprint("routes", __name__)

# Logger setup
logging.basicConfig(level=logging.INFO)

model_choices = {
    "Mixtral-8x7b-32768": "Mixtral-8x7b-32768",
    "Gemma2-9b-It": "Gemma2-9b-It"
}

model_name = model_choices["Mixtral-8x7b-32768"]
llm = ChatGroq(groq_api_key=config.GROQ_API_KEY, model_name=model_name)

# Load embeddings model (shared across all chatbots)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Cache for loaded university data
university_cache = {}

def detect_and_translate(user_input, target_lang="en"):
    """
    Detect the language of user input and translate if necessary.
    """
    detected_langs = detect_langs(user_input)
    input_lang = detected_langs[0].lang if detected_langs else "en"
    translated_input = user_input

    if input_lang != target_lang:
        translated_input = GoogleTranslator(source="auto", target=target_lang).translate(user_input)

    return translated_input, input_lang


async def search_web(query):
    """
    Search the web using SERPAPI and return results.
    """
    url = "https://serpapi.com/search"
    params = {"q": query, "api_key": SERPAPI_KEY}
    async with aiohttp.ClientSession() as session:
        async with session.get(url, params=params) as response:
            return await response.json()


async def rank_results(search_results, user_input, embeddings):
    """
    Rank web search results based on relevance to user input.
    """
    try:
        organic_results = search_results.get("organic_results", [])
        input_embedding = embeddings.embed_query(user_input)
        result_embeddings = [embeddings.embed_query(res["snippet"]) for res in organic_results]
        similarities = [cosine_similarity([input_embedding], [emb])[0][0] for emb in result_embeddings]
        ranked_results = sorted(zip(organic_results, similarities), key=lambda x: x[1], reverse=True)
        return [result[0] for result in ranked_results[:3]]
    except Exception as e:
        logging.error(f"Error ranking results: {e}")
        return []

@routes.route('/<university_name>', methods=['POST'])
async def load_chatbot(university_name):
    """
    Load chatbot dynamically based on the university name.
    If the data for the university is not available, return an error response.
    """
    university_name = university_name.lower().replace("-", "_")
    user_input = request.json.get("message", "")
    user_language = request.json.get("language", "en-US")
    logging.info(f"Loading data for {university_name}")

    # Check if the university data is already cached
    if university_name not in university_cache:
        university_file = f"data/{university_name}_data.pkl"

        if not os.path.exists(university_file):
            logging.warning(f"Data file for {university_name} does not exist.")
            return jsonify({
                "error": f"The chatbot for {university_name.replace('_', ' ').capitalize()} is not registered yet. Please contact the administrator."
            }), 404

        # Load the data for the university
        try:
            logging.info(f"Checking for data file: {university_file}")
            with open(university_file, "rb") as f:
                documents = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading data for {university_name}: {e}")
            return jsonify({"error": "Failed to load university data. Please try again later."}), 500

        # Prepare the chatbot chain
        try:
            # Split documents for vector creation
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            split_documents = text_splitter.split_documents(documents)

            # Create FAISS vector store
            vectors = FAISS.from_documents(split_documents, embeddings)
            univ_name = university_name.replace('_', ' ')
            # Define the dynamic prompt template
            prompt_template = ChatPromptTemplate.from_template(
                f"""
                You are {univ_name.capitalize()}Bot, the official chatbot for {univ_name.capitalize()}.
                Your role is to assist users by providing accurate and helpful information about {univ_name.capitalize()}.

                <context>
                {{context}}
                </context>
                Question: {{input}}
                """
            )

            # Create document and retrieval chains
            document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt_template)
            retriever = vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            # Cache the prepared data
            university_cache[university_name] = retrieval_chain

        except Exception as e:
            logging.error(f"Error creating chatbot for {university_name}: {e}")
            return jsonify({"error": "Failed to create chatbot. Please try again later."}), 500

    # Get the cached retrieval chain
    retrieval_chain = university_cache[university_name]
    translated_input, detected_lang = detect_and_translate(user_input)

    # Asynchronous web search and chatbot inference
    search_task = search_web(translated_input)
    chatbot_task = asyncio.to_thread(retrieval_chain.invoke, {"input": translated_input})

    search_results, chatbot_response = await asyncio.gather(search_task, chatbot_task)

    # Process web search results
    ranked_results = await rank_results(search_results, translated_input, embeddings)
    snippets = [result["snippet"] for result in ranked_results]
    sources = "\n".join([f"[{res['title']}]({res['link']})" for res in ranked_results])

    # Create enriched response
    enriched_response = f"{chatbot_response['answer']}\n\n**Additional Info from the Web:** {' '.join(snippets)}\n\n**Sources:** {sources}"

    # Handle language translation for the enriched response
    if detected_lang != "en":
        print(f"Detected input lang: {detected_lang}")
        if detected_lang in {"zh-CN", "zh-TW", "ko", "zh-cn", "zh-tw"}:
            chatbot_response = GoogleTranslator(source="auto", target="zh-CN").translate(enriched_response)
        else:
            chatbot_response = GoogleTranslator(source="auto", target=detected_lang).translate(enriched_response)
    elif user_language == "zh-CN":
        chatbot_response = GoogleTranslator(
            source="auto",
            target="zh-CN"
        ).translate(enriched_response)
    else:
        chatbot_response = enriched_response

    return jsonify({"response": chatbot_response})

@routes.route('/register-university', methods=['POST'])
def register_university():
    """
    Endpoint to register a university by creating its database.
    Expects `university_name` and `sitemap_url` in the request body.
    """
    try:
        data = request.json
        university_name = data.get('university_name')
        sitemap_url = data.get('sitemap_url')
        manual_data = data.get('manual_data')

        logging.info(f"Incoming request data: {data}")

        if not university_name:
            return jsonify({"error": "University name is required."}), 400

        # Initialize the KnowledgeBaseManager
        manager = KnowledgeBaseManager(university_name=university_name, sitemap_url=sitemap_url)

        documents = []

        # If manual data is provided, process it
        if manual_data:
            documents.extend([
                Document(page_content=text.strip(), metadata={"source": "user-input"})
                for text in manual_data.split("\n") if text.strip()
            ])
            logging.info(f"Processed {len(documents)} documents from textual input.")

        # Fetch URLs and process them
        if sitemap_url:
            urls = manager.fetch_urls_from_sitemap_or_html()
            if not urls:
                return jsonify({"error": "Failed to fetch URLs from the sitemap. Please check the URL."}), 400
            
            manager.process_urls_in_batches(urls)
            logging.info("Processed URLs and added to the knowledge base.")
        
        # Save the combined data
        if documents:
            with open(manager.data_file, 'wb') as f:
                pickle.dump(documents, f)
            logging.info(f"Saved {len(documents)} documents to {manager.data_file}.")
        return jsonify({"message": f"University '{university_name}' has been successfully registered."}), 200
    except Exception as e:
        logging.error(f"Error in registering university: {e}")
        return jsonify({"error": "An error occurred while registering the university. Please try again later."}), 500

@routes.route('/registered-universities', methods=['GET'])
def get_registered_universities():
    """Return the list of registered universities."""
    try:
        data_folder = "data"  # Adjust this path as needed
        university_files = [
            file.replace("_data.pkl", "").replace("_", "-")
            for file in os.listdir(data_folder)
            if file.endswith("_data.pkl")
        ]
        return jsonify({"universities": university_files})
    except Exception as e:
        logging.error(f"Error fetching registered universities: {e}")
        return jsonify({"error": "Failed to fetch registered universities"}), 500

@routes.route('/update-university-data', methods=['POST'])
def update_university_data():
    """
    Endpoint to update existing university-specific pickle files with additional data.
    Expects `university_name` and `manual_data` in the request body.
    """
    try:
        data = request.json
        university_name = data.get('university_name')
        manual_data = data.get('manual_data')

        if not university_name or not manual_data:
            return jsonify({"error": "University name and manual data are required."}), 400

        # Initialize KnowledgeBaseManager for the university
        manager = KnowledgeBaseManager(university_name=university_name, sitemap_url=None)

        # Process manual data
        documents = manager.process_manual_data(manual_data)
        if not documents:
            return jsonify({"error": "No valid data provided to update."}), 400

        # Save data into the existing pickle file
        existing_data = manager.load_data()
        updated_data = existing_data + documents
        with open(manager.data_file, 'wb') as f:
            pickle.dump(updated_data, f)
        
        logging.info(f"Successfully updated {len(documents)} documents for {university_name}.")
        return jsonify({"message": f"Data for '{university_name}' updated successfully.", "total_documents": len(updated_data)}), 200

    except Exception as e:
        logging.error(f"Error updating university data: {e}")
        return jsonify({"error": "An error occurred while updating the data. Please try again later."}), 500

