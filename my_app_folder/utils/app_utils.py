import aiohttp
from deep_translator import GoogleTranslator
from langdetect import detect_langs
from sklearn.metrics.pairwise import cosine_similarity
import logging

logging.basicConfig(level=logging.INFO)

# Asynchronous web search using SerpAPI
async def search_web(query, serpapi_key):
    """
    Perform a web search using SerpAPI.

    Args:
        query (str): The search query.
        serpapi_key (str): The SerpAPI key for authentication.

    Returns:
        dict: The JSON response from the API or an empty dictionary on error.
    """
    try:
        url = "https://serpapi.com/search"
        params = {"q": query, "api_key": serpapi_key}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                return await response.json()
    except Exception as e:
        logging.error(f"Error in search_web: {e}")
        return {}

# Rank results based on cosine similarity
async def rank_results(search_results, user_input, embeddings):
    """
    Rank web search results based on similarity to user input.

    Args:
        search_results (dict): The JSON response from the web search.
        user_input (str): The user's input message.
        embeddings (object): An embeddings object for computing vector similarity.

    Returns:
        list: The top 3 ranked results.
    """
    try:
        organic_results = search_results.get("organic_results", [])
        input_embedding = embeddings.embed_query(user_input)
        result_embeddings = [embeddings.embed_query(res['snippet']) for res in organic_results]
        similarities = [cosine_similarity([input_embedding], [emb])[0][0] for emb in result_embeddings]
        ranked_results = sorted(zip(organic_results, similarities), key=lambda x: x[1], reverse=True)
        return [result[0] for result in ranked_results[:3]]
    except Exception as e:
        logging.error(f"Error in rank_results: {e}")
        return []

# Summarize web results using LLM
def summarize_with_llm(ranked_results, llm):
    """
    Summarize the top ranked results using an LLM.

    Args:
        ranked_results (list): The top ranked results.
        llm (object): The LLM object for generating summaries.

    Returns:
        str: The summary or an error message.
    """
    try:
        snippets = [result['snippet'].replace("...", "") for result in ranked_results]
        combined_text = " ".join(snippets)
        summarization_prompt = f"Summarize the following information concisely: {combined_text}"
        return llm.invoke(summarization_prompt).content
    except Exception as e:
        logging.error(f"Error in summarize_with_llm: {e}")
        return "Could not generate a summary at this time."

# Format web sources for display
def format_sources(ranked_results):
    """
    Format web sources into a clickable link list.

    Args:
        ranked_results (list): The top ranked results.

    Returns:
        str: A formatted string of sources or an error message.
    """
    try:
        return "\n".join([f"[{result['title']}]({result['link']})" for result in ranked_results])
    except Exception as e:
        logging.error(f"Error in format_sources: {e}")
        return "Sources could not be formatted."

# Detect language and translate input to English
def detect_and_translate(user_input, user_language):
    """
    Detect the input language and translate it to English if necessary.

    Args:
        user_input (str): The user's input message.
        user_language (str): The default language for translation.

    Returns:
        tuple: Translated input and detected input language.
    """
    try:
        detected_languages = detect_langs(user_input)
        input_lang = detected_languages[0].lang if detected_languages else 'en'
        if input_lang != 'en':
            translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
        else:
            translated_input = user_input
        return translated_input, input_lang
    except Exception as e:
        logging.error(f"Error in detect_and_translate: {e}")
        return user_input, 'en'
