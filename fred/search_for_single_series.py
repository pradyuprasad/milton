# search_for_single_series.py
import chromadb
import os
from typing import List, Dict, Set
import instructor
from .config import config, APIKeyNotFoundError
from openai import OpenAI
from groq import Groq
from .models import SeriesForSearch, Keywords, ClassifiedSeries
from .database import Database, DatabaseConnectionError
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

print("Imported libraries")

# Initialize Chroma client with persistent storage
chroma_persist_directory = os.path.join(os.path.dirname(__file__), "chroma_db")
chroma_client = chromadb.PersistentClient(path=chroma_persist_directory)
    
# Get the collection
collection = chroma_client.get_collection("fred-economic-series")
#tags_collection = chroma_client.get_collection("fred-tags")


try:
    FRED_API_KEY = config.get_api_key('FRED_API_KEY')
    OPENAI_API_KEY = config.get_api_key("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    client = Groq(api_key=config.get_api_key("GROQ_API_KEY"))
    instructor_client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))
    instructor_groq_client = instructor.from_groq(Groq(api_key=config.get_api_key("GROQ_API_KEY")))
except APIKeyNotFoundError as e:
    raise e



def extract_keyword(user_query: str) -> Keywords:
    return instructor_client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Keywords,
        messages=[
            {"role": "system", "content": """
You are an expert in economic terminology and the FRED (Federal Reserve Economic Data) database. Your task is to convert user queries into the most appropriate search terms for the FRED database.
            """},
            {"role": "user", "content": f"""
Convert the following query into search terms for the FRED database:
"{user_query}"
Follow these rules strictly:
1. Provide only full terms as they would appear in the FRED database. For example, use "Gross Domestic Product" instead of "GDP".
2. Expand all common economic acronyms to their full forms.
3. Focus on specific economic indicators, measurements, or concepts.
4. Provide at most 3 key terms, prioritizing specificity and relevance to the FRED database. 
5. Do not include general words like "current", "rate" unless they are part of a specific economic term.
6. If the query mentions a specific country, include the full country name as part of the relevant economic term(s).
Respond with only the list of search terms, nothing else.
7. Be strict in your question. Ensure that the geographic region and all other data match what the user asked
            """}
        ]
    )

def keyword_semantic_search(keywords: List[str], n_results: int = 5, verbose:bool = False) -> Set[SeriesForSearch]:

    
    results = []
    ''' # TODO: Decide if this is needed or not
    for keyword in keywords:
        # Assuming collection.query is defined and functional
        keyword_results = collection.query(
            query_texts=[keyword],
            n_results=n_results
        )
        if verbose:
            print("the keyword is", keyword, "and the results are", keyword_results)
        results.append(keyword_results)
    '''
    
    query = ' '.join(keywords)

    keyword_results = collection.query(query_texts=[query], n_results=n_results)
    if verbose:
        print("the query is", query, "and the results are", keyword_results)
    results.append(keyword_results)


    series_set = set()
    
    for keyword_results in results:
        for i in range(len(keyword_results['ids'][0])):
            series = SeriesForSearch(
                fred_id=keyword_results['ids'][0][i],
                title=keyword_results['metadatas'][0][i]['title'],
                units=keyword_results['metadatas'][0][i]['units'],
                popularity=keyword_results['metadatas'][0][i]['popularity'],
                relevance_lower_better=keyword_results['distances'][0][i]
            )
            series_set.add(series)
    
    return series_set





def print_series_list(series_list:List[SeriesForSearch]) -> None:
    series_list = list(filter(lambda x: not x.relevance_lower_better or x.relevance_lower_better <= 1, series_list))
    series_list.sort(key=lambda x: x.relevance_lower_better)
    for series in series_list:
        print("id:", series.fred_id)
        print("title:", series.title)
        print("popularity:", series.popularity)
        print("units:", series.units)
        print("relevance:", series.relevance_lower_better)
        print("\n\n")
        print("------------------------")

def rank_relevant_outputs(series_list:List[SeriesForSearch], query:str) -> SeriesForSearch:
    instructor
    try:
    
        return instructor_groq_client.chat.completions.create(response_model=ClassifiedSeries, messages=[
            {"role": "system", "content": "You will be given a number of economic series from the user. Your job is to mark them as relevant or irrelevant for a given query and output it in the given format"}, {
                "role": "user", "content":f"The user's query is {query}. The possible datasets are {series_list}"
            }
        ], model="llama3-70b-8192")
    except Exception:
        return rank_relevant_outputs(series_list=series_list, query=query) 

def keyword_text_search(keywords: List[str]) -> List[SeriesForSearch]:
    results_set = set()
    cursor = Database.get_cursor()
    for keyword in keywords:
    # Use the LIKE command to search for each keyword separately
        cursor.execute("""
            SELECT fred_id, title, units, popularity
            FROM series
            WHERE title LIKE ?
        """, (f'%{keyword}%',))
        
        # Fetch all matching rows
        rows = cursor.fetchall()
        
        # Add each row to the results set to avoid duplicates
        for row in rows:
            results_set.add(SeriesForSearch(
                fred_id=row[0],
                title=row[1],
                units=row[2],
                popularity=row[3]
            ))

# Close the database connection
    
    
    # Return the results as a list
    return list(results_set)

def find_relevant_series(query:str, verbose:bool = False) -> List[SeriesForSearch]:
    keyword_list = extract_keyword(query)
    if verbose:
        print("Extracted keywords:", keyword_list.word)
    semantic_search_results = keyword_semantic_search(keyword_list.word, n_results=5)
    possible: ClassifiedSeries = rank_relevant_outputs(list(semantic_search_results), query=query)

    if verbose:
        print("the relevant series are", possible.relevant)
        print("\n\n")
        print("the not relevant series are", possible.notRelevant)
    return possible.relevant


if __name__ == "__main__":
    firstTime = True
    while True:
        if not firstTime:
            print("\n\n")
        print("What is your query?")
        query = input()
        print(f"\nQuery: {query}")
        find_relevant_series(query=query, verbose=True)
        firstTime = False