from .models import SeriesForSearch, Chart, DateValuePair
import instructor
from .config import config, APIKeyNotFoundError
from groq import Groq
import os
from openai import OpenAI
from typing import List
import json
import re
import pandas as pd
from fred.search_for_single_series import find_relevant_series
from fred.single_series import load_series_observations

try:
    FRED_API_KEY = config.get_api_key('FRED_API_KEY')
    OPENAI_API_KEY = config.get_api_key("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    instructor_client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))
    groq_client = instructor.from_groq(Groq(api_key=config.get_api_key("GROQ_API_KEY")))
except APIKeyNotFoundError as e:
    raise e

json_file_name = "LlmGenCode/test.json"

def generate_json_writer_script(series_list: List[SeriesForSearch], user_query: str) -> str:
    # Prepare the data to include in the prompt
    series_data = []
    for series in series_list:
        csv_file = f"{series.fred_id}.csv"
        df = pd.read_csv(csv_file)
        head_data = df.head().to_dict(orient='records')
        series_data.append({
            'fred_id': series.fred_id,
            'title': series.title,
            'units': series.units,
            'head_data': head_data
        })

    # Include Pydantic model definitions
    pydantic_models = """
from pydantic import BaseModel
from typing import List

class DateValuePair(BaseModel):
    date: str
    value: float

class Chart(BaseModel):
    title: str
    dataList: List[DateValuePair]
    units: str

class SeriesForSearch(BaseModel):
    fred_id: str
    title: str
    units: str
    popularity: int
    relevance_lower_better: float | None
    """

    prompt = f"""
    Use the following Pydantic model definitions:

    {pydantic_models}

    Write a Python function `write_to_json(series_list: List[SeriesForSearch], user_query: str) -> None` that does the following:
    
    1. Process each SeriesForSearch object in the series_list.
    2. Load the corresponding CSV file for each series (named {{fred_id}}.csv).
    3. Based on the user_query "{user_query}" and the original units of each series, determine if a transformation is needed:
       - If the original units are already in a growth rate format (e.g., "Percent Change from Year Ago"), use the data as is.
       - If the original units are in absolute terms (e.g., "Billions of Dollars"), calculate the year-over-year growth rate.
    4. Create a list of Chart objects using the provided Pydantic models, where:
       - Update the title to reflect any transformations made (e.g., append "- Year-over-Year Growth Rate" if transformed)
       - Adjust the units accordingly:
         * If transformed to growth rate, use "Percent Change from Year Ago"
         * If not transformed, keep the original units
       - The dataList should contain either the original data or the calculated growth rates, as appropriate
    5. Convert the list of Chart objects to JSON format.
    6. Write the JSON data to 'LlmGenCode/test.json'.
    
    Use pandas for CSV reading and data manipulation. Ensure proper error handling and commenting.
    The output JSON must be compatible with the following validation code:
    
    with open("LlmGenCode/test.json", 'r') as f:
        chartList = json.load(f)
        for i in chartList:
            chart = Chart(**i)
    
    Here is the data for each series:
    {json.dumps(series_data, indent=2)}
    
    Consider the structure and content of each series when processing the data.
    Make sure to format dates as "YYYY-MM-DD" strings in the JSON output.
    
    Provide only the Python code, enclosed in triple backticks.
    """

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a Python expert specializing in data processing and JSON manipulation."},
            {"role": "user", "content": prompt}
        ]
    )

    generated_content = response.choices[0].message.content
    print(generated_content)
    code_match = re.search(r'```python\n(.*?)```', generated_content, re.DOTALL)
    if code_match:
        return code_match.group(1)
    else:
        raise ValueError("No Python code found in the generated content")

def write_to_json(series_list: List[SeriesForSearch], user_query: str) -> None:
    # Generate the Python script
    generated_script = generate_json_writer_script(series_list, user_query)
    
    # Save the generated script to a file
    script_filename = "LlmGenCode/json_writer.py"
    with open(script_filename, "w") as f:
        f.write(generated_script)
    
    # Import and execute the generated function
    import importlib.util
    spec = importlib.util.spec_from_file_location("json_writer", script_filename)
    json_writer_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(json_writer_module)
    
    # Call the write_to_json function from the generated script
    json_writer_module.write_to_json(series_list, user_query)
    
    # Validate the generated JSON
    try:
        with open(json_file_name, 'r') as f:
            chartList = json.load(f)
            for i in chartList:
                chart = Chart(**i)
        print("JSON file successfully generated and validated.")
    except Exception as e:
        print(f"Error validating JSON: {e}")

# Usage example
if __name__ == "__main__":
    # Get the user query
    query = "What is the current inflation rate in the US"
    
    # Find relevant series based on the query
    seriesList: List[SeriesForSearch] = find_relevant_series(query=query)
    print(seriesList)
    
    # Load CSV files for each series if they don't exist
    for series in seriesList:
        output_file = f"{series.fred_id}.csv"
        if not os.path.isfile(f"./{output_file}"):
            load_series_observations(series_fred_id=series.fred_id, output_file=output_file, verbose=True)
    
    # Generate and execute the JSON writer script
    write_to_json(series_list=seriesList, user_query=query)