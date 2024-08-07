# single_series.py
import os
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from .config import config, APIKeyNotFoundError
from .database_ops import get_units, check_series_exists
from .utils import store_series_in_DB
from openai import OpenAI
from groq import Groq
import subprocess
import time
from .models import SeriesForSearch, InstructionsList, CodeBlock
from .search_for_single_series import find_relevant_series 
import re
import instructor



try:
    FRED_API_KEY = config.get_api_key('FRED_API_KEY')
    OPENAI_API_KEY = config.get_api_key("OPENAI_API_KEY")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    instructor_client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))
    groq_client = instructor.from_groq(Groq(api_key=config.get_api_key("GROQ_API_KEY")))
except APIKeyNotFoundError as e:
    raise e


def call_groq(messages: List[Dict[str, str]], model:str="llama3-70b-8192"):
    return client.chat.completions.create(messages=messages, model=model, temperature = 0.0)


def remove_all_extras(code:str):
    messages = [
        {
            "role": "system",
            "content": '''Your job is to remove EVERYTHING THAT IS NOT CODE. REMOVE EVERYTHING THAT IS NOT CODE. IF IT IS CODE IT STAYS. SOMETIMES THE CODE HAS THINGS LIKE "here is your code" or some other useless prefix. I need you to remove it. Just give me the code that is given to you. If only code is given to you, leave it untouched. Everything you put will be a python file. So don't output anything except code. '''
        },

        {
            "role": "user",
            "content": f"The input is {code}"
        }
    ]
    output = call_groq(messages, "llama3-8b-8192").choices[0].message.content
    return re.sub(r'^```|```$', '', output, flags=re.MULTILINE)
    

def fix_code(instructions:str, old_code:str, error:str) -> str:
    prompt =f''' The old code is {old_code}. The original instructions were {instructions}. The error in the old code is {error}. Write what is wrong with the code in the thoughts section, and then put code and code only in the code section. 
'''
    codeblock: CodeBlock = instructor_client.chat.completions.create(response_model=CodeBlock, messages=[
        {"role": "user", "content": prompt}
    ], model="gpt-4o")

    print(codeblock)

    code = (remove_all_extras(codeblock.code))
    print(code)
    return code

def load_series_observations(series_fred_id: str, output_file: str, verbose:bool = True) -> bool:
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_fred_id}&api_key={FRED_API_KEY}&file_type=json"
    response = requests.get(url)
    response.raise_for_status()

    ans = response.json()
    observations = ans['observations']
    observations_data = []

    for obs in observations:
        if 'date' in obs and 'value' in obs:
            try:
                if (obs['value']) != '.':
                    value = float(obs['value'])
                    observations_data.append({
                        'date': datetime.strptime(obs['date'], '%Y-%m-%d').date(),
                        'value': value
                    })
            except ValueError:
                print(f"Error converting value to float for {series_fred_id}: date = {obs['date']}, value = {obs['value']}")

    if verbose:
        print("got data for", series_fred_id)
        if observations_data:
            print(observations_data[0])
        else:
            print("No valid observations found")

    df = pd.DataFrame(observations_data)
    df.to_csv(output_file, index=False)
    print("saved csv", output_file)
    return True    
    
def run_code(file_name:str, code:str) -> Tuple[int, Optional[str]]:
    with open(file_name, 'w') as file:
        file.write(code)

    script_output = subprocess.run(["python3", file_name], capture_output=True, text=True)
    works = script_output.returncode == 0
    return (script_output.returncode, script_output.stdout if works else script_output.stderr)

        

def ask_questions_about_series(series_list: List[SeriesForSearch], query:str):
    try:
        output_file_list = []
        head_list = []
        for series in series_list:
            fred_id = series.fred_id
            series_exists = check_series_exists(fred_id)
            output_file = f"{fred_id}.csv" 
            if not series_exists:
                store_series_in_DB(series_fred_id=fred_id)
            else:
                print("series exists")
            string_to_append = f"Series name: {series.title}: csv_path:{output_file}. Units:{series.units}"
            output_file_list.append(string_to_append)
            units = get_units(fred_id=fred_id)
            if not units:
                raise ValueError("unable to get units")
            
            if not os.path.isfile(f"./{output_file}"):
                load_series_observations(series_fred_id=fred_id, output_file=output_file, verbose=True)

            df = pd.read_csv(output_file)
            head = f"{output_file}: " + str(df.head())
            head_list.append(head)

         
        prompt = f'''

Write me python code which answer's the user's question. Give as many details in the print statement as possible, all of these are going to be given to the highly-data curious user. The code should print that and that only. If you have more than one dataset think hard on which one would be more relevant and timely (higher frequency). Do not plot anything, do not graph anything. You have these CSV files csv file - {output_file_list}. It has the schema, {head_list}. The date is in YYYY-MM-DD format, and the data is in float.
'''
        instructions_prompt = f'''

You have these CSV files csv file - {output_file_list}. It has the schema, {head_list}. The date is in YYYY-MM-DD format, and the data is in float. Write me instructions to pass on to an AI agent to write code which answer's the user's question. Give as many details in the print statement as possible, all of these are going to be given to the highly-data curious user. The instructions to instruct the AI agent to print that and that only. If you have more than one dataset think hard on which one would be more relevant and timely (higher frequency). Do not plot anything, do not graph anything. Give highly detailed instructions in a setp by step format. You should print it in the format
1. Step 1
2. Step 2

These instructions should be very detailed and express to the AI code agent every step of what to do. The AI aigent should print one or two lines explaning the data and summarizing it. Print relevant details only, and do not print more than 5 lines at once. Do not ever print data out. Give an example of what to print so it is clear to the agent
'''
        print(instructions_prompt)
        
        instructions = groq_client.chat.completions.create(response_model=InstructionsList, model="llama3-70b-8192", messages=[
            {"role": "system", "content": instructions_prompt}, 
            {"role": "user", "content": f"the user query is {query}"}
        ])

        print(instructions)
        completion:CodeBlock = instructor_client.chat.completions.create(response_model=CodeBlock, messages=[
            {"role": "system", "content": prompt}, 
            {"role": "user", "content": f"The instructions are {str(instructions)}"}
        ], model="gpt-4o")

        print(completion)

        code = (remove_all_extras(completion.code))
        num_times = 1
        print("got code!")
        return_code, output =  run_code("LLMGenCode/test.py", code=code)
        while num_times <= 5:
            if return_code != 0:
                code = fix_code(instructions=instructions, old_code=code, error=output)
                run_code("LLMGenCode/test.py", code=code) 
                num_times+= 1
            else:
                return output
        
        raise Exception("Too many retries")

    except Exception as e:
        print(e)
        raise e
        return None



if __name__ == "__main__":
    firstTime = True
    print("What is your query?")
    query = input()
    start = time.time()
    print("started at", start)
    series_list: List[SeriesForSearch] = find_relevant_series(query=query, verbose=True)
    end_time = time.time()
    print("took", end_time-start, "time for this to happen")
    print(series_list)
    
    
    ask_questions_about_series(series_list, query)

