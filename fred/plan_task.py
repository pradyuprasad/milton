#plan_task.py
from .config import config, APIKeyNotFoundError
import instructor
from openai import OpenAI
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from collections import deque
import time
from .models import *
from .search_for_single_series import find_relevant_series
from .single_series import ask_questions_about_series
from groq import Groq



try:
    client = instructor.from_openai(OpenAI(api_key=config.get_api_key("OPENAI_API_KEY")))
    groq_client = instructor.from_groq(Groq(api_key=config.get_api_key("GROQ_API_KEY")))
    FRED_API_KEY = config.get_api_key('FRED_API_KEY')
except APIKeyNotFoundError as e:
    print("API KEY Not Found!")
except Exception as e:
    print(e)


prompt = '''
You are an expert economist AI assistant specializing in the FRED (Federal Reserve Economic Data) database. Your task is to generate a Directed Acyclic Graph (DAG) of tasks to gather, analyze, and display data for answering user queries about economic topics.

Given a user query, your job is to:
1. Analyze the query to identify the main economic topics and indicators involved.
2. Generate a DAG of tasks, where each task is a node in the graph.
3. Use three types of nodes:
   a. SearchNode: for searching FRED series (query parameter required). the search node should be in natural language english, and not FRED specific terms. Add a few synonyms as well, as well as technical terms. 
   b. CodeNode: for writing analysis code (code parameter required, must depend on at least one SearchNode). No instructiong, nothing, just leave it as it is for now
   c. DisplayNode: for displaying results (display_type parameter required, must depend on atleast one CodeNode)

Each node should have:
   - A unique ID
   - A list of dependencies (IDs of other nodes it depends on)
   - A task description. In CodeNode, just give a natural language instruction of how to do the tasks using numpy, pandas only. there must be only one CodeNode. Do not give code in this, just give detailed instructions on what to do.
   - Specific parameters based on the node type

Ensure that:
1. The graph is acyclic (no circular dependencies)
2. Each task is specific and actionable
3. The overall graph, when executed, would provide a comprehensive answer to the user's query
4. Include a mix of search, code, and display nodes to fully address the query. There should be only one codeNode, after all the search has been done

Your output should be a list of DAG nodes, each with the appropriate structure based on its type.
'''


def topological_sort(dag: DAG) -> List[DAGNodeBase]:
    in_degree = {node.id: len(node.dependencies) for node in dag.nodes}
    nodes_by_id = {node.id: node for node in dag.nodes}
    queue = deque([node for node in dag.nodes if len(node.dependencies) == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        for other_node in dag.nodes:
            if node.id in other_node.dependencies:
                in_degree[other_node.id] -= 1
                if in_degree[other_node.id] == 0:
                    queue.append(other_node)
    
    if len(result) != len(dag.nodes):
        raise ValueError("The graph contains a cycle")
    
    return result

def execute_dag(dag: DAG, query:int) -> DAGWithOutput:
    sorted_nodes = topological_sort(dag)
    node_outputs: Dict[str, Any] = {}
    output_nodes: List[DAGNodeWithOutput] = []
    
    for node in sorted_nodes:
        print(f"Executing node: {node.id}")
        inputs = {dep: node_outputs[dep] for dep in node.dependencies}
        
        if node.node_type == "SearchNode":
            search_node = SearchNode(**node.model_dump())
            relevant_series = find_relevant_series(query=search_node.query, verbose=True)
            node_outputs[node.id] = relevant_series
            output_node = SearchNodeWithOutput(**node.dict(), output=relevant_series)
        
        elif node.node_type == "CodeNode":
            all_series = [s for dep_output in inputs.values() for s in dep_output if isinstance(dep_output, list) and all(isinstance(item, SeriesForSearch) for item in dep_output)]
            print(all_series)

            result = ask_questions_about_series(series_list=all_series, query=query)
            node_outputs[node.id] = result
            output_node = CodeNodeWithOutput(**node.dict(), output=result)
        
        elif node.node_type == "DisplayNode":
            node_outputs[node.id] = None 
            #output_node = DisplayNodeWithOutput(**node.dict(), output=inputs)
        
        else:
            raise ValueError(f"Unknown node type: {node.node_type}")
        
        output_nodes.append(output_node)
        print(f"Node {node.id} output: {output_node.output}")

    return DAGWithOutput(nodes=output_nodes)

def makeDAG(query:str) -> DAG:
    return client.chat.completions.create(
        response_model=DAG,
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"The user query is: {query}"}
        ]
    )

def main():
    print("Ask your question")
    query = input()
    start = time.time()

    dag: DAG = client.chat.completions.create(
        response_model=DAG,
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"The user query is: {query}"}
        ]
    )

    print("DAG structure:")
    print(dag)

    print("\nExecuting DAG:")
    dag_with_output = execute_dag(dag=dag, query=query)

    print("\nDAG with outputs:")
    #print(dag_with_output)

    print(f"\n\n\n. The time taken is {time.time() - start} seconds")

if __name__ == "__main__":
    main()