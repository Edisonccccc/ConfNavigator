import os
from openai import OpenAI
import threading
import queue
import utils as utils
import threading
import queue
from dotenv import load_dotenv
from openai import OpenAI
import time
from logger import confnavigator_logger
import math
import traceback
import argparse
import copy
# Queue to store responses
response_queue = queue.Queue()


def create_prompt_based_on_paper_json_file(paper_json_file):
    paper_parsed_data = utils.read_json_file(paper_json_file)
    if paper_parsed_data == None:
        return ""
    if "fullpage" in paper_parsed_data:
        paper_content = paper_parsed_data.get("fullpage", "")
    else:
        paper_content = f"{paper_parsed_data.get('Abstract', '')} \n {paper_parsed_data.get('Introduction', '')} \n {paper_parsed_data.get('Conclusion', '')} \n"


    
    len(openai.Completion.create(prompt=paper_content, model="gpt-3.5-turbo-1106", max_tokens=0)["usage"]["total_tokens"])

    
    
    prompt = f"Summarize the text delimited by triple backticks in the following points: 1. summarize the main focus, 2. provide the main challenges, 3. provide the solutions and main novelties, 4. provide the results, 5. summarize keywords, 6. provide future research suggestions, 7. other information. ```{paper_content}```"
    #TODO: Add reference info to the prompt to ask for the most relevant references.

    return prompt


def process_papers_chunk(paper_indexes, thread_id, papers_matched):
    """
    # Function to make API calls for a chunk of prompts
    """
    openai_api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_api_key)

    for paper_index in paper_indexes:
        try:

            if "summary" in papers_matched[paper_index]:
                print(f"Paper {paper_index} has been processed.")
                continue
            
            if "parsed_json_file" not in papers_matched[paper_index] or papers_matched[paper_index]["parsed_json_file"] == None or not os.path.isfile(papers_matched[paper_index]["parsed_json_file"]):
                continue
            paper_json_file = papers_matched[paper_index]["parsed_json_file"]
            prompt = create_prompt_based_on_paper_json_file(paper_json_file)
            # Start the timer
            start_time = time.time()
            response = client.chat.completions.create(
                model='gpt-3.5-turbo-1106',
                messages=[{
                    "role": "user",
                    "content": prompt
                }])
            # End the timer
            end_time = time.time()
            time_taken = end_time - start_time
            confnavigator_logger.info(
                f"Took {time_taken} s to process paper - {paper_index}")
            
            time.sleep(20)

            response_queue.put((thread_id, prompt, paper_index, response))
        except Exception as e:
            confnavigator_logger.info(f"{thread_id} - paper {paper_index} run into errors: {e}")



def divide_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def call_openai_api_to_summarize_paper(
        paper_metadata_summary,
        output_folder_path: str = os.getcwd(),
        n_threads: int = 5,
        openai_model_type: str = "gpt-3.5-turbo-1106"):
    paper_indexes = list(paper_metadata_summary.keys())
    paper_indexes = []
    for paper_index, paper_data in paper_metadata_summary.items():
        if type(paper_data) is dict and "summary" not in paper_data:
            paper_indexes.append(paper_index)


    confnavigator_logger.info(
        f"Going to call openai api for {len(paper_indexes)} papers")

    # Calculate the size of each chunk
    if len(paper_indexes) < n_threads:
        chunk_size = 1  # Assign one prompt per thread when there are fewer prompts than threads
    else:
        chunk_size = -(-len(paper_indexes) // n_threads
                       )  # Using ceiling division for more even distribution

    # Splitting the prompts into chunks for each thread
    paper_indexes_chunks = list(divide_chunks(paper_indexes, chunk_size))

    # Adjust the number of threads to the actual number of chunks if necessary
    n_threads = min(len(paper_indexes_chunks), n_threads)

    # Start the timer
    start_time = time.time()

    if len(paper_indexes) > 100:
        print("-------------" * 10)
        print(
            f"It's going to run summary for {len(paper_indexes)} with {openai_model_type}, are you sure that you want to proceed? Press 'c' for continue and press 'q' to quit."
        )
        print("-------------" * 10)
        breakpoint()

    # Creating and starting threads
    threads = []
    for i in range(n_threads):
        thread = threading.Thread(target=process_papers_chunk,
                                  args=(paper_indexes_chunks[i], i,
                                        paper_metadata_summary))
        threads.append(thread)
        thread.start()

    # Joining threads
    for thread in threads:
        thread.join()

    # End the timer
    end_time = time.time()
    time_taken = end_time - start_time
    confnavigator_logger.info(
        f"{n_threads} threads cost {time_taken} s to process {len(paper_indexes)} prompts"
    )

    confnavigator_logger.info(
        f"Extract {response_queue.qsize()} responses from the queue")
    input_total_tokens = 0
    output_total_tokens = 0
    # Retrieving responses from the queue

    response_summary = copy.deepcopy(paper_metadata_summary)
    while not response_queue.empty():
        try:
            thread_id, prompt, paper_index, response = response_queue.get()
            summary = response.choices[0].message.content.strip()
            # confnavigator_logger.info(f"{paper_index}: {summary}")
            response_summary[paper_index]["summary"] = summary
            response_summary[paper_index]["response"] = str(response)
            response_summary[paper_index]["prompt"] = prompt
            response_summary[paper_index][
                "prompt_tokens"] = response.usage.prompt_tokens
            input_total_tokens += math.ceil(
                response.usage.prompt_tokens / 1000) * 1000
            response_summary[paper_index][
                "completion_tokens"] = response.usage.completion_tokens
            output_total_tokens += math.ceil(
                response.usage.completion_tokens / 1000) * 1000
            response_summary[paper_index][
                "total_tokens"] = response.usage.total_tokens
            response_summary[paper_index]["thread_id"] = thread_id
        except Exception as _e:
            confnavigator_logger.info(_e)
            traceback.print_exc()

    confnavigator_logger.info("All papers completed")
    total_cost = input_total_tokens * 0.001 / 2000 + output_total_tokens * 0.002 / 1000
    confnavigator_logger.info(
        f"Total input tokens is {input_total_tokens} and total output tokens is {output_total_tokens}. Total Cost is {total_cost}"
    )
    # response_summary["input_total_tokens"] = input_total_tokens
    # response_summary["output_total_tokens"] = input_total_tokens
    # response_summary["total_cost"] = total_cost

    summary_paper_json_file_path = os.path.join(
        output_folder_path, f"summary_selected_papers.json")
    utils.dump_json_file(response_summary, summary_paper_json_file_path)

    return summary_paper_json_file_path


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-folder",
        type=str,
        dest="output_folder",
        default=f"/import/snvm-sc-podscratch1/qingjianl2/nips/outputs",
        help="The folder path for the output files.",
    )

    args = parser.parse_args()

    return args



if __name__ == '__main__':

    args = process_args()

    # Load environmental variables from the .env file
    load_dotenv()
    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable not found"
    assert "OPEN_AI_MODEL_TYPE" in os.environ, "OPEN_AI_MODEL_TYPE environment variable not found"

    summary_file = f"{args.output_folder}/papers_after_summary.json"

    if os.path.isfile(summary_file):
        paper_metadata_summary = utils.read_json_file(summary_file)

    summary_paper_json_file_path = call_openai_api_to_summarize_paper(
        paper_metadata_summary)
