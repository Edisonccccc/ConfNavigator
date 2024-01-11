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

# Queue to store responses
response_queue = queue.Queue()


def create_prompt_based_on_paper_json_file(paper_json_file):
    paper_parsed_data = utils.read_json_file(paper_json_file)
    paper_content = f"{paper_parsed_data.get('Abstract', '')} \n {paper_parsed_data.get('Introduction', '')} \n {paper_parsed_data.get('Conclusion', '')} \n"
    prompt = f"Summarize the text delimited by triple backticks in the following points: 1. summarize the main focus, 2. provide the main challenges, 3. provide the solutions and main novelties, 4. provide the results, 5. summarize keywords, 6. provide future research suggestions, 7. other information. ```{paper_content}```"
    #TODO: Add reference info to the prompt to ask for the most relevant references.

    return prompt


# Function to make API calls for a chunk of prompts
def process_prompts(paper_indexes, thread_id, papers_matched):
    openai_api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_api_key)

    for paper_index in paper_indexes:
        try:
            paper_json_file = papers_matched[paper_index]["json"]
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

            response_queue.put((thread_id, prompt, paper_index, response))
        except Exception as e:
            response_queue.put((thread_id, prompt, f"Error - {e}"))


def divide_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def call_openai_api_to_summarize_paper(
        parsed_summary_file,
        n_threads: int = 3,
        openai_model_type: str = "gpt-3.5-turbo-1106"):
    paper_metadata_summary = utils.read_json_file(parsed_summary_file)
    papers_matched = paper_metadata_summary["papers_matched"]
    paper_indexes = list(sorted(papers_matched.keys()))[0:10]

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

    if len(paper_indexes) > 10:
        print("-------------" * 10)
        print(
            f"It's going to run summary for {len(paper_indexes)} with {openai_model_type}, are you sure that you want to proceed? Press c for continue and press q to quit."
        )
        print("-------------" * 10)
        breakpoint()

    # Creating and starting threads
    threads = []
    for i in range(n_threads):
        thread = threading.Thread(target=process_prompts,
                                  args=(paper_indexes_chunks[i], i,
                                        papers_matched))
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

    response_summary = papers_matched
    while not response_queue.empty():
        try:
            thread_id, prompt, paper_title, response = response_queue.get()
            summary = response.choices[0].message.content.strip()
            # confnavigator_logger.info(f"{paper_title}: {summary}")
            response_summary[paper_title]["summary"] = summary
            response_summary[paper_title]["response"] = str(response)
            response_summary[paper_title]["prompt"] = prompt
            response_summary[paper_title][
                "prompt_tokens"] = response.usage.prompt_tokens
            input_total_tokens += math.ceil(
                response.usage.prompt_tokens / 1000) * 1000
            response_summary[paper_title][
                "completion_tokens"] = response.usage.completion_tokens
            output_total_tokens += math.ceil(
                response.usage.completion_tokens / 1000) * 1000
            response_summary[paper_title][
                "total_tokens"] = response.usage.total_tokens
            response_summary[paper_title]["thread_id"] = thread_id
        except Exception as _e:
            confnavigator_logger.info(_e)
            traceback.print_exc()
            confnavigator_logger.info(f"Failed to parse file {paper_title}")

    confnavigator_logger.info("All threads completed")

    total_cost = input_total_tokens * 0.001 / 2000 + output_total_tokens * 0.002 / 1000

    confnavigator_logger.info(
        f"Total input tokens is {input_total_tokens} and total output tokens is {output_total_tokens}. Total Cost is {total_cost}"
    )
    response_summary["input_total_tokens"] = input_total_tokens
    response_summary["output_total_tokens"] = input_total_tokens
    response_summary["total_cost"] = total_cost

    output_folder_path = os.path.dirname(parsed_summary_file)
    summary_paper_json_file_path = os.path.join(
        output_folder_path, f"Summary_selected_papers.json")
    utils.dump_json_file(response_summary, summary_paper_json_file_path)

    return summary_paper_json_file_path


if __name__ == '__main__':
    # Load environmental variables from the .env file
    load_dotenv()
    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable not found"
    assert "OPEN_AI_MODEL_TYPE" in os.environ, "OPEN_AI_MODEL_TYPE environment variable not found"

    parsed_summary_file = f"{os.getcwd()}/paper_metadata_summary.json"
    summary_paper_json_file_path = call_openai_api_to_summarize_paper(
        parsed_summary_file)
