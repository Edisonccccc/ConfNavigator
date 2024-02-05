import os
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
from utils import PAPER_PARSING_INFO_FILE, GPT_SUMMARY_LIBRARY_FILE, PAPER_SUMMARY_INFO_FILE
import tiktoken
# Queue to store responses
response_queue = queue.Queue()


def num_tokens_from_string(string: str,
                           encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def create_prompt_based_on_paper_json_file(paper_json_file,
                                           ratio: float = 1.0):
    paper_parsed_data = utils.read_json_file(paper_json_file)
    if paper_parsed_data == None:
        return ""
    if "fullpage" in paper_parsed_data:
        paper_content = paper_parsed_data.get("fullpage", "")
    else:
        paper_content = f"{paper_parsed_data.get('Abstract', '')} \n {paper_parsed_data.get('Introduction', '')} \n {paper_parsed_data.get('Conclusion', '')} \n"

    if ratio < 1.0:
        # Truncate the message if necessary
        truncate_index = int(len(paper_content) * ratio)

        paper_content = paper_content[0:truncate_index]
    elif ratio > 1:
        breakpoint()

    prompt = f"Summarize the text delimited by triple backticks in the following points: 1. summarize the main focus, 2. provide the main challenges, 3. provide the solutions and main novelties, 4. provide the results, 5. summarize keywords, 6. provide future research suggestions, 7. other information. ```{paper_content}```"
    #TODO: Add reference info to the prompt to ask for the most relevant references.

    return prompt


def process_papers_chunk(papers_chunk, thread_id):
    """
    # Function to make API calls for a chunk of prompts
    """
    openai_api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_api_key)

    for paper_data in papers_chunk:
        try:
            if "summary" in paper_data:
                print(f"Paper {paper_data['id']} has been processed.")
                continue

            if "parsed_json_file" not in paper_data or paper_data[
                    "parsed_json_file"] == None or not os.path.isfile(
                        paper_data["parsed_json_file"]):
                continue
            paper_json_file = paper_data["parsed_json_file"]
            prompt = create_prompt_based_on_paper_json_file(paper_json_file)

            number_tokens = num_tokens_from_string(prompt)
            print(f"Original number of tokens {number_tokens}")
            if number_tokens >= 16385:
                ratio = float(16385) / (number_tokens + 200) * 0.8
                prompt = create_prompt_based_on_paper_json_file(
                    paper_json_file, ratio)
            else:
                ratio = 1.0

            number_tokens = num_tokens_from_string(prompt)
            print(
                f"Original number of tokens {number_tokens} with ratio {ratio}"
            )

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
                f"Took {time_taken} s to process paper - {paper_data['id']}")

            # time.sleep(40)

            response_queue.put((thread_id, prompt, paper_data['id'], response))
        except Exception as e:
            confnavigator_logger.info(
                f"{thread_id} - paper {paper_data['id']} run into errors: {e}")


def divide_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Function to continuously process responses from the queue
def process_queue(summarized_papers_to_record, gpt_summary_tmp_file):
    while True:
        if not response_queue.empty():
            try:
                thread_id, prompt, paper_index, response = response_queue.get()
                if thread_id == -1:
                    break
                paper_index = str(paper_index)
                summary = response.choices[0].message.content.strip()
                summarized_papers_to_record[paper_index]["summary"] = summary
                summarized_papers_to_record[paper_index]["response"] = str(
                    response)
                summarized_papers_to_record[paper_index]["prompt"] = prompt
                summarized_papers_to_record[paper_index][
                    "prompt_tokens"] = response.usage.prompt_tokens
                summarized_papers_to_record[paper_index][
                    "completion_tokens"] = response.usage.completion_tokens
                summarized_papers_to_record[paper_index][
                    "total_tokens"] = response.usage.total_tokens
                summarized_papers_to_record[paper_index][
                    "thread_id"] = thread_id
                utils.dump_json_file(summarized_papers_to_record,
                                     gpt_summary_tmp_file)
            except Exception as _e:
                confnavigator_logger.info(_e)
                traceback.print_exc()
                # You can also add logic to break the loop if certain conditions are met
                # For example, if a specific 'end' signal is put in the queue by main thread


def call_openai_api_to_summarize_paper(
        paper_parsing_info,
        gpt_summary_library_file,
        n_threads: int = 5,
        openai_model_type: str = "gpt-3.5-turbo-1106"):

    if os.path.isfile(gpt_summary_library_file):
        gpt_summary_info = utils.read_json_file(gpt_summary_library_file)
    else:
        gpt_summary_info = {}

    paper_indexes_total = list(paper_parsing_info.keys())
    papers_to_summarize = []
    summarized_papers_to_record = {}

    skip_paper_index = ['470', '1450']

    for paper_index, paper_data in paper_parsing_info.items():
        if type(
                paper_data
        ) is dict and paper_index not in gpt_summary_info and paper_index not in skip_paper_index:
            papers_to_summarize.append(paper_data)
            summarized_papers_to_record[paper_index] = paper_data

    papers_to_summarize = []
    processing_papers_number = len(papers_to_summarize)
    confnavigator_logger.info(
        f"Going to call openai api for {processing_papers_number} papers")

    # Calculate the size of each chunk
    if processing_papers_number < n_threads:
        chunk_size = 1  # Assign one prompt per thread when there are fewer prompts than threads
    else:
        chunk_size = -(-processing_papers_number // n_threads
                       )  # Using ceiling division for more even distribution

    # Splitting the prompts into chunks for each thread
    paper_indexes_chunks = list(divide_chunks(papers_to_summarize, chunk_size))

    # Adjust the number of threads to the actual number of chunks if necessary
    n_threads = min(len(paper_indexes_chunks), n_threads)

    # Start the timer
    start_time = time.time()

    if processing_papers_number > 100:
        print("-------------" * 10)
        print(
            f"It's going to run summary for {processing_papers_number} with {openai_model_type}, are you sure that you want to proceed? Press 'c' for continue and press 'q' to quit."
        )
        print("-------------" * 10)
        breakpoint()

    # Creating and starting threads
    threads = []
    for i in range(n_threads):
        thread = threading.Thread(target=process_papers_chunk,
                                  args=(paper_indexes_chunks[i], i))
        threads.append(thread)
        thread.start()

    # Starting the queue processing thread
    gpt_summary_tmp_file = os.path.join(os.getcwd(), "gpt_summary_tmp.json")
    queue_thread = threading.Thread(target=process_queue,
                                    args=(summarized_papers_to_record,
                                          gpt_summary_tmp_file))
    queue_thread.start()

    # Joining threads
    for thread in threads:
        thread.join()

    # All threads are done, put the end marker in the queue
    response_queue.put((-1, "", -1, None))

    # Optionally join the queue processing thread (if you want to wait for it to finish)
    queue_thread.join()

    # End the timer
    end_time = time.time()
    time_taken = end_time - start_time
    confnavigator_logger.info(
        f"{n_threads} threads cost {time_taken} s to process {processing_papers_number} prompts"
    )

    summarized_papers_to_record = utils.read_json_file(gpt_summary_tmp_file)
    input_total_tokens = 0
    output_total_tokens = 0

    for paper_index, paper_data in summarized_papers_to_record.items():
        if paper_index not in gpt_summary_info and "summary" in paper_data:
            input_total_tokens += math.ceil(
                paper_data["prompt_tokens"] / 1000) * 1000
            output_total_tokens += math.ceil(
                paper_data["completion_tokens"] / 1000) * 1000

            gpt_summary_info[paper_index] = paper_data

    confnavigator_logger.info(
        f"Summarized {len(summarized_papers_to_record)} papers.")
    total_cost = input_total_tokens * 0.001 / 2000 + output_total_tokens * 0.002 / 1000
    confnavigator_logger.info(
        f"Total input tokens is {input_total_tokens} and total output tokens is {output_total_tokens}. Total Cost is {total_cost}"
    )

    utils.dump_json_file(gpt_summary_info, gpt_summary_library_file)

    return gpt_summary_library_file


def integrate_gpt_summary(paper_parsing_info_file, gpt_summary_library_file,
                          paper_summary_info_file):
    paper_parsing_info = utils.read_json_file(paper_parsing_info_file)
    gpt_summary_info = utils.read_json_file(gpt_summary_library_file)

    for paper_index, paper_info in paper_parsing_info.items():
        if "summary" in paper_parsing_info or paper_index not in gpt_summary_info:
            continue
        paper_info["summary"] = gpt_summary_info[paper_index]["summary"]
        paper_info["prompt_tokens"] = gpt_summary_info[paper_index][
            "prompt_tokens"]
        paper_info["completion_tokens"] = gpt_summary_info[paper_index][
            "completion_tokens"]

    utils.dump_json_file(paper_parsing_info, paper_summary_info_file)


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-folder",
        type=str,
        dest="output_folder",
        default=f"***",
        help="The folder path for the output files.",
    )

    args = parser.parse_args()

    return args


def paper_gpt_summarizing(output_folder):
    gpt_summary_library_file = f"{output_folder}/{GPT_SUMMARY_LIBRARY_FILE}"
    paper_parsing_info_file = f"{output_folder}/{PAPER_PARSING_INFO_FILE}"
    paper_summary_info_file = f"{output_folder}/{PAPER_SUMMARY_INFO_FILE}"

    assert os.path.isfile(
        paper_parsing_info_file
    ) == True, f"The provided {paper_parsing_info_file} could not be found!"
    paper_parsing_info = utils.read_json_file(paper_parsing_info_file)

    gpt_summary_library_file = call_openai_api_to_summarize_paper(
        paper_parsing_info=paper_parsing_info,
        gpt_summary_library_file=gpt_summary_library_file)

    # Combine the summary info from gpt_summary_library.json into paper summary info.

    paper_summary_info_file = integrate_gpt_summary(
        paper_parsing_info_file=paper_parsing_info_file,
        gpt_summary_library_file=gpt_summary_library_file,
        paper_summary_info_file=paper_summary_info_file)


if __name__ == '__main__':

    args = process_args()

    # Load environmental variables from the .env file
    load_dotenv()

    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable not found"
    assert "OPEN_AI_MODEL_TYPE" in os.environ, "OPEN_AI_MODEL_TYPE environment variable not found"

    paper_gpt_summarizing(args.output_folder)
