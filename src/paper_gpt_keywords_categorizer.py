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
from utils import PAPER_SUMMARY_INFO_FILE, GPT_KEYWORDS_LIBRARY_FILE, PAPER_KEYWORDS_INFO_FILE
import tiktoken

# Queue to store responses
category_response_queue = queue.Queue()

# Categorize each abstract and update the JSON data
keywords_lib = 'multimodal, diffusion model, language model, reinforcement learning, generative AI, computer vision, graph neural network, adversarial attacks and model poisoning, knowledge distillation and memory reduction schemes, federated learning, dataset or benchmark, differential privacy, in-context learning, model robustness, theory or algorithms, mixture of experts, multi-agent'
keywords_lib_set = set([item.strip() for item in keywords_lib.split(',')])

num_repeat_gen = 1


def num_tokens_from_string(string: str,
                           encoding_name: str = "cl100k_base") -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def create_categorize_prompt_based_on_abstract(paper_data, ratio: float = 1.0):

    abstract_content = f"{paper_data.get('abstract', '')} \n"

    if ratio < 1.0:
        # Truncate the message if necessary
        truncate_index = int(len(abstract_content) * ratio)

        abstract_content = abstract_content[0:truncate_index]
    elif ratio > 1:
        breakpoint()

    prompt = f"Carefully read the research abstract enclosed within triple backticks. Based on its content, identify and report up to three keywords that best categorize the abstract. It is crucial that your selections come exclusively from the following list. Do not introduce or infer keywords outside of this list. The keywords to choose from are: {keywords_lib}.  List your chosen keywords separated by commas, without numbering them. \n\n ```{abstract_content}```"
    #TODO: Add reference info to the prompt to ask for the most relevant references.

    return prompt


def process_papers_chunk(papers_chunk, thread_id, openai_model_type):
    """
    # Function to make API calls for a chunk of prompts
    """
    openai_api_key = os.environ["OPENAI_API_KEY"]
    client = OpenAI(api_key=openai_api_key)

    for paper_data in papers_chunk:
        try:

            prompt = create_categorize_prompt_based_on_abstract(paper_data)

            number_tokens = num_tokens_from_string(prompt)
            print(f"Original number of tokens {number_tokens}")
            if number_tokens >= 16385:
                ratio = float(16385) / (number_tokens + 200) * 0.8
                prompt = create_categorize_prompt_based_on_abstract(
                    paper_data, ratio)
            else:
                ratio = 1.0

            number_tokens = num_tokens_from_string(prompt)
            print(
                f"Original number of tokens {number_tokens} with ratio {ratio}"
            )

            # Start the timer
            start_time = time.time()

            filtered_keywords_set = None
            counter_repeat_gen = 0
            data = {}
            while (counter_repeat_gen < 3 and not filtered_keywords_set):
                response = client.chat.completions.create(
                    model=openai_model_type,
                    messages=[{
                        "role": "user",
                        "content": prompt
                    }],
                    temperature=0.1,
                    top_p=0.2)

                categories = response.choices[0].message.content.strip()
                cleaned_categories = ', '.join(
                    [word for word in categories.split(';') if word.strip()])
                print('*** Paper ID= ', paper_data['id'], ' | ',
                      f"Model = {openai_model_type} | ", cleaned_categories)
                categories_set = set(
                    [item.strip() for item in cleaned_categories.split(',')])
                filtered_keywords_set = categories_set & keywords_lib_set
                counter_repeat_gen += 1

            data['Keywords_GPT3.5'] = cleaned_categories
            if counter_repeat_gen < 3:
                filtered_keywords_str = ', '.join(
                    str(item) for item in list(filtered_keywords_set))
                data['Keywords_GPT3.5_filtered'] = filtered_keywords_str
            else:
                data['Keywords_GPT3.5_filtered'] = 'UNKNOWN'

            # End the timer
            end_time = time.time()
            time_taken = end_time - start_time
            confnavigator_logger.info(
                f"Took {time_taken} s to process paper - {paper_data['id']}")

            time.sleep(5)

            category_response_queue.put(
                (thread_id, prompt, paper_data['id'], response, data))
        except Exception as e:
            confnavigator_logger.info(
                f"{thread_id} - paper {paper_data['id']} run into errors: {e}")


def divide_chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# Function to continuously process responses from the queue
def process_queue(categorized_papers_to_record, gpt_keywords_tmp_file):
    while True:
        if not category_response_queue.empty():
            try:
                thread_id, prompt, paper_index, response, data = category_response_queue.get(
                )
                if thread_id == -1:
                    break
                paper_index = str(paper_index)

                categorized_papers_to_record[paper_index][
                    "Keywords_GPT3.5"] = data['Keywords_GPT3.5']
                categorized_papers_to_record[paper_index][
                    "Keywords_GPT3.5_filtered"] = data[
                        'Keywords_GPT3.5_filtered']

                categorized_papers_to_record[paper_index][
                    "keywords_response"] = str(response)
                categorized_papers_to_record[paper_index][
                    "keywords_prompt"] = prompt
                categorized_papers_to_record[paper_index][
                    "keywords_prompt_tokens"] = response.usage.prompt_tokens
                categorized_papers_to_record[paper_index][
                    "keywords_completion_tokens"] = response.usage.completion_tokens
                categorized_papers_to_record[paper_index][
                    "keywords_total_tokens"] = response.usage.total_tokens
                categorized_papers_to_record[paper_index][
                    "keywords_thread_id"] = thread_id
                utils.dump_json_file(categorized_papers_to_record,
                                     gpt_keywords_tmp_file)
            except Exception as _e:
                confnavigator_logger.info(_e)
                traceback.print_exc()


def call_openai_api_to_categorize_paper(
        paper_summary_info,
        gpt_keywords_library_file,
        n_threads: int = 5,
        openai_model_type: str = "gpt-3.5-turbo-1106"):

    if os.path.isfile(gpt_keywords_library_file):
        gpt_keywords_info = utils.read_json_file(gpt_keywords_library_file)
    else:
        gpt_keywords_info = {}

    papers_to_categorize = []
    categorized_papers_to_record_tmp = {
    }  # Pass into the thread to use the info

    for paper_index, paper_data in paper_summary_info.items():
        if type(paper_data) is not dict:
            continue
        if paper_index not in gpt_keywords_info:
            papers_to_categorize.append(paper_data)
            categorized_papers_to_record_tmp[paper_index] = paper_data
        elif gpt_keywords_info[paper_index].get("Keywords_GPT3.5_filtered",
                                                "") in ["", "UNKNOWN"]:
            papers_to_categorize.append(paper_data)
            categorized_papers_to_record_tmp[paper_index] = paper_data

    papers_to_categorize = papers_to_categorize[0:5]
    categorized_papers_to_record = {}
    for paper_data in papers_to_categorize:
        categorized_papers_to_record[str(
            paper_data["id"])] = categorized_papers_to_record_tmp[str(
                paper_data["id"])]

    processing_papers_number = len(papers_to_categorize)
    confnavigator_logger.info(
        f"Going to call openai api for {processing_papers_number} papers")

    # Calculate the size of each chunk
    if processing_papers_number < n_threads:
        chunk_size = 1  # Assign one prompt per thread when there are fewer prompts than threads
    else:
        chunk_size = -(-processing_papers_number // n_threads
                       )  # Using ceiling division for more even distribution

    # Splitting the prompts into chunks for each thread
    paper_indexes_chunks = list(divide_chunks(papers_to_categorize,
                                              chunk_size))

    # Adjust the number of threads to the actual number of chunks if necessary
    n_threads = min(len(paper_indexes_chunks), n_threads)

    # Start the timer
    start_time = time.time()

    if processing_papers_number > 100:
        print("-------------" * 10)
        print(
            f"It's going to run keywords for {processing_papers_number} with {openai_model_type}, are you sure that you want to proceed? Press 'c' for continue and press 'q' to quit."
        )
        print("-------------" * 10)
        breakpoint()

    # Creating and starting threads
    threads = []
    for i in range(n_threads):
        thread = threading.Thread(target=process_papers_chunk,
                                  args=(paper_indexes_chunks[i], i,
                                        openai_model_type))
        threads.append(thread)
        thread.start()

    # Starting the queue processing thread
    gpt_keywords_tmp_file = os.path.join(os.getcwd(), "gpt_keywords_tmp.json")
    queue_thread = threading.Thread(target=process_queue,
                                    args=(categorized_papers_to_record,
                                          gpt_keywords_tmp_file))
    queue_thread.start()

    # Joining threads
    for thread in threads:
        thread.join()

    # All threads are done, put the end marker in the queue
    category_response_queue.put((-1, "", -1, None, {}))

    # Optionally join the queue processing thread (if you want to wait for it to finish)
    queue_thread.join()

    # End the timer
    end_time = time.time()
    time_taken = end_time - start_time
    confnavigator_logger.info(
        f"{n_threads} threads cost {time_taken} s to process {processing_papers_number} prompts"
    )

    categorized_papers_to_record = utils.read_json_file(gpt_keywords_tmp_file)
    input_total_tokens = 0
    output_total_tokens = 0
    keywords_paper = 0

    for paper_index, paper_data in categorized_papers_to_record.items():

        input_total_tokens += math.ceil(
            paper_data.get("keywords_prompt_tokens", 0) / 1000) * 1000
        output_total_tokens += math.ceil(
            paper_data.get("keywords_completion_tokens", 0) / 1000) * 1000

        if paper_index not in gpt_keywords_info:
            gpt_keywords_info[paper_index] = paper_data
            keywords_paper += 1

        elif "Keywords_GPT3.5" not in gpt_keywords_info[paper_index]:
            gpt_keywords_info[paper_index] = paper_data
            keywords_paper += 1

        elif "Keywords_GPT3.5_filtered" in gpt_keywords_info[
                paper_index] and gpt_keywords_info[paper_index][
                    "Keywords_GPT3.5_filtered"] == "UNKNOWN" and paper_data.get(
                        "Keywords_GPT3.5_filtered", "") != "UNKNOWN":
            gpt_keywords_info[paper_index] = paper_data

            keywords_paper += 1

    confnavigator_logger.info(
        f"categorized {len(categorized_papers_to_record)} papers.")
    confnavigator_logger.info(f"categorized {keywords_paper} new papers.")

    total_cost = input_total_tokens * 0.001 / 2000 + output_total_tokens * 0.002 / 1000
    confnavigator_logger.info(
        f"Total input tokens is {input_total_tokens} and total output tokens is {output_total_tokens}. Total Cost is {total_cost}"
    )

    utils.dump_json_file(gpt_keywords_info, gpt_keywords_library_file)

    return gpt_keywords_library_file


def integrate_gpt_keywords(paper_summary_info_file, gpt_keywords_library_file,
                           paper_keywords_info_file):
    paper_summary_info = utils.read_json_file(paper_summary_info_file)
    gpt_keywords_info = utils.read_json_file(gpt_keywords_library_file)

    for paper_index, paper_info in paper_summary_info.items():
        if "Keywords_GPT3.5_filtered" in paper_info and paper_info[
                "Keywords_GPT3.5_filtered"] != "UNKNOWN" or paper_index not in gpt_keywords_info:
            continue

        if "Keywords_GPT3.5" not in gpt_keywords_info[paper_index]:
            continue
        paper_info["Keywords_GPT3.5"] = gpt_keywords_info[paper_index][
            "Keywords_GPT3.5"]
        paper_info["Keywords_GPT3.5_filtered"] = gpt_keywords_info[
            paper_index]["Keywords_GPT3.5_filtered"]
        paper_info["keywords_prompt_tokens"] = gpt_keywords_info[paper_index][
            "keywords_prompt_tokens"]
        paper_info["keywords_completion_tokens"] = gpt_keywords_info[
            paper_index]["keywords_completion_tokens"]

    utils.dump_json_file(paper_summary_info, paper_keywords_info_file)


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


def paper_gpt_categorizing(output_folder):

    gpt_keywords_library_file = f"{output_folder}/{GPT_KEYWORDS_LIBRARY_FILE}"
    paper_summary_info_file = f"{output_folder}/{PAPER_SUMMARY_INFO_FILE}"
    paper_keywords_info_file = f"{output_folder}/{PAPER_KEYWORDS_INFO_FILE}"

    assert os.path.isfile(
        paper_summary_info_file
    ) == True, f"The provided {paper_summary_info_file} could not be found!"
    paper_summary_info = utils.read_json_file(paper_summary_info_file)

    gpt_keywords_library_file = call_openai_api_to_categorize_paper(
        paper_summary_info=paper_summary_info,
        gpt_keywords_library_file=gpt_keywords_library_file,
        openai_model_type="gpt-3.5-turbo-1106")

    # Combine the keywords info from gpt_keywords_library.json into paper keywords info.
    paper_keywords_info_file = integrate_gpt_keywords(
        paper_summary_info_file=paper_summary_info_file,
        gpt_keywords_library_file=gpt_keywords_library_file,
        paper_keywords_info_file=paper_keywords_info_file)


if __name__ == '__main__':

    args = process_args()

    # Load environmental variables from the .env file
    load_dotenv()
    assert "OPENAI_API_KEY" in os.environ, "OPENAI_API_KEY environment variable not found"
    assert "OPEN_AI_MODEL_TYPE" in os.environ, "OPEN_AI_MODEL_TYPE environment variable not found"

    paper_gpt_categorizing(args.output_folder)
