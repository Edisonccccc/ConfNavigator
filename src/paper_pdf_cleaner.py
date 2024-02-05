import argparse
import sys
import time
import os
import json
import traceback

import shutil
import utils as utils

from dotenv import load_dotenv
import pandas as pd
import fitz  # PyMuPDF

def filter_papers_in_folder(conference_paper_folder):
    paper_metadata_summary = {}
    paper_pdf_files = utils.find_all_pdf_files_in_folder(
        folder_path=conference_paper_folder)
    # Select papers that were published in 2023.
    papers_in_2023, papers_not_in_2023 = select_papers_based_on_year(
        paper_pdf_files)
    paper_metadata_summary["papers_in_2023"] = papers_in_2023
    paper_metadata_summary["papers_not_in_2023"] = papers_not_in_2023
    paper_metadata_summary["papers_in_2023_count"] = len(papers_in_2023)
    paper_metadata_summary["papers_not_in_2023_count"] = len(
        papers_not_in_2023)

    return paper_metadata_summary



def select_papers_based_on_year(paper_pdf_files):
    """
    Remove the paper that was not published in 2023.
    """
    papers_in_2023 = []
    papers_not_in_2023 = []
    for paper_pdf_file in paper_pdf_files:
        document = fitz.open(paper_pdf_file)
        page = document.load_page(0)  # zero-based index
        page_text = page.get_text()

        if '2023' in page_text:
            papers_in_2023.append(paper_pdf_file)
        else:
            papers_not_in_2023.append(paper_pdf_file)

    return papers_in_2023, papers_not_in_2023


def convert_arxiv_csv_to_json(csv_file_path):

    # Reading the CSV data into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Dropping the first column as it's not needed (optional)
    # df = df.drop(df.columns[0], axis=1)

    # Convert the DataFrame to a list of dictionaries (each dictionary representing a paper)
    papers_metadata = df.to_dict(orient='records')

    paper_metadata_dict = {}

    for paper_metadata in papers_metadata:
        paper_metadata_dict[int(paper_metadata["id"])] = paper_metadata

    # Extract the file name with extension
    filename_with_extension = os.path.basename(csv_file_path)
    # Split the file name from its extension
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    arxiv_json_file = os.path.join(os.path.dirname(csv_file_path),
                                   f"{filename_without_extension}.json")
    utils.dump_json_file(data=paper_metadata_dict, json_file_path=arxiv_json_file)

    return arxiv_json_file


def filter_papers_in_folder(conference_paper_folder):
    paper_metadata_summary = {}
    paper_pdf_files = utils.find_all_pdf_files_in_folder(
        folder_path=conference_paper_folder)
    # Select papers that were published in 2023.
    papers_in_2023, papers_not_in_2023 = select_papers_based_on_year(
        paper_pdf_files)
    paper_metadata_summary["papers_in_2023"] = papers_in_2023
    paper_metadata_summary["papers_not_in_2023"] = papers_not_in_2023
    paper_metadata_summary["papers_in_2023_count"] = len(papers_in_2023)
    paper_metadata_summary["papers_not_in_2023_count"] = len(
        papers_not_in_2023)

    return paper_metadata_summary


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



def filter_paper_by_year(paper_pdf_file):
    """
    Remove the paper that was not published in 2023.
    """

    document = fitz.open(paper_pdf_file)
    page = document.load_page(0)  # zero-based index
    page_text = page.get_text()

    if '2023' in page_text:
        return True
    else:
        return False



def paper_pdf_cleaning(arxiv_paper_pdf_folder, openreview_paper_pdf_folder, csv_file_path, combined_paper_pdf_folder, output_folder):
    # Convert the csv to json file
    csv_json_file = convert_arxiv_csv_to_json(csv_file_path)
    paper_csv_json = utils.read_json_file(csv_json_file)
    paper_overview_file = f"{output_folder}/paper_overview.json"

    papers_overview = {}
    valid_arxiv_pdf = 0
    valid_openview_pdf = 0
    combined_pdf = 0
    for paper_id, paper_data in paper_csv_json.items():
        # !@#$%^&*()-+?_=,<>/"
        correct_title = paper_data["title"].replace(":", "_").replace("'", "_").replace("?", "_").replace("\\", "_")

        arxiv_pdf_file_path = os.path.join(arxiv_paper_pdf_folder, f'{correct_title}.pdf')
        
        if os.path.isfile(arxiv_pdf_file_path):
            paper_data["arxiv_pdf_path"] = arxiv_pdf_file_path

            valid_arxiv_pdf += 1
        else:
            paper_data["arxiv_pdf_path"] = ""
            # print(arxiv_pdf_file_path)
        
        openreview_pdf_file_path = os.path.join(openreview_paper_pdf_folder,  f'{correct_title}.pdf')

        if os.path.isfile(openreview_pdf_file_path):
            paper_data["openreview_pdf_path"] = openreview_pdf_file_path
            valid_openview_pdf += 1
        else:
            paper_data["openreview_pdf_path"] = ""

        pdf_file_to_parse = os.path.join(combined_paper_pdf_folder, f'{correct_title}.pdf')
        if paper_data["arxiv_pdf_path"]:
            if not os.path.isfile(pdf_file_to_parse):
                shutil.copy(paper_data["arxiv_pdf_path"], pdf_file_to_parse)
            paper_data["pdf_file_to_parse"] = pdf_file_to_parse
            combined_pdf += 1
        elif paper_data["openreview_pdf_path"]:
            if not os.path.isfile(pdf_file_to_parse):
                shutil.copy(paper_data["openreview_pdf_path"], pdf_file_to_parse)
            paper_data["pdf_file_to_parse"] = pdf_file_to_parse
            combined_pdf += 1
        else:
            paper_data["pdf_file_to_parse"] = ""

        
    papers_overview['arxiv_total_downloaded_papers'] = len(utils.find_all_pdf_files_in_folder(arxiv_paper_pdf_folder))

    papers_overview['arxiv_found_pdf_papers_based_on_title'] = valid_arxiv_pdf


    papers_overview['openreview_total_downloaded_papers'] = len(utils.find_all_pdf_files_in_folder(openreview_paper_pdf_folder))

    papers_overview['openreview_found_pdf_papers_based_on_title'] = valid_openview_pdf

    papers_overview['combined_found_pdf_papers_based_on_title'] = combined_pdf

    paper_cleaning_json_file = f"{args.output_folder}/paper_cleaning_with_arxiv_and_openreview.json"

    utils.dump_json_file(paper_csv_json, paper_cleaning_json_file)

    utils.dump_json_file(papers_overview, paper_overview_file)







