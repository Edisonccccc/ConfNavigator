import argparse
import sys
import time
import os
import json
import shutil
import utils as utils

from dotenv import load_dotenv

from paper_pdf_cleaner import paper_pdf_cleaning

from paper_pdf_parser import paper_pdf_parsing
from paper_gpt_summarizer import paper_gpt_summarizing
from paper_gpt_keywords_categorizer import paper_gpt_categorizing


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--arxiv-pdf-folder",
        type=str,
        dest="arxiv_pdf_folder",
        default=
        f"***",
        help="The folder path for the output files.",
    )

    parser.add_argument(
        "--openreview-pdf-folder",
        type=str,
        dest="openreview_pdf_folder",
        default=
        f"***",
        help="The folder path for the output files.",
    )

    parser.add_argument(
        "--nips-csv",
        type=str,
        dest="nips_csv",
        default=
        f"***.csv",
        help="The file path for the csv file.",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        dest="output_folder",
        default=f"***",
        help="The folder path for the output files.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = process_args()
    # Load environmental variables from the .env file
    load_dotenv()

    combined_paper_pdf_folder = "***"

    do_cleaning_paper = True
    do_parsing_paper = True
    do_summarizing_paper = True
    do_categorizing_paper = True

    if do_cleaning_paper:

        # Paper cleaning
        paper_pdf_cleaning(
            arxiv_paper_pdf_folder=args.arxiv_pdf_folder,
            openreview_paper_pdf_folder=args.openreview_pdf_folder,
            csv_file_path=args.nips_csv,
            combined_paper_pdf_folder=combined_paper_pdf_folder,
            output_folder=args.output_folder)

    if do_parsing_paper:
        paper_pdf_parsing(args.output_folder)

    if do_summarizing_paper:
        paper_gpt_summarizing(args.output_folder)

    if do_categorizing_paper:
        paper_gpt_categorizing(args.output_folder)
