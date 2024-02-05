import argparse
import sys
import time
import os
import json
import traceback
from abc import ABC, abstractmethod

from dotenv import load_dotenv
import pandas as pd
import fitz  # PyMuPDF

import utils as utils
from utils import CONCLUSION_TITLES, Trie
from logger import confnavigator_logger
from utils import PAPER_PARSING_INFO_FILE, PAPER_CLEANING_INFO_FILE


class PaperChapter(ABC):
    """
    Chapters:
    Title Chapter
    Abstract Chapter
    Introduction Chapter
    Conclusion Chapter
    References Chapter
    """
    def __init__(self):
        self.raw_text = ""
        self.page_numbers = []

    def append_page(self, page_raw_text, page_number):
        # Finding the index of the last second "\n" character in the string to remove the footer
        indices = [i for i, char in enumerate(page_raw_text) if char == "\n"]
        if len(indices) > 1:
            last_second_newline_index = indices[-2]
            self.raw_text += page_raw_text[:last_second_newline_index + 2]
        else:
            self.raw_text += page_raw_text
        self.page_numbers.append(page_number)

    @abstractmethod
    def chapter_parsing(self):
        pass

    @abstractmethod
    def create_content(self):
        """Different data cleansing method will be implemented in here"""
        pass

    @abstractmethod
    def extract_chapter_index(self):
        """"""
        pass

    def print_chapter(self):
        return self.content


class TitleChapter(PaperChapter):
    def __init__(self):
        super().__init__()

    def chapter_parsing(self):
        self.chapter_index = self.extract_chapter_index()
        self.content, self.content_lines_list = self.create_content()

    def extract_chapter_index(self):
        return -1

    def create_content(self):
        """
        Assumption:
        1. The title might be one line or multiple lines. 
        2. Each line will be more than two words to distinguish it from the author lines.
        3. There won't be a title with more than 5 lines.
        """
        raw_text_in_lines = self.raw_text.split("\n")
        title_content_lines_list = []

        for index, line in enumerate(raw_text_in_lines[0:2]):
            if index == 0:
                title_content_lines_list.append(line)
            else:
                # Condition index > 0

                if "." in line:
                    break

                if "," in line:
                    break

                if "Institute" in line:
                    break

                if "University" in line:
                    break

                if "1" in line:
                    break

                title_content_lines_list.append(line)

        content = " ".join(title_content_lines_list)
        content_lines_list = title_content_lines_list

        return content, content_lines_list


class AbstractChapter(PaperChapter):
    def __init__(self):
        super().__init__()

    def chapter_parsing(self):
        self.chapter_index = self.extract_chapter_index()
        self.content, self.content_lines_list = self.create_content()

    def extract_chapter_index(self):
        # Define the index of the Abstract chapter is 0.
        return 0

    def chapter_ending_criteria(self):
        return "\nIntroduction\n" in self.raw_text

    def create_content(self):
        raw_text_in_lines = self.raw_text.split("\n")
        abstract_content_lines_list = []

        is_in_abstract_chapter = False
        is_in_next_chapter = False

        for line in raw_text_in_lines:
            if "Abstract" in line or "ABSTRACT" in line:
                is_in_abstract_chapter = True

            if "Introduction" in line:
                is_in_abstract_chapter = False
                is_in_next_chapter = True

            if is_in_next_chapter:
                break

            if is_in_abstract_chapter:
                abstract_content_lines_list.append(line)

        content = " ".join(abstract_content_lines_list)
        content_lines_list = abstract_content_lines_list

        return content, content_lines_list


class IntroductionChapter(PaperChapter):
    def __init__(self):
        super().__init__()
        self.chapter_index = self.extract_chapter_index()

    def chapter_parsing(self):
        self.content, self.content_lines_list = self.create_content()

    def extract_chapter_index(self):
        # Define the index of the Introduction chapter is 0.
        return 1

    def create_content(self):
        raw_text_in_lines = self.raw_text.split("\n")
        introduction_content_lines_list = []
        is_in_introduction_chapter = False
        is_in_next_chapter = False
        for line in raw_text_in_lines:
            if "Introduction" in line:
                is_in_introduction_chapter = True

            if f"{self.chapter_index + 1}" == line:
                is_in_next_chapter = True

            if is_in_next_chapter:
                break

            # Ignore the footer
            if ".edu" in line or "NeurIPS" in line:
                continue

            if is_in_introduction_chapter:
                introduction_content_lines_list.append(line)

        content = " ".join(introduction_content_lines_list)
        content_lines_list = introduction_content_lines_list

        return content, content_lines_list


class ConclusionChapter(PaperChapter):
    def __init__(self):
        super().__init__()

    def chapter_parsing(self):

        self.chapter_index = self.extract_chapter_index()
        self.content, self.content_lines_list = self.create_content()

    def extract_chapter_index(self):
        raw_text_in_lines = self.raw_text.split("\n")

        for i, line in enumerate(raw_text_in_lines):
            if "Conclusion" in line:
                # \n5\nConclusion\n

                if utils.can_be_int(raw_text_in_lines[i - 1]):
                    return int(raw_text_in_lines[i - 1])

        return -1

    def create_content(self):
        raw_text_in_lines = self.raw_text.split("\n")
        conclusion_content_lines_list = []

        is_in_conclusion_chapter = False
        is_in_next_chapter = False

        for line in raw_text_in_lines:
            if line in CONCLUSION_TITLES:
                is_in_conclusion_chapter = True

            if "References" in line:
                is_in_conclusion_chapter = False
                is_in_next_chapter = True

            if is_in_next_chapter:
                break

            if is_in_conclusion_chapter:
                conclusion_content_lines_list.append(line)

        content = " ".join(conclusion_content_lines_list)
        content_lines_list = conclusion_content_lines_list

        return content, content_lines_list


class ReferencesChapter(PaperChapter):
    def __init__(self):
        super().__init__()

    def chapter_parsing(self):
        self.chapter_index = self.extract_chapter_index()
        self.content, self.content_lines_list = self.create_content()

    def extract_chapter_index(self):
        # Reference chapter usually don't have chapter index.
        return -1

    def create_content(self):
        raw_text_in_lines = self.raw_text.split("\n")
        references_content_lines_list = []

        is_in_reference_chapter = False
        is_in_next_chapter = False

        for line in raw_text_in_lines:
            if "References" in line:
                is_in_reference_chapter = True

            # pattern = r"^\[\d+\]"
            # if not re.match(pattern, line):
            #     is_in_next_chapter = True
            #     is_in_reference_chapter = False

            # if is_in_next_chapter:
            #     break

            if is_in_reference_chapter:
                references_content_lines_list.append(line)

        content = " ".join(references_content_lines_list)
        content_lines_list = references_content_lines_list

        return content, content_lines_list


class PaperPDFParser():
    def __init__(self, pdf_file_path: str):
        assert os.path.isfile(
            pdf_file_path
        ), f"Error: The provided {pdf_file_path} is not valid!"

        self.pdf_file_path = pdf_file_path
        self.title = TitleChapter()
        self.abstract = AbstractChapter()
        self.introduction = IntroductionChapter()
        self.conclusion = ConclusionChapter()
        self.references = ReferencesChapter()

    def paper_pdf_parsing(self):
        document = fitz.open(self.pdf_file_path)
        # Iterate over each page and extract text
        text = ""
        page_index = 0

        page = document.load_page(0)  # zero-based index
        page_text = page.get_text()

        while page_index < len(document):
            page = document.load_page(page_index)  # zero-based index
            page_text = page.get_text()

            if "\nAbstract\n" in page_text or "\nABSTRACT\n" in page_text:
                self.title.append_page(page_text, page_index)

            if "\nAbstract\n" in page_text or "\nABSTRACT\n" in page_text:
                is_in_abstract_chapter = True

                while is_in_abstract_chapter:
                    self.abstract.append_page(page_text, page_index)
                    if "\nIntroduction\n" in page_text:
                        is_in_abstract_chapter = False
                        break

                    page_index += 1
                    if page_index >= len(document):
                        break

                    page = document.load_page(page_index)  # zero-based index
                    page_text = page.get_text()

            if "\nIntroduction\n" in page_text:
                is_in_introduction_chapter = True

                while is_in_introduction_chapter:
                    self.introduction.append_page(page_text, page_index)
                    if "2\n" in page_text[:-4]:
                        is_in_introduction_chapter = False
                        break

                    page_index += 1
                    if page_index >= len(document):
                        break
                    page = document.load_page(page_index)  # zero-based index
                    page_text = page.get_text()

            if self.check_if_conclusion_title_in_page_text(page_text):
                is_in_conclusion_chapter = True

                while is_in_conclusion_chapter:
                    self.conclusion.append_page(page_text, page_index)
                    if "References\n" in page_text:
                        is_in_conclusion_chapter = False
                        break

                    page_index += 1
                    if page_index >= len(document):
                        break
                    page = document.load_page(page_index)  # zero-based index
                    page_text = page.get_text()

            if "References\n" in page_text:
                is_in_reference_chapter = True

                while is_in_reference_chapter:
                    page_lines = page_text.split("\n")
                    if "References\n" not in page_text and page_lines[
                            0] != "References" and not page_lines[
                                0].startswith("["):
                        is_in_reference_chapter = False
                        break
                    else:
                        self.references.append_page(page_text, page_index)

                    page_index += 1
                    if page_index >= len(document):
                        break
                    page = document.load_page(page_index)  # zero-based index
                    page_text = page.get_text()

            page_index += 1

        self.title.chapter_parsing()
        self.abstract.chapter_parsing()
        self.introduction.chapter_parsing()
        self.conclusion.chapter_parsing()
        self.references.chapter_parsing()

    def check_if_conclusion_title_in_page_text(self, page_text):
        for conclusion_title in CONCLUSION_TITLES:
            if f"\n{conclusion_title}\n" in page_text:
                return True
        return False

    def dump_parsed_chapters(self, parsed_paper_json_file_path: str = ""):
        chapters_content = {
            "Title": self.title.print_chapter(),
            "Abstract": self.abstract.print_chapter(),
            "Introduction": self.introduction.print_chapter(),
            "Conclusion": self.conclusion.print_chapter(),
            "References": self.references.print_chapter()
        }

        if parsed_paper_json_file_path == "":
            parsed_paper_json_file_path = utils.create_parsed_json_file_path_by_pdf_path(
                self.pdf_file_path)

        utils.dump_json_file(chapters_content, parsed_paper_json_file_path)
        with open(parsed_paper_json_file_path, 'w') as file:
            json.dump(chapters_content, file, indent=4)

        return parsed_paper_json_file_path


def parse_paper(paper_pdf_file,
                is_full_paper: bool = False,
                is_redo_parsing: bool = False):

    paper_json_file_path = utils.create_parsed_json_file_path_by_pdf_path(
        paper_pdf_file)

    if os.path.isfile(paper_json_file_path) and not is_redo_parsing:
        # Skip parsing
        return paper_json_file_path

    try:
        if is_full_paper:
            paper_content = convert_full_pdf_to_text(paper_pdf_file)
        else:

            paper_pdf = PaperPDFParser(paper_pdf_file)
            paper_pdf.paper_pdf_parsing()

            parsed_paper_json_file_path = paper_pdf.dump_parsed_chapters(
                paper_json_file_path)

            return parsed_paper_json_file_path
    except Exception as _e:
        print(_e)
        traceback.print_exc()
        print(f"Failed to parse file {paper_pdf_file}")
        confnavigator_logger.exception(
            f"Exception in parsing {paper_pdf_file}", _e)
        confnavigator_logger.error(f"Failed to parse file {paper_pdf_file}")

        return str(_e)


def convert_full_pdf_to_text(pdf_file_path):
    # Open the provided PDF document
    document = fitz.open(pdf_file_path)
    # Iterate over each page and extract text
    text = ""
    for page_num in range(min(len(document), 3)):
        page = document.load_page(page_num)  # zero-based index
        text += page.get_text()[:-4]

    # Extract the file name with extension
    filename_with_extension = os.path.basename(pdf_file_path)
    # Split the file name from its extension
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    output_folder_path = os.path.dirname(pdf_file_path)
    full_paper_json_file_path = os.path.join(
        output_folder_path, f"{filename_without_extension}_full_paper.json")

    utils.dump_json_file({"FullPage": text}, full_paper_json_file_path)

    return full_paper_json_file_path


def parse_all_papers(paper_metadata_summary, paper_overview):
    paper_overview["mapping_pdf_to_json"] = {}
    paper_overview["mapping_json_to_pdf"] = {}

    title_empty_papers = []
    abstract_empty_papers = []
    introduction_empty_papers = []
    conclusion_empty_papers = []
    # TODO: Skip the paper pdf file if a json file is already available.
    # Step 1: Parse all papers in PDF format and convert them into Json format.
    count = 0
    for paper_index, paper_metadata in paper_metadata_summary.items():
        print(f"parsing paper {paper_index}")
        paper_pdf_file = paper_metadata["pdf_file_to_parse"]

        if "parsed_json_file" in paper_metadata:
            continue
        if not os.path.isfile(paper_pdf_file):
            paper_metadata["parsing_error"] = "No valid PDF file."
            continue

        parsed_paper_json_file = parse_paper(paper_pdf_file)

        if not os.path.isfile(parsed_paper_json_file):
            paper_metadata["parsing_error"] = parsed_paper_json_file
            continue

        paper_metadata["parsed_json_file"] = parsed_paper_json_file
        parsed_paper_json = utils.read_json_file(parsed_paper_json_file)

        if parsed_paper_json["Title"] == "":
            title_empty_papers.append(parsed_paper_json_file)
        if parsed_paper_json["Abstract"] == "":
            abstract_empty_papers.append(parsed_paper_json_file)

        if parsed_paper_json["Introduction"] == "":
            introduction_empty_papers.append(parsed_paper_json_file)

        if parsed_paper_json["Conclusion"] == "":
            conclusion_empty_papers.append(parsed_paper_json_file)

        if parsed_paper_json["Abstract"] == "" and parsed_paper_json[
                "Introduction"] == "" and parsed_paper_json["Conclusion"] == "":
            parsed_paper_json_file = parse_paper(paper_pdf_file,
                                                 is_full_paper=True)
            paper_metadata["parsed_json_file"] = parsed_paper_json_file
            paper_metadata["fullpage"] = True

        print(f"parsed json file {paper_metadata['parsed_json_file']}")

        paper_overview["mapping_pdf_to_json"][
            paper_pdf_file] = parsed_paper_json_file
        paper_overview["mapping_json_to_pdf"][
            parsed_paper_json_file] = paper_pdf_file

    paper_overview['2023_paper_parsed_jsons_count'] = count

    paper_overview['2023_paper_title_empty'] = title_empty_papers
    paper_overview['2023_paper_title_empty_count'] = len(title_empty_papers)
    paper_overview['2023_paper_abstract_empty'] = abstract_empty_papers
    paper_overview['2023_paper_abstract_empty_count'] = len(
        abstract_empty_papers)
    paper_overview['2023_paper_introduction_empty'] = introduction_empty_papers
    paper_overview['2023_paper_introduction_empty_count'] = len(
        introduction_empty_papers)
    paper_overview['2023_paper_conclusion_empty'] = conclusion_empty_papers
    paper_overview['2023_paper_conclusion_empty_count'] = len(
        conclusion_empty_papers)

    return paper_metadata_summary, paper_overview


def cross_check_paper_titles(paper_metadata_summary, paper_overview):

    # Cross check if the downloaded PDF file matches the title from the paper downloder list
    trie = Trie()

    titles_from_arxiv = [
        paper_metadata['title']
        for id, paper_metadata in paper_metadata_summary.items()
    ]

    title_to_index_csv = {}
    for id, paper_metadata in paper_metadata_summary.items():
        title_to_index_csv[paper_metadata['title']] = id

    for title in titles_from_arxiv:
        trie.insert(title)

    count = 0
    pdf_paper_title_to_index = {}
    unmatched_titles_from_pdf = {}
    for paper_id, paper_metadata in paper_metadata_summary.items():

        if "parsed_json_file" not in paper_metadata or paper_metadata[
                "parsed_json_file"] == None or not os.path.isfile(
                    paper_metadata["parsed_json_file"]):
            continue
        parsed_json_file = paper_metadata["parsed_json_file"]

        parsed_paper_data = utils.read_json_file(parsed_json_file)
        if parsed_paper_data == None or parsed_paper_data.get('Title',
                                                              "") == "":
            continue
        paper_title = parsed_paper_data['Title']
        matched_title_from_csv = trie.search(paper_title)
        pdf_paper_title_to_index[paper_title] = paper_id

        if matched_title_from_csv:
            paper_metadata["paper_title_from_pdf"] = paper_title
            paper_metadata["paper_title_from_csv"] = matched_title_from_csv
            paper_metadata["matched_with_downloaded_pdf"] = True
            count += 1
        else:
            unmatched_titles_from_pdf[paper_title] = {}
            unmatched_titles_from_pdf[paper_title][
                "parsed_json_file"] = paper_metadata["parsed_json_file"]
            unmatched_titles_from_pdf[paper_title][
                "pdf_file_to_parse"] = paper_metadata["pdf_file_to_parse"]
            unmatched_titles_from_pdf[paper_title][
                "paper_title_from_pdf"] = paper_title

    paper_overview["papers_matched_count"] = count
    paper_overview["csv_title_to_id"] = title_to_index_csv
    paper_overview["pdf_title_to_id"] = pdf_paper_title_to_index
    paper_overview["unmatched_titles_from_pdf"] = unmatched_titles_from_pdf

    return paper_metadata_summary, paper_overview


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


def paper_pdf_parsing(output_folder):

    paper_cleaning_info_file = f"{output_folder}/{PAPER_CLEANING_INFO_FILE}"

    paper_parsing_info_file = f"{output_folder}/{PAPER_PARSING_INFO_FILE}"

    paper_overview_file = f"{output_folder}/paper_overview.json"

    # Step 2: Paper parsing - parse the pdf into a json with different chapters
    if not skip_paper_parsing:
        paper_metadata_summary = utils.read_json_file(paper_cleaning_info_file)
        paper_overview = utils.read_json_file(paper_overview_file)

        # Start the timer for paper parsing
        start_time = time.time()
        paper_metadata_summary, paper_overview = parse_all_papers(
            paper_metadata_summary, paper_overview)
        # End the timer for paper parsing
        end_time = time.time()
        time_taken = end_time - start_time
        confnavigator_logger.info(f"Parsing 2023 papers cost {time_taken} s")

        utils.dump_json_file(paper_metadata_summary, paper_parsing_info_file)
        utils.dump_json_file(paper_overview, paper_overview_file)


if __name__ == "__main__":

    args = process_args()

    # To skip the paper parsing
    skip_paper_parsing = False

    # Load environmental variables from the .env file
    load_dotenv()

    # parse pdfs
    paper_pdf_parsing(args.output_folder)

# else:
#     paper_metadata_summary = utils.read_json_file(paper_parsing_result_file)
#     paper_overview = utils.read_json_file(paper_overview_file)

# # Step 3: Paper cross checking - compare the title in PDF matches with the paper title from arxic csv
# paper_metadata_summary, paper_overview = cross_check_paper_titles(paper_metadata_summary, paper_overview)

# # papers_after_matching_file = f"{output_folder}/papers_after_matching.json"

# utils.dump_json_file(paper_metadata_summary, papers_after_matching_file)
# utils.dump_json_file(paper_overview, paper_overview_file)

# mismatch_between_openreview_and_arxiv = []

# summary_data = utils.read_json_file(f"/import/snvm-sc-podscratch1/qingjianl2/nips/arxiv_outputs/summary_selected_papers.json")
# summary_count = 0
# for _, paper_summary in summary_data.items():
#     pdf_paper_title = paper_summary["paper_title"]

#     if pdf_paper_title not in paper_overview["pdf_title_to_id"]:
#         mismatch_between_openreview_and_arxiv.append(pdf_paper_title)
#         continue

#     paper_id = paper_overview["pdf_title_to_id"][pdf_paper_title]
#     if "summary" in paper_summary:
#         paper_metadata_summary[paper_id]['summary'] = paper_summary.get("summary", "")
#         paper_metadata_summary[paper_id]['response'] = paper_summary.get("response", "")
#         paper_metadata_summary[paper_id]['prompt'] = paper_summary.get("prompt", "")
#         paper_metadata_summary[paper_id]['prompt_tokens'] = paper_summary.get("prompt_tokens", 0)
#         paper_metadata_summary[paper_id]['completion_tokens'] = paper_summary.get("completion_tokens", 0)
#         paper_metadata_summary[paper_id]['total_tokens'] = paper_summary.get("total_tokens", 0)
#         summary_count += 1

# papers_after_summary_file = f"{args.output_folder}/papers_after_summary.json"
# breakpoint()
# utils.dump_json_file(paper_metadata_summary, papers_after_summary_file)

# paper_overview["summary_count"] = summary_count
# utils.dump_json_file(paper_overview, paper_overview_file)

# mismatch_between_openreview_and_arxiv_file = f"{args.output_folder}/mismatch_between_openreview_and_arxiv_file.json"
# utils.dump_json_file(mismatch_between_openreview_and_arxiv, mismatch_between_openreview_and_arxiv_file)
