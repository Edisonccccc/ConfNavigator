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
        last_second_newline_index = indices[-2] if len(indices) > 1 else None

        self.raw_text += page_raw_text[:last_second_newline_index + 2]
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

        assert '2023' in page_text, f"{self.pdf_file_path} is not published in 2023."

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
                    if page_index == len(document):
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
                    if page_index == len(document):
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
                    if page_index == len(document):
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
                    if page_index == len(document):
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

    def dump_parsed_chapters(self, output_folder_path: str = ""):

        # Extract the file name with extension
        filename_with_extension = os.path.basename(self.pdf_file_path)

        # Split the file name from its extension
        filename_without_extension, _ = os.path.splitext(
            filename_with_extension)

        if output_folder_path == "":
            output_folder_path = os.path.dirname(self.pdf_file_path)
        parsed_chapters_json_file_path = os.path.join(
            output_folder_path,
            f"{filename_without_extension}_parsed_chapters.json")

        chapters_content = {
            "Title": self.title.print_chapter(),
            "Abstract": self.abstract.print_chapter(),
            "Introduction": self.introduction.print_chapter(),
            "Conclusion": self.conclusion.print_chapter(),
            "References": self.references.print_chapter()
        }

        utils.dump_json_file(chapters_content, parsed_chapters_json_file_path)
        with open(parsed_chapters_json_file_path, 'w') as file:
            json.dump(chapters_content, file, indent=4)

        return parsed_chapters_json_file_path


def find_all_pdf_files_in_folder(folder_path):
    # List all the files in the directory
    all_files = os.listdir(folder_path)
    # Filter out the files that are not PDFs
    pdf_files = [
        os.path.join(folder_path, file) for file in all_files
        if file.lower().endswith('.pdf')
    ]

    return pdf_files


def parse_paper(paper_pdf_file, is_full_paper: bool = False):

    try:
        if is_full_paper:
            paper_content = convert_full_pdf_to_text(paper_pdf_file)
        else:
            paper_pdf = PaperPDFParser(paper_pdf_file)
            paper_pdf.paper_pdf_parsing()

            # TODO: validate if the parsing file is valid like each chapter has content
            assert paper_pdf.title.content != "", f"{paper_pdf_file} - title is empty"
            assert paper_pdf.abstract.content != "", f"{paper_pdf_file} - abstract is empty"
            assert paper_pdf.introduction.content != "", f"{paper_pdf_file} - introduction is empty"
            assert paper_pdf.conclusion.content != "", f"{paper_pdf_file} - conclusion is empty"

            parsed_paper_json_file_path = paper_pdf.dump_parsed_chapters()

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
    for page_num in range(len(document)):
        page = document.load_page(page_num)  # zero-based index
        breakpoint()
        text += page.get_text()[:-4]

    # Extract the file name with extension
    filename_with_extension = os.path.basename(pdf_file_path)
    # Split the file name from its extension
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    output_folder_path = os.path.dirname(pdf_file_path)
    full_paper_json_file_path = os.path.join(
        output_folder_path, f"{filename_without_extension}_full_paper.json")

    utils.dump_json_file(text, full_paper_json_file_path)

    return full_paper_json_file_path


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
    json_data = df.to_dict(orient='records')
    # Extract the file name with extension
    filename_with_extension = os.path.basename(csv_file_path)
    # Split the file name from its extension
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    arxiv_json_file = os.path.join(os.path.dirname(csv_file_path),
                                   f"{filename_without_extension}.json")
    utils.dump_json_file(data=json_data, json_file_path=arxiv_json_file)

    return arxiv_json_file


def filter_papers_in_folder(conference_paper_folder):
    paper_metadata_summary = {}
    paper_pdf_files = find_all_pdf_files_in_folder(
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


def parse_all_papers(paper_metadata_summary):
    paper_metadata_summary["mapping_pdf_to_json"] = {}
    paper_metadata_summary["mapping_json_to_pdf"] = {}
    parsed_2023_paper_json_files = []

    title_empty_papers = []
    abstract_empty_papers = []
    introduction_empty_papers = []
    conclusion_empty_papers = []
    # TODO: Skip the paper pdf file if a json file is already available.
    # Step 1: Parse all papers in PDF format and convert them into Json format.
    for paper_pdf_file in paper_metadata_summary["papers_in_2023"]:
        parsed_paper_json_file = parse_paper(paper_pdf_file)
        if os.path.isfile(parsed_paper_json_file):
            parsed_2023_paper_json_files.append(parsed_paper_json_file)
            paper_metadata_summary["mapping_pdf_to_json"][
                paper_pdf_file] = parsed_paper_json_file
            paper_metadata_summary["mapping_json_to_pdf"][
                parsed_paper_json_file] = paper_pdf_file
        elif 'title is empty' in parsed_paper_json_file:
            title_empty_papers.append(parsed_paper_json_file)
        elif 'abstract is empty' in parsed_paper_json_file:
            abstract_empty_papers.append(parsed_paper_json_file)
        elif 'introduction is empty' in parsed_paper_json_file:
            introduction_empty_papers.append(parsed_paper_json_file)
        elif 'conclusion is empty' in parsed_paper_json_file:
            conclusion_empty_papers.append(parsed_paper_json_file)

    paper_metadata_summary[
        "2023_paper_parsed_jsons"] = parsed_2023_paper_json_files
    paper_metadata_summary['2023_paper_parsed_jsons_count'] = len(
        parsed_2023_paper_json_files)

    paper_metadata_summary['2023_paper_title_empty'] = title_empty_papers
    paper_metadata_summary['2023_paper_title_empty_count'] = len(
        title_empty_papers)
    paper_metadata_summary['2023_paper_abstract_empty'] = abstract_empty_papers
    paper_metadata_summary['2023_paper_abstract_empty_count'] = len(
        abstract_empty_papers)
    paper_metadata_summary[
        '2023_paper_introduction_empty'] = introduction_empty_papers
    paper_metadata_summary['2023_paper_introduction_empty_count'] = len(
        introduction_empty_papers)
    paper_metadata_summary[
        '2023_paper_conclusion_empty'] = conclusion_empty_papers
    paper_metadata_summary['2023_paper_conclusion_empty_count'] = len(
        conclusion_empty_papers)

    return paper_metadata_summary


def cross_check_paper_titles():
    # Convert the csv file from the conference paper downloader to json file
    csv_file_path = './neurips2023_pdfs.csv'
    parsed_chapters_json_file_path = convert_arxiv_csv_to_json(csv_file_path)

    # Cross check if the downloaded PDF file matches the title from the paper downloder list
    trie = Trie()
    papers_list_from_arxiv_csv = utils.read_json_file(
        './neurips2023_pdfs.json')

    paper_metadata_dict = {}
    for paper_metadata in papers_list_from_arxiv_csv:
        paper_metadata_dict[paper_metadata['title']] = paper_metadata

    titles_from_arxiv = [
        paper_metadata['title']
        for paper_metadata in papers_list_from_arxiv_csv
    ]

    for title in titles_from_arxiv:
        trie.insert(title)

    parsed_summary_file = f"{os.getcwd()}/paper_metadata_summary.json"
    paper_metadata_summary = utils.read_json_file(parsed_summary_file)
    papers_match_csv = {}
    index = 0
    for parsed_json_file in paper_metadata_summary["2023_paper_parsed_jsons"]:
        parsed_paper_data = utils.read_json_file(parsed_json_file)
        paper_title = parsed_paper_data['Title']
        matched_title_from_csv = trie.search(paper_title)

        if matched_title_from_csv:
            papers_match_csv[index] = {}
            papers_match_csv[index]["paper_title"] = paper_title

            papers_match_csv[index]["json"] = parsed_json_file
            papers_match_csv[index]["metadata"] = paper_metadata_dict[
                matched_title_from_csv]
            papers_match_csv[index]["pdf"] = paper_metadata_summary[
                "mapping_json_to_pdf"][parsed_json_file]
            index += 1

    paper_metadata_summary["papers_matched"] = papers_match_csv
    paper_metadata_summary["papers_matched_count"] = len(papers_match_csv)

    return paper_metadata_summary


def process_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--conference-paper-folder",
        dest="conference_paper_folder",
        required=False,
        default=
        "/import/snvm-sc-podscratch1/qingjianl2/nips_2023_conference_papers/neurips2023_pdf_0109",
        help="The folder path of conference papers",
    )

    parser.add_argument(
        "--test",
        dest="test",
        action='store_true',
        required=False,
        help="For testing purpose.",
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        dest="output_folder",
        default=f"{os.curdir}",
        help="The folder path for the output files.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = process_args()

    # To skip the paper filtering
    skip_year_filtering = True
    # To skip the paper parsing
    skip_paper_parsing = True

    # Load environmental variables from the .env file
    load_dotenv()

    paper_metadata_summary = {}
    parsed_summary_file = f"{os.getcwd()}/paper_metadata_summary.json"

    # Step 1: Paper cleaning - to remove papers that are not belonging to this conference
    if not skip_year_filtering:
        paper_metadata_summary = filter_papers_in_folder(
            conference_paper_folder=args.conference_paper_folder)
        utils.dump_json_file(paper_metadata_summary, parsed_summary_file)
    else:
        paper_metadata_summary = utils.read_json_file(parsed_summary_file)

    # Step 2: Paper parsing - parse the pdf into a json with different chapters
    if not skip_paper_parsing:
        # Start the timer for paper parsing
        start_time = time.time()
        paper_metadata_summary = parse_all_papers(paper_metadata_summary)
        # End the timer for paper parsing
        end_time = time.time()
        time_taken = end_time - start_time
        confnavigator_logger.info(f"Parsing 2023 papers cost {time_taken} s")

        utils.dump_json_file(paper_metadata_summary, parsed_summary_file)
    else:
        paper_metadata_summary = utils.read_json_file(parsed_summary_file)

    # Step 3: Paper cross checking - compare the title in PDF matches with the paper title from arxic csv
    paper_metadata_summary = cross_check_paper_titles()
    utils.dump_json_file(paper_metadata_summary, parsed_summary_file)
