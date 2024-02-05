import json
import os

PAPER_CLEANING_INFO_FILE = "paper_cleaning_info.json"
PAPER_PARSING_INFO_FILE = "paper_parsing_info.json"
PAPER_SUMMARY_INFO_FILE = "paper_summary_info.json"
GPT_SUMMARY_LIBRARY_FILE = "gpt_summary_library.json"
GPT_KEYWORDS_LIBRARY_FILE = "gpt_keywords_library.json"
PAPER_KEYWORDS_INFO_FILE = "paper_keywords_info.json"

CONCLUSION_TITLES = [
    "Conclusions", "Conclusion", "CONCLUSION"
    "Conclusions and Future Work", "Conclusion and Future Work",
    "Conclusion and limitation", "Discussion and Limitations",
    "Conclusion and future work", "Summary", "Discussion",
    "Conclusion, Limitations, and Border Impacts",
    "Limitations and Future Directions", "Conclusion and Discussion",
    "Concluding Remarks", "Conclusions and Limitations", "Future Work",
    "Conclusion, Discussion and Limitations",
    "Related Work Beyond Active Learning", "Numerical demonstrations",
    "Discussion and Future Directions", "Perspectives",
    "Conclusion, Limitations and Future Directions",
    "Discussion and Conclusion", "Limitations and Future Work",
    "Conclusions and future work", "Discussion and future directions",
    "Summary and Discussion", "Discussion and Conclusion", "Limitations",
    "Limitations and Conclusions", "Numerical experiments",
    "Discussion & Conclusion", "Discussions, Limitations and Future Work",
    "Limitations and Conclusion", "Conclusion and limitations",
    "Concluding remarks", "Discussions", "Conclusion and Limitation",
    "Discussion and Future Work", "Experiments", "Related work",
    "CONCLUDING REMARK", "Conclusion Remarks", "Experimental Evaluation",
    "Conclusion and Limitations", "Conclusions and Future Work",
    "Discussion, Limitations and Future Work", "Conclusions and Discussions",
    "Conclusion and open problems", "Discussion and future work",
    "Further applications", "Future work",
    "Conclusion, Limitations, and Future Work", "Conclusions and Future Work",
    "Building a polynomial-time approximate oracle",
    "Limitations and future work", "Discussion and limitations",
    "Conclusion, Future Work and Limitations",
    "Further discussion and conclusion", "Open directions",
    "Technical Overview", "Conclusion and Future Directions",
    "Related Work and Conclusion", "Discussion: Extensions and Limitations",
    "Discussions and Future Work"
]


def read_json_file(json_file_path: str):
    """Check if the json file exists and read the file"""
    if json_file_path and os.path.isfile(json_file_path):
        with open(json_file_path, 'r') as data_file:
            return json.load(data_file)
    else:
        print(f"Could not find the json file {(json_file_path)}.")
        return None


def dump_json_file(data, json_file_path):
    with open(json_file_path, 'w') as fp:
        json.dump(data, fp, indent=4)


def can_be_int(s):
    try:
        int(s)  # Try to convert the string to an integer
        return True  # Return True if successful
    except ValueError:
        return False  # Return False if a ValueError is raised


def create_parsed_json_file_path_by_pdf_path(pdf_file_path):

    filename_with_extension = os.path.basename(pdf_file_path)
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    output_folder_path = os.path.dirname(pdf_file_path)

    parsed_paper_json_file_path = os.path.join(
        output_folder_path,
        f"{filename_without_extension}_parsed_chapters.json")

    return parsed_paper_json_file_path


class TrieNode:
    def __init__(self):
        self.children = {}
        self.isEndOfWord = False


class Trie:
    """
    Title extraction from the pdf file is a bit of tricky since there is no keyword to tell how many lines a title has or when to ends. the second line of the title might have one word or two words which makes it even tricker to distinguish it from the author line. So potentially, the first name of the first author might be appended to the extracted title.
    Due to the limitation in the extracted title from the pdf, the Trie is added to check if the prefix of the extracted title exists in the set which contains all titles from the arxiv csv
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.isEndOfWord = True

    def search(self, word):
        node = self.root
        prefix = ""
        for char in word:
            if char not in node.children:
                return None
            node = node.children[char]
            prefix += char
            if node.isEndOfWord:
                return prefix
        return None


def find_all_pdf_files_in_folder(folder_path):
    # List all the files in the directory
    all_files = os.listdir(folder_path)
    # Filter out the files that are not PDFs
    pdf_files = [
        os.path.join(folder_path, file) for file in all_files
        if file.lower().endswith('.pdf')
    ]

    return pdf_files
