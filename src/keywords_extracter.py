import os
import json


import utils as utils
summary_file = f"{os.getcwd()}/summary_selected_papers.json"
summary_data = utils.read_json_file(summary_file)

keywords_summary = {}
count = 0
real_count = 0
for paper_index, paper_data in summary_data.items():

    try:
        if paper_index in ["input_total_tokens", "output_total_tokens", "total_cost"]:
            continue

        if "summary" in paper_data:
            count += 1
            summary_paragraph = paper_data["summary"]

            summary_paragraph = summary_paragraph.replace("\n\n", "\n")
            summary_paragraph = summary_paragraph.replace(":\n", ":")
            summary_paragraph = summary_paragraph.replace(":-", ":")
            summary_paragraph = summary_paragraph.replace(":   -", ":")
            summary_paragraph = summary_paragraph.replace("\n   -", "")
             

            summary_lines = summary_paragraph.split("\n")
            if len(summary_lines) > 4:
                keywords_line = summary_lines[4]
            else:
                start_index = summary_paragraph.find("5. Keywords: ")
                if start_index == -1:
                    print("Start substring not found")
                    continue

                end_index = summary_paragraph.find(", 6.", start_index)
                if end_index == -1:
                    print("End substring not found")
                    continue

                # Adjust end_index to include the end_substring in the result
                # end_index += len(end_substring)
                keywords_line = summary_paragraph[start_index, end_index]

                # return full_string[start_index:end_index]
                # continue

            if "5. Keywords:" in keywords_line:

                keywords_line = keywords_line.replace("5. Keywords:", "")
                keywords_list = keywords_line.split(", ")
                keywords_list[-1] = keywords_list[-1][:-1]
                keywords_list = [element.strip() for element in keywords_list]

                keywords_summary[paper_index] = keywords_list

                if len(keywords_list) > 0:
                    real_count += 1
            elif "5. Keywords include" in keywords_line:
                keywords_line = keywords_line.replace("5. Keywords include", "")
                keywords_list = keywords_line.split(", ")
                keywords_list[-1] = keywords_list[-1][:-1]
                keywords_summary[paper_index] = keywords_list
                keywords_list = [element.strip() for element in keywords_list]

                if len(keywords_list) > 0:
                    real_count += 1
            else:
                pass
                # breakpoint()

                    
    except Exception as _e:
        print(_e)
        breakpoint()

breakpoint()
utils.dump_json_file(keywords_summary, f"{os.getcwd()}/keywords_summary.json")
        









