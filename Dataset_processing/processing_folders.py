import os
import re
import shutil
from typing import List, Dict, Tuple

def analyze_filenames_in_folder(folder_path: str, file_type: str) -> Tuple[Dict[str, List[int]], Dict[str, Tuple[int, int]]]:
    """
    Analyzes filenames in the specified folder, extracting attribute names and number ranges.

    :param folder_path: Path to the folder containing the files.
    :return: A tuple containing two dictionaries:
             - The first dictionary has attributes as keys and lists of numbers as values.
             - The second dictionary has attributes as keys and tuples representing the number ranges as values.
    """
    # Get filenames from the folder
    filenames = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    # Extract 'xxx' part and numbers
    attribute_dict = {}
    for filename in filenames:
        match = re.match(rf"([a-z,A-Z,0-9,_]+)_([0-9]+)\.{file_type}", filename)
        if match:
            attribute, number = match.groups()
            number = int(number)
            if attribute in attribute_dict:
                attribute_dict[attribute].append(number)
            else:
                attribute_dict[attribute] = [number]

    # Determine the range for each attribute
    range_dict = {attr: (min(numbers), max(numbers)) for attr, numbers in attribute_dict.items()}
    # Print the name of this function and the result
    print(f"{analyze_filenames_in_folder.__name__}({folder_path}): attributes: {attribute_dict.keys()}, ranges: {range_dict}")
    return attribute_dict, range_dict

def clear_folder(folder_path):
    """
    Clears all files and subdirectories in the specified folder.

    Parameters:
    folder_path (str): The path to the folder to clear.
    """
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")