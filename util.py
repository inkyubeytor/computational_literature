"""
Utilities for preprocessing text files.
"""
import os
import shutil
from typing import List


def split_chapters(path: str, filename: str, sep: str) -> List[str]:
    """
    Processes a text file on disk into chapters.
    :param path: The directory for data relating to the text file.
    :param filename: The name of the raw file in its parent directory.
    :param sep: The separator between chapters in the text file.
    :return: A list of chapter names.
    """
    with open(f"{path}/{filename}", "r", encoding="utf8") as f:
        chapters = [c for c in f.read().split(sep) if len(c) > 0]

    # Create or clear the chapters directory.
    try:
        os.mkdir(f"{path}/chapters")
    except FileExistsError:
        shutil.rmtree(f"{path}/chapters")
        os.mkdir(f"{path}/chapters")

    index = []
    for chapter in chapters:
        # The chapter title is the first line of the chapter, with some
        # substitutions required.
        name = chapter.splitlines()[0] \
            .replace(" ", "_") \
            .replace(".", "-") \
            .replace("#", "") \
            .lower()
        index.append(name)
        with open(f"{path}/chapters/{name}.txt", "w+", encoding="utf8") as f:
            f.writelines(" ".join(
                filter(lambda x: len(x), chapter.splitlines()[1:])))
    with open(f"{path}/chapters_index.txt", "w+", encoding="utf8") as f:
        f.write("\n".join(index))
    return index


def load_chapter(path: str, name: str) -> str:
    """
    Loads a given chapter from a data directory.

    :param path: The data directory for the text file's project.
    :param name: The chapter name.
    :return: The chapter contents as a string.
    """
    with open(f"{path}/chapters/{name}.txt", "r", encoding="utf8") as f:
        return f.read()


def load_index(path: str) -> List[str]:
    """
    Loads a chapter index from a data directory.

    :param path: The data directory for the text file's project.
    :return: The list of chapter titles for the directory.
    """
    with open(f"{path}/chapters_index.txt", "r", encoding="utf8") as f:
        return f.read().splitlines()
