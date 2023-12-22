import json
import sys
import re
import os
from pdb import set_trace

MAPPING = {
    "1 Timothy": "1tim",
}

REVERSE_MAPPING = {v: k for k, v in MAPPING.items()}


def parse_chapter(text: str) -> dict:
    verse = 1
    res = {}
    while len(text) > 0:
        pattern = f"(\[{verse}\])"
        match_start = re.search(pattern, text)
        if not match_start:
            break
        text = text[match_start.end() + 1 :]
        match_end = re.search(f"(\[{verse+1}\])", text)
        if not match_end:
            res[verse] = text
            break
        res[verse] = text[: match_end.start()].strip()
        verse += 1
    return res


def load_bible() -> dict:
    bible = {}
    for filename in os.listdir("raw/"):
        filename = filename.removesuffix(".txt")
        book, chapter = filename.split("-")
        book = MAPPING.get(book, book)
        text = open(f"raw/{filename}.txt").read()
        if book not in bible:
            bible[book] = {}
        bible[book][int(chapter)] = parse_chapter(text)
    return bible


def transform_text(input_text):
    """Transform text inside <text> to add color"""
    pattern = re.compile(r"<(.*?)>")
    output_text = pattern.sub(r'<span style="color:orange">\1</span>', input_text)
    return output_text


def add_bible_verses(input_text: str):
    """Add bible verses to the text

    For each line, if it starts with \B, then we replace it with the bible verse
    e.g.
        \B Genesis 1 1-4 means we replace it with the bible verse from Genesis 1:1-4
    """
    lines = input_text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("\B"):
            print(line, file=sys.stderr)
            book, remnant = line[3:].split(" ")
            chapter, verses = remnant.split(":")
            verses = verses.split("-")
            if len(verses) == 1:
                line = bible[book][int(chapter)][int(verses[0])]
            else:
                line = " ".join(
                    [
                        bible[book][int(chapter)][verse]
                        for verse in range(int(verses[0]), int(verses[1]) + 1)
                    ]
                )
            lines[i] = (
                "> "
                + line
                + f"\n `[ESV] {REVERSE_MAPPING[book]} {chapter}:{'-'.join(verses)}`"
            )
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1:  # we check if we received any argument
        if sys.argv[1] == "supports":
            # then we are good to return an exit status code of 0, since the other argument will just be the renderer's name
            sys.exit(0)

    # load both the context and the book representations from stdin
    context, book = json.load(sys.stdin)

    bible = load_bible()

    if "sections" in book:
        for item in book["sections"]:
            chapter = item["Chapter"]
            chapter["content"] = transform_text(chapter["content"])
            chapter["content"] = add_bible_verses(chapter["content"])
            if len(chapter["sub_items"]) > 0:
                for sub_item in chapter["sub_items"]:
                    sub_chapter = sub_item["Chapter"]
                    sub_chapter["content"] = transform_text(sub_chapter["content"])
                    sub_chapter["content"] = add_bible_verses(sub_chapter["content"])

    # we are done with the book's modification, we can just print it to stdout
    # print(json.dumps(book["sections"][1]), file=sys.stderr)
    # print(json.dumps(book["sections"][1]["Chapter"]), file=sys.stderr)
    print(json.dumps(book))
