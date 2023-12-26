import json
import sys
import re


def transform_text(input_text):
    """Transform text inside <text> to add color"""
    pattern = re.compile(r"<(.*?)>")
    output_text = pattern.sub(r'<span style="color:#86B300">\1</span>', input_text)
    return output_text


if __name__ == "__main__":
    if len(sys.argv) > 1:  # we check if we received any argument
        if sys.argv[1] == "supports":
            # then we are good to return an exit status code of 0, since the other argument will just be the renderer's name
            sys.exit(0)

    # load both the context and the book representations from stdin
    context, book = json.load(sys.stdin)

    if "sections" in book:
        for item in book["sections"]:
            chapter = item["Chapter"]
            chapter["content"] = transform_text(chapter["content"])
            if len(chapter["sub_items"]) > 0:
                for sub_item in chapter["sub_items"]:
                    sub_chapter = sub_item["Chapter"]
                    sub_chapter["content"] = transform_text(sub_chapter["content"])

    # we are done with the book's modification, we can just print it to stdout
    # print(json.dumps(book["sections"][1]), file=sys.stderr)
    # print(json.dumps(book["sections"][1]["Chapter"]), file=sys.stderr)
    print(json.dumps(book))
