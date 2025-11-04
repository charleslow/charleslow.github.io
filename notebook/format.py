import json
import sys
import re


def transform_text(input_text):
    """Transform text inside <<text>> to add color"""
    pattern = re.compile(r"(?<!<)<<(?!<)(.+?)(?<!>)>>(?!>)", re.DOTALL)
    output_text = pattern.sub(r'<span style="color:orange">\1</span>', input_text)
    return output_text

def find_arxiv_urls(text):
    """Find all arXiv URLs in the text"""
    arxiv_pattern = re.compile(r"https?://arxiv\.org/abs/\d+\.\d+")
    return arxiv_pattern.findall(text)

if __name__ == "__main__":
    if len(sys.argv) > 1:  # we check if we received any argument
        if sys.argv[1] == "supports":
            # then we are good to return an exit status code of 0, since the other argument will just be the renderer's name
            sys.exit(0)

    # load both the context and the book representations from stdin
    context, book = json.load(sys.stdin)

    arxiv_urls = []

    if "sections" in book:
        for item in book["sections"]:
            chapter = item["Chapter"]
            chapter["content"] = transform_text(chapter["content"])
            arxiv_urls.extend(find_arxiv_urls(chapter["content"]))
            if len(chapter["sub_items"]) > 0:
                for sub_item in chapter["sub_items"]:
                    sub_chapter = sub_item["Chapter"]
                    sub_chapter["content"] = transform_text(sub_chapter["content"])
                    arxiv_urls.extend(find_arxiv_urls(sub_chapter["content"]))

    # Write arXiv URLs to a file
    with open("arxiv_urls.txt", "w") as f:
        for url in sorted(set(arxiv_urls)):
            f.write(url + "\n")

    # we are done with the book's modification, we can just print it to stdout
    # print(json.dumps(book["sections"][1]), file=sys.stderr)
    # print(json.dumps(book["sections"][1]["Chapter"]), file=sys.stderr)
    print(json.dumps(book))
