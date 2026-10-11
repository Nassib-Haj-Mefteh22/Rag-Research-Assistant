import argparse
import sys

from rag.generate import answer
from rag.retrieve import DEFAULT_K
from rag.tools import DEFAULT_MODE, load_titles


def format_sources(sources: list[tuple[str, int]], titles: dict[str, str]) -> list[str]:
    seen: set[tuple[str, int]] = set()
    lines = []
    for doc_id, page in sources:
        if (doc_id, page) in seen:
            continue
        seen.add((doc_id, page))
        lines.append(f"[{doc_id}] {titles.get(doc_id, '(title unknown)')}, p.{page}")
    return lines


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag.cli", description="Ask the RAG assistant")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer a question from the paper corpus")
    ask.add_argument("question")
    ask.add_argument("--k", type=int, default=DEFAULT_K, help="passages given to the model")
    ask.add_argument("--mode", default=DEFAULT_MODE, help="dense, hybrid, [+rerank]")
    ask.add_argument("--show-chunks", action="store_true", help="also print retrieved passages")
    return parser


def main(argv: list[str] | None = None) -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = build_parser().parse_args(argv)
    result = answer(args.question, k=args.k, mode=args.mode)
    print(result["answer"])
    if args.show_chunks:
        print("\nRetrieved passages:")
        for chunk in result["chunks"]:
            snippet = chunk["text"][:200].replace("\n", " ")
            print(f"  {chunk['rank']}. [{chunk['doc_id']}] p.{chunk['page']} | {snippet}")
    lines = format_sources(result["sources"], load_titles())
    if lines:
        print("\nSources:")
        for line in lines:
            print(f"  {line}")
    if result["invalid_citations"]:
        print(f"\nWarning: the model cited non-existent passages {result['invalid_citations']}")


if __name__ == "__main__":
    main()
