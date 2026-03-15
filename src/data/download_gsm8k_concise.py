"""Download GSM8K and save self-distill-ready concise prompts to JSON."""

from __future__ import annotations

import json
from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = Path(__file__).resolve().parents[2] / "gsm8k-concise.json"
PROMPT_PREFIX = "be concise\n\n"


def main() -> None:
    dataset = load_dataset("gsm8k", "main")
    rows: list[dict[str, str]] = []

    for row in dataset["train"]:
        question = str(row["question"])
        answer = str(row["answer"])
        if not question.startswith(PROMPT_PREFIX):
            question = f"{PROMPT_PREFIX}{question}"

        rows.append(
            {
                "student_prompt": question,
                "teacher_prompt": f"{question}\n\nReference solution:\n{answer}",
                "question": question,
                "answer": answer,
            }
        )

    OUTPUT_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} training examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
