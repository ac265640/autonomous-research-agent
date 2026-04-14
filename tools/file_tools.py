import os


def save_report(title: str, content: str, filename: str):
    os.makedirs("reports", exist_ok=True)

    path = f"reports/{filename}.md"

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write(content)

    return f"Saved to {path}"