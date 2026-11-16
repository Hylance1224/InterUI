"""Infer screenshot predecessors with the vision model; save only under result/."""
import argparse
from pathlib import Path

from common import get_interaction_images, get_layout_tree_transitions, save_text


def render_transitions(page_dir):
    """Send main.jpg and every interaction screenshot in a named, ordered batch."""
    from llm_use_gpt import generate_use_llm

    page = Path(page_dir)
    images = [str(page / "main.jpg"), *get_interaction_images(page_dir)]
    prompt_dir = Path(__file__).resolve().parent / "prompt" / "our"
    system = (prompt_dir / "system_generate_transitions.txt").read_text(encoding="utf-8")
    user = (prompt_dir / "user_generate_transitions.txt").read_text(encoding="utf-8")
    mapping = "\n".join(f"Image {i}: {Path(image).name}" for i, image in enumerate(images, 1))
    response = generate_use_llm(images, system, user.replace("{{image_mapping}}", mapping))
    if not response or not response.strip():
        raise ValueError(f"Empty transition response for {page}")
    text = response.strip()
    # Accept a single Markdown fence, while retaining strict record validation.
    lines = text.splitlines()
    if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
        text = "\n".join(lines[1:-1]).strip()
    return text + "\n"


def ensure_transitions(page_dir, output_dir, overwrite=False):
    """Validate cached results or infer and validate before publishing a new file."""
    destination = Path(output_dir) / "transitions.txt"
    if destination.is_file() and not overwrite:
        return get_layout_tree_transitions(page_dir, output_dir)
    if not (Path(page_dir) / "main.jpg").is_file():
        raise ValueError(f"Missing main.jpg: {page_dir}")
    print(f"Inferring screenshot predecessors: {page_dir}")
    content = render_transitions(page_dir) if get_interaction_images(page_dir) else ""
    transitions = get_layout_tree_transitions(page_dir, output_dir, content=content)
    save_text(content, str(destination))
    return transitions


def generate_templates(ui_root, dry_run=False, overwrite=False, mode="InterUI"):
    ui_root = Path(ui_root).resolve()
    output_root = ui_root.parent / "result" / mode
    created = skipped = missing_main = 0
    pages = sorted({p.parent for p in ui_root.rglob("*")
                    if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}})
    for page in pages:
        if not (page / "main.jpg").is_file():
            print(f"SKIP (missing main.jpg): {page.relative_to(ui_root)}")
            missing_main += 1
            continue
        output_dir = output_root / page.relative_to(ui_root)
        if (output_dir / "transitions.txt").exists() and not overwrite:
            get_layout_tree_transitions(page, output_dir)
            skipped += 1
            continue
        if dry_run:
            print(f"Would infer: {page} -> {output_dir / 'transitions.txt'}")
        else:
            ensure_transitions(page, output_dir, overwrite=overwrite)
        created += 1
    print(f"{'Would write' if dry_run else 'Written'}: {created}; cached: {skipped}; missing main.jpg: {missing_main}")
    return created, skipped, missing_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="List outputs without model requests or writes")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate cached transitions")
    parser.add_argument("--mode", choices=["InterUI", "ablation-knowledge"], default="InterUI")
    args = parser.parse_args()
    ui_root = Path.cwd() / "UI"
    if not ui_root.is_dir():
        parser.error(f"UI directory not found: {ui_root}. Run from the project directory.")
    generate_templates(ui_root, args.dry_run, args.overwrite, args.mode)


if __name__ == "__main__":
    main()
