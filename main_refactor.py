import logging

logger = logging.getLogger(__name__)


import os
from pathlib import Path
import re
import json

from knowledge_retrieval import calculate_example_score, find_top_examples

from common import get_interaction_images, save_text
from generate_transitions import ensure_transitions

from generate import (
    generate_layout_tree,
    generate_question,
    update_layout_tree,
    generate_code_with_layout_tree,
    select_example,
    answer_question,
)


# =========================
# 配置
# =========================

UI_ROOT = str(Path(__file__).resolve().parent / "UI")
OUTPUT_ROOT = "result/InterUI"
ARKUI_EXAMPLES_ROOT = "arkui_examples"
TOP_K_EXAMPLES = 10

# =========================
# 工具函数
# =========================

def clean_json_text(text):
    """
    清理大模型可能返回的 ```json ... ``` Markdown 包裹。
    """
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*",
            "",
            text,
            flags=re.IGNORECASE
        )
        text = re.sub(
            r"\s*```$",
            "",
            text
        )

    return text.strip()


def load_arkui_examples(examples_root):
    """
    读取 arkui_examples 目录下所有 JSON 文件，
    将每个文件中的 examples 展开成统一列表。
    """
    all_examples = []

    if not os.path.exists(examples_root):
        print(f"ArkUI examples directory not found: {examples_root}")
        return all_examples

    for root, directories, files in os.walk(examples_root):
        directories.sort()
        for filename in sorted(files):
            if not filename.lower().endswith(".json"):
                continue

            file_path = os.path.join(root, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
                continue

            document_title = data.get("title", "")
            url = data.get("url", "")
            examples = data.get("examples", [])

            for index, example in enumerate(examples):
                all_examples.append({
                    "document_title": document_title,
                    "example_title": example.get("title", ""),
                    "description": example.get("description", ""),
                    "code": example.get("code", ""),
                    "url": url,
                    "source_file": file_path,
                    "example_index": index
                })

    return all_examples


def retrieve_examples_for_questions(
    questions_text,
    examples_root,
    top_k=10
):
    """
    每个 question 使用其全部 keywords，
    找到 Top-K 最相关 ArkUI examples。
    """

    questions_text = clean_json_text(
        questions_text
    )

    try:
        questions = json.loads(
            questions_text
        )
    except json.JSONDecodeError as e:
        print(f"Failed to parse questions JSON: {e}")
        return []

    all_examples = load_arkui_examples(
        examples_root
    )

    logger.debug(
        f"Loaded {len(all_examples)} ArkUI examples."
    )

    if not all_examples:
        return []

    results = []

    for question_item in questions:
        question = question_item.get(
            "question",
            ""
        )

        keywords = question_item.get(
            "keywords",
            []
        )

        logger.debug("")
        logger.debug(f"Question: {question}")
        logger.debug(f"Keywords: {keywords}")

        matched_examples = find_top_examples(
            keywords,
            all_examples,
            top_k=top_k
        )

        for i, example in enumerate(
            matched_examples,
            start=1
        ):
            logger.debug(
                f"    Top {i}: "
                f"{example['example_title']} "
                f"(score={example['score']})"
            )

        results.append({
            "question": question,
            "keywords": keywords,
            "matched_examples": matched_examples
        })

    return results


def select_best_examples_with_llm(matched_examples):
    """
    For each question, ask the LLM to select one candidate example.
    Selection sees only title + description, not code.
    """
    results = []

    for item in matched_examples:
        question = item.get("question", "")
        keywords = item.get("keywords", [])
        candidates = item.get("matched_examples", [])

        logger.debug("")
        logger.debug(f"Selecting example for: {question}")

        if not candidates:
            logger.debug("    No candidate examples.")
            results.append({
                "question": question,
                "keywords": keywords,
                "selected_example": None
            })
            continue

        # Only expose title and description to the selector LLM.
        candidate_summaries = []
        for i, example in enumerate(candidates):
            candidate_summaries.append({
                "example_index": i,
                "title": example.get("example_title", ""),
                "description": example.get("description", "")
            })

        raw_result = select_example(
            question,
            candidate_summaries
        )

        selected_index = None

        if raw_result:
            try:
                parsed = json.loads(clean_json_text(raw_result))
                selected_index = parsed.get("example_index")
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"    Failed to parse selector result: {e}")

        selected_example = None

        if isinstance(selected_index, int) and 0 <= selected_index < len(candidates):
            selected_example = candidates[selected_index]
            logger.debug(
                f"    Selected: {selected_example.get('example_title', '')}"
            )
        else:
            logger.debug("    Selected: None")

        results.append({
            "question": question,
            "keywords": keywords,
            "selected_example": selected_example
        })

    return results


def generate_answers_for_questions(selected_examples):
    """
    Generate one concise answer for each question using its selected example.
    """
    results = []

    for item in selected_examples:
        question = item.get("question", "")
        selected_example = item.get("selected_example")

        logger.debug("")
        logger.debug(f"Answering: {question}")

        answer = answer_question(
            question,
            selected_example
        )

        if not answer:
            answer = ""

        results.append({
            "question": question,
            "answer": answer.strip()
        })

    return results


# =========================
# 处理一个 page
# =========================

def process_page(category_name, app_name, page_name, ui_root):
    logger.debug("=" * 80)
    print(f"Processing: {app_name}/{page_name}")

    # 根据 UI 根目录、app_name、page_name 自动得到页面目录
    page_dir = os.path.join(
        ui_root,
        app_name,
        page_name
    )

    if not os.path.exists(page_dir):
        print(f"Page directory not found: {page_dir}")
        return

    main_image = os.path.join(page_dir, "main.jpg")

    if not os.path.exists(main_image):
        print(f"Skip: main.jpg not found: {page_dir}")
        return

    # -------------------------
    # 创建输出目录
    # -------------------------

    output_dir = os.path.join(
        OUTPUT_ROOT,
        category_name,
        app_name,
        page_name
    )

    os.makedirs(output_dir, exist_ok=True)

    # Infer predecessors from all screenshots before building the layout tree.
    transitions = ensure_transitions(page_dir, output_dir)

    # -------------------------
    # Step 1
    # main.jpg -> 初始 layout tree
    # -------------------------

    final_layout_tree_path = os.path.join(output_dir, "layout_tree.json")
    interaction_images = list(dict.fromkeys(target for _, target in transitions))

    if os.path.exists(final_layout_tree_path):
        print("1. Loading existing final layout tree...")
        with open(final_layout_tree_path, "r", encoding="utf-8") as f:
            layout_tree = f.read()
    else:
        print("1. Generating initial layout tree...")
        layout_tree = generate_layout_tree([main_image])
        if not layout_tree:
            print("Failed to generate initial layout tree.")
            return

        print("2. Updating layout tree...")
        for predecessor_image, image_path in transitions:
            image_name = os.path.basename(image_path)
            updated_tree = update_layout_tree(
                [predecessor_image, image_path], layout_tree
            )
            if updated_tree:
                layout_tree = updated_tree
            else:
                print(f"Warning: update failed for {image_name}")

        save_text(layout_tree, final_layout_tree_path)

    # Keep retrieval and selection results in memory only.
    if not os.path.exists(os.path.join(output_dir, "answers.json")):
        print("3. Generating questions...")
        questions = generate_question(layout_tree)
        if not questions:
            print("Failed to generate questions.")
            return

        print("4. Retrieving relevant ArkUI examples...")
        matched_examples = retrieve_examples_for_questions(
            questions, ARKUI_EXAMPLES_ROOT, top_k=TOP_K_EXAMPLES
        )

        print("5. Selecting the best example with LLM...")
        selected_examples = select_best_examples_with_llm(matched_examples)

    # -------------------------
    # Step 6
    # question + selected example -> concise answer
    # -------------------------

    answers_path = os.path.join(
        output_dir,
        "answers.json"
    )

    if os.path.exists(answers_path):
        print("6. Loading existing answers...")

        with open(
            answers_path,
            "r",
            encoding="utf-8"
        ) as f:
            answers = json.load(f)

        # Normalize legacy caches before passing answers to subsequent steps.
        qa_answers = [
            {"question": item.get("question", ""), "answer": item.get("answer", "")}
            for item in answers
        ]
        if qa_answers != answers:
            with open(answers_path, "w", encoding="utf-8") as f:
                json.dump(qa_answers, f, ensure_ascii=False, indent=2)
        answers = qa_answers

    else:
        print("6. Generating concise answers...")

        answers = generate_answers_for_questions(
            selected_examples
        )

        with open(
            answers_path,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                answers,
                f,
                ensure_ascii=False,
                indent=2
            )


    # -------------------------
    # Step 7
    # screenshots + layout tree + questions/answers -> final ArkUI code
    # -------------------------

    code_path = os.path.join(
        output_dir,
        "Index.ets"
    )

    if os.path.exists(code_path):
        print("7. Final ArkUI code already exists, skipping generation.")

    else:
        print("7. Generating final ArkUI code...")

        # 给代码生成模型提供所有页面截图：
        # main.jpg + p1.jpg + p2.jpg + ...
        all_images = [
            main_image
        ] + interaction_images


        code = generate_code_with_layout_tree(
            all_images,
            layout_tree,
            answers
        )

        if not code:
            print("Failed to generate final ArkUI code.")
            return

        # 清理模型可能返回的 ```typescript / ```arkts 包裹
        code = code.strip()

        if code.startswith("```"):
            code = re.sub(
                r"^```(?:typescript|ts|arkts|ets)?\\s*",
                "",
                code,
                flags=re.IGNORECASE
            )
            code = re.sub(
                r"\\s*```$",
                "",
                code
            )

        save_text(
            code.strip(),
            code_path
        )

        print(f"Final ArkUI code saved to: {code_path}")



# =========================
# 批量处理全部页面
# =========================

def main():

    if not os.path.exists(UI_ROOT):
        print(f"UI directory not found: {UI_ROOT}")
        return

    # UI/
    #   app1/
    #       page1/
    #       page2/
    #   app2/
    #       page1/

    for category_name in sorted(os.listdir(UI_ROOT)):

        category_dir = os.path.join(
            UI_ROOT,
            category_name
        )

        if not os.path.isdir(category_dir):
            continue

        for app_name in sorted(os.listdir(category_dir)):

            app_dir = os.path.join(
                category_dir,
                app_name
            )

            if not os.path.isdir(app_dir):
                continue

            for page_name in sorted(os.listdir(app_dir)):

                page_dir = os.path.join(
                    app_dir,
                    page_name
                )

                if not os.path.isdir(page_dir):
                    continue

                process_page(
                    category_name,
                    app_name,
                    page_name,
                    category_dir
                )


# =========================
# 单页面测试
# =========================

def test_one_page():

    category_name = "Social"

    app_name = "Reddit"
    page_name = "Create"

    process_page(
        category_name,
        app_name,
        page_name,
        UI_ROOT
    )


def test_one_app(category_name, app_name, app_root):
    """
    处理一个 app 下所有 page

    app_root:
        app 所在目录
    """

    app_dir = os.path.join(
        app_root,
        app_name
    )

    if not os.path.exists(app_dir):
        print(
            f"App directory not found: {app_dir}"
        )
        return


    for page_name in sorted(os.listdir(app_dir)):

        page_dir = os.path.join(
            app_dir,
            page_name
        )

        if not os.path.isdir(page_dir):
            continue


        print(
            f"Found page: {app_name}/{page_name}"
        )


        process_page(
            category_name,
            app_name,
            page_name,
            app_root
        )

def test_category(category_name, ui_root=UI_ROOT):
    """
    根据 category 自动遍历所有 app，
    然后调用 test_one_app。

    目录结构:
        UI/
          category_name/
              app1/
                  page1/
                  page2/
              app2/
                  page1/

    参数:
        category_name: 类别名称
    """

    category_dir = os.path.join(
        ui_root,
        category_name
    )

    if not os.path.exists(category_dir):
        print(
            f"Category directory not found: {category_dir}"
        )
        return


    for app_name in sorted(os.listdir(category_dir)):

        app_dir = os.path.join(
            category_dir,
            app_name
        )

        # 只处理文件夹(app)
        if not os.path.isdir(app_dir):
            continue


        print(
            f"\nProcessing app: {category_name}/{app_name}"
        )


        test_one_app(
            category_name,
            app_name,
            app_root=category_dir
        )


if __name__ == "__main__":
    test_category("Music & Audio", UI_ROOT)
