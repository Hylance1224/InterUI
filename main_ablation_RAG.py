import logging

logger = logging.getLogger(__name__)


import os
from pathlib import Path
import re
import json
from difflib import SequenceMatcher

from common import get_interaction_images, save_text
from generate_transitions import ensure_transitions

from generate import (
    generate_layout_tree,
    update_layout_tree,
    generate_code_without_RAG,
    select_example,
    answer_question,
)


# =========================
# 配置
# =========================

UI_ROOT = str(Path(__file__).resolve().parent / "UI")
OUTPUT_ROOT = "result/ablation-knowledge"
ARKUI_EXAMPLES_ROOT = "arkui_examples"
TOP_K_EXAMPLES = 5

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

    for root, _, files in os.walk(examples_root):
        for filename in files:
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


def normalize_text(text):
    if not text:
        return ""
    return str(text).lower().strip()


def tokenize(text):
    return re.findall(
        r"[a-zA-Z0-9_]+",
        normalize_text(text)
    )


def calculate_example_score(keywords, example):
    """
    根据一个 question 的全部 keywords，
    对一个 example 进行整体相关度评分。

    评分重点：
    1. keyword 完整出现在 example title
    2. keyword 完整出现在 description
    3. keyword 完整出现在文档 title
    4. token 匹配
    5. 字符串相似度

    通用事件词（如 onClick）适当降权，
    更具体的 ArkUI API 名称权重更高。
    """

    example_title = normalize_text(
        example.get("example_title", "")
    )
    description = normalize_text(
        example.get("description", "")
    )
    document_title = normalize_text(
        example.get("document_title", "")
    )

    title_tokens = set(tokenize(example_title))
    description_tokens = set(tokenize(description))
    document_tokens = set(tokenize(document_title))

    score = 0.0

    generic_keywords = {
        "onclick",
        "onchange",
        "onscroll",
        "ontouch",
        "ongesture",
        "button",
        "text",
        "image",
        "icon"
    }

    for keyword in keywords:
        keyword_norm = normalize_text(keyword)

        if not keyword_norm:
            continue

        # 通用词降权，专有 API 默认权重更高
        keyword_weight = 0.1 if keyword_norm in generic_keywords else 1.0

        # 完整关键词匹配
        if keyword_norm in example_title:
            score += 100 * keyword_weight

        if keyword_norm in description:
            score += 100 * keyword_weight

        if keyword_norm in document_title:
            score += 50 * keyword_weight

        # Token 匹配
        keyword_tokens = set(tokenize(keyword))

        for token in keyword_tokens:
            token_norm = normalize_text(token)
            token_weight = (
                0.35 if token_norm in generic_keywords else keyword_weight
            )

            if token in title_tokens:
                score += 50 * token_weight

            if token in description_tokens:
                score += 50 * token_weight

            if token in document_tokens:
                score += 25 * token_weight

    return score


def find_top_examples(
    keywords,
    all_examples,
    top_k=5
):
    """
    使用一个 question 的全部 keywords，
    从全部 example 中返回 Top-K 最相关结果。
    """

    scored_examples = []

    for example in all_examples:
        score = calculate_example_score(
            keywords,
            example
        )

        if score <= 0:
            continue

        result = dict(example)
        result["score"] = round(score, 4)

        scored_examples.append(result)

    scored_examples.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return scored_examples[:top_k]


def retrieve_examples_for_questions(
    questions_text,
    examples_root,
    top_k=5
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
                raise RuntimeError(f"Layout tree update failed for {image_name}; stopping this run.")

        save_text(layout_tree, final_layout_tree_path)

    # -------------------------
    # Step 7
    # screenshots + layout tree -> final ArkUI code
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


        code = generate_code_without_RAG(
            all_images,
            layout_tree
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

    app_name = "RedBook"
    page_name = "User"

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
    os.chdir(Path(__file__).resolve().parent)
    main()
