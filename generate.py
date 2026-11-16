import json

from llm_use_gpt import generate_use_llm
# from llm_use_qwen import generate_use_llm

def generate_layout_tree(image_sources):
    with open("prompt/our/system_layout_tree.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    user_prompt = "Analyze the provided UI screenshot and generate its complete Hierarchical UI Layout Tree."
    layout_tree = generate_use_llm(image_sources, system_prompt, user_prompt)
    return layout_tree


def update_layout_tree(image_sources, layout_tree):
    with open("prompt/our/system_update_layout_tree.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open("prompt/our/user_update_layout_tree.txt", "r", encoding="utf-8") as f:
        user_prompt = f.read()
    user_prompt = user_prompt.replace("{{layout_tree}}", layout_tree)
    updated_tree = generate_use_llm(image_sources, system_prompt, user_prompt)
    return updated_tree


def generate_question(layout_tree):
    with open("prompt/our/system_generate_keyword.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open("prompt/our/user_generate_keyword.txt", "r", encoding="utf-8") as f:
        user_prompt = f.read()
    user_prompt = user_prompt.replace("{{layout_tree}}", layout_tree)
    question = generate_use_llm([], system_prompt, user_prompt)
    return question


def generate_code_with_layout_tree(image_sources, layout_tree, answers):
    """
    Generate final ArkUI code using:
    1. UI screenshots
    2. final layout tree
    3. implementation questions and their answers
    """
    with open("prompt/our/system_generate_code.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open("prompt/our/user_generate_code.txt", "r", encoding="utf-8") as f:
        user_prompt = f.read()

    # 只保留 question 和 answer
    qa_list = []

    for item in answers:
        qa_list.append({
                "question": item.get("question", ""),
                "answer": item.get("answer", "")
        })

    answers_text = json.dumps(
        qa_list,
        ensure_ascii=False,
        indent=2
    )

    user_prompt = user_prompt.replace("{{layout_tree}}", layout_tree)
    user_prompt = user_prompt.replace("{{questions_and_answers}}", answers_text)

    code = generate_use_llm(
        image_sources,
        system_prompt,
        user_prompt
    )
    return code


def generate_code_without_RAG(image_sources, layout_tree):
    """
    Generate final ArkUI code using:
    1. UI screenshots
    2. final layout tree
    3. implementation questions and their answers
    """
    with open("prompt/ablation_RAG/system_generate_code.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open("prompt/ablation_RAG/user_generate_code.txt", "r", encoding="utf-8") as f:
        user_prompt = f.read()

    user_prompt = user_prompt.replace("{{layout_tree}}", layout_tree)

    code = generate_use_llm(
        image_sources,
        system_prompt,
        user_prompt
    )
    return code

def generate_code(image_sources):
    with open("prompt/baseline/system_generate_code.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open("prompt/baseline/user_generate_code.txt", "r", encoding="utf-8") as f:
        user_prompt = f.read()
    code = generate_use_llm(image_sources, system_prompt, user_prompt)
    return code




def select_example(question, examples):
    """
    Ask the LLM to select the single most useful candidate example.
    `examples` should contain only candidate metadata intended for selection,
    such as example_index, title and description.
    """
    with open("prompt/our/system_select_example.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open("prompt/our/user_select_example.txt", "r", encoding="utf-8") as f:
        user_prompt = f.read()

    examples_text = json.dumps(
        examples,
        ensure_ascii=False,
        indent=2
    )

    user_prompt = user_prompt.replace("{{question}}", question)
    user_prompt = user_prompt.replace("{{examples}}", examples_text)

    result = generate_use_llm([], system_prompt, user_prompt)
    return result


def answer_question(question, example):
    """
    Generate a concise answer for one ArkUI implementation question.
    `example` may be None when no suitable reference example was selected.
    """
    with open("prompt/our/system_answer_question.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()
    with open("prompt/our/user_answer_question.txt", "r", encoding="utf-8") as f:
        user_prompt = f.read()

    if example is None:
        example_text = "No suitable reference example was found."
    else:
        example_text = json.dumps(
            {
                "title": example.get("example_title", ""),
                "description": example.get("description", ""),
                "code": example.get("code", "")
            },
            ensure_ascii=False,
            indent=2
        )

    user_prompt = user_prompt.replace("{{question}}", question)
    user_prompt = user_prompt.replace("{{example}}", example_text)

    answer = generate_use_llm([], system_prompt, user_prompt)
    return answer


def generate_question_by_image(image_sources):
    """Generate interaction questions from main.jpg followed by p*.jpg."""
    from pathlib import Path

    prompt_dir = Path(__file__).resolve().parent / "prompt" / "ablation_layout_tree"
    system_prompt = (prompt_dir / "system_generate_keyword.txt").read_text(encoding="utf-8")
    user_prompt = (prompt_dir / "user_generate_keyword.txt").read_text(encoding="utf-8")
    return generate_use_llm(image_sources, system_prompt, user_prompt)


def generate_code_without_layout_tree(image_sources, answers):
    """Generate ArkUI code from screenshots and retrieved implementation answers."""
    from pathlib import Path

    prompt_dir = Path(__file__).resolve().parent / "prompt" / "ablation_layout_tree"
    system_prompt = (prompt_dir / "system_generate_code.txt").read_text(encoding="utf-8")
    user_prompt = (prompt_dir / "user_generate_code.txt").read_text(encoding="utf-8")
    qa_list = [
        {"question": item.get("question", ""), "answer": item.get("answer", "")}
        for item in answers
    ]
    user_prompt = user_prompt.replace(
        "{{questions_and_answers}}",
        json.dumps(qa_list, ensure_ascii=False, indent=2),
    )
    return generate_use_llm(image_sources, system_prompt, user_prompt)
