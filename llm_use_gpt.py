import os
import base64
import getpass
import sys
import warnings
from pathlib import Path
from openai import OpenAI

_client = None
_model = None


def configure_api():
    """Read credentials once per process without saving them to disk."""
    global _client, _model
    if _client is not None:
        return _client
    if not sys.stdin.isatty():
        raise RuntimeError("Run in an interactive terminal and enter your API key manually.")
    try:
        while True:
            # Refuse getpass's visible-input fallback in unsupported terminals.
            with warnings.catch_warnings():
                warnings.simplefilter("error", getpass.GetPassWarning)
                key = getpass.getpass("Enter API key (input hidden): ").strip()
            if key:
                break
            print("API key cannot be empty. Please try again.")
        model = input("Model name [gpt-5.6-sol]: ").strip() or "gpt-5.6-sol"
    except (EOFError, getpass.GetPassWarning) as exc:
        raise RuntimeError("Cannot read input securely. Please use PowerShell, CMD, or a system terminal.") from exc
    _client = OpenAI(api_key=key, base_url="https://api.openai.com/v1")
    _model = model
    return _client


def encode_image(image_path):

    with open(image_path, "rb") as f:
        data = base64.b64encode(
            f.read()
        ).decode("utf-8")

    return data

def build_image_content(image_sources):
    """
    根据图片路径列表构造 input_image

    image_sources:
        [
            "a.jpg",
            "b.png"
        ]
    """

    contents = []


    for image_path in image_sources:

        img = Path(image_path)


        if img.suffix.lower() not in [
            ".png",
            ".jpg",
            ".jpeg"
        ]:
            continue


        base64_image = encode_image(
            img
        )


        # 获取正确mime
        suffix = img.suffix.lower()[1:]


        contents.append(
            {
                "type": "input_image",
                "image_url":
                    f"data:image/{suffix};base64,{base64_image}"
            }
        )


    return contents


def generate_use_llm(image_dir, SYSTEM_PROMPT, USER_PROMPT):


    images = build_image_content(
        image_dir
    )


    client = configure_api()
    response = client.responses.create(

        model=_model,
        reasoning={
            "effort": "high"
        },
        input=[

            {
                "role":"system",
                "content":
                SYSTEM_PROMPT
            },


            {
                "role":"user",

                "content":[

                    {
                        "type":"input_text",
                        "text": USER_PROMPT
                    }

                ]
                +
                images

            }

        ]

    )


    return response.output_text