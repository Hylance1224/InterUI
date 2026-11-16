import logging

logger = logging.getLogger(__name__)

import os
from pathlib import Path
import re
import json

from common import get_interaction_images, save_text

from generate import generate_code


# =========================
# 配置
# =========================

UI_ROOT = str(Path(__file__).resolve().parent / "UI")
# OUTPUT_ROOT = "result/gpt5.6_high"
OUTPUT_ROOT = "result/baseline"
# =========================
# 工具函数
# =========================

# =========================
# 处理一个 page
# =========================

def process_page(category_name, app_name, page_name, ui_root):
    logger.debug("=" * 80)
    print(f"Processing: {app_name}/{page_name}")

    # =========================
    # 页面目录
    # =========================

    page_dir = os.path.join(
        ui_root,
        category_name,
        app_name,
        page_name
    )

    if not os.path.exists(page_dir):
        print(f"Page directory not found: {page_dir}")
        return


    # =========================
    # 获取页面下所有图片
    # =========================

    image_sources = []

    for file_name in sorted(os.listdir(page_dir)):

        if file_name.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            image_sources.append(
                os.path.join(page_dir, file_name)
            )


    if not image_sources:
        print(
            f"Skip: no images found: {page_dir}"
        )
        return


    logger.debug(f"Found {len(image_sources)} images:")

    for image_path in image_sources:
        logger.debug(
            f"  {os.path.basename(image_path)}"
        )


    # =========================
    # 创建输出目录
    # =========================

    output_dir = os.path.join(
        OUTPUT_ROOT,
        category_name,
        app_name,
        page_name
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )


    # =========================
    # 已生成则跳过
    # =========================

    output_file = os.path.join(
        output_dir,
        "Index.ets"
    )


    if os.path.exists(output_file):

        # 文件非空，认为已经生成
        if os.path.getsize(output_file) > 0:

            print(
                f"Skip existing: {app_name}/{page_name}/Index.ets"
            )

            return

        else:

            print(
                f"Empty Index.ets found, regenerate: {output_file}"
            )


    # =========================
    # 所有图片 -> ArkUI代码
    # =========================

    print("Generating ArkUI code...")


    try:

        code = generate_code(
            image_sources
        )

    except Exception as e:

        print(
            f"Generation failed: {e}"
        )

        return


    if not code:

        print(
            "Failed to generate code."
        )

        return


    # =========================
    # 保存代码
    # =========================

    save_text(
        code,
        output_file
    )


    print(
        f"Finished: {app_name}/{page_name}"
    )


# =========================
# 批量处理全部页面
# =========================

def main():

    if not os.path.exists(UI_ROOT):
        print(f"UI directory not found: {UI_ROOT}")
        return


    # category
    for category_name in sorted(os.listdir(UI_ROOT)):

        category_dir = os.path.join(
            UI_ROOT,
            category_name
        )

        if not os.path.isdir(category_dir):
            continue


        # app
        for app_name in sorted(os.listdir(category_dir)):

            app_dir = os.path.join(
                category_dir,
                app_name
            )

            if not os.path.isdir(app_dir):
                continue


            # page
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

def test_one_page(category_name, app_name, page_name, UI_ROOT):

    # app_name = "RedBook"
    # page_name = "User"

    process_page(
        "",
        app_name,
        page_name,
        UI_ROOT
    )


def test_one_app(app_name, app_root, category_name=""):
    """
    处理一个 app 下所有 page

    app_root:
        app 所在目录
    """

    app_dir = os.path.join(
        app_root,
        category_name,
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
            app_name,
            app_root=ui_root,
            category_name=category_name
        )


if __name__ == "__main__":

    # 批量处理全部页面
    # main()

    # 如果只想测试单个页面，
    # 注释上面的 main()，打开下面这一行

    process_page("Communication","TrueCaller","calls", ui_root=UI_ROOT, )
    # test_category("Music & Audio", UI_ROOT)


