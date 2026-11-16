"""InterUI command-line entry point."""
import argparse
import importlib
import os
from pathlib import Path

MODES = {
    'InterUI': 'main_refactor',
    'baseline': 'main_baseline_refactor',
    'ablation-knowledge': 'main_ablation_RAG',
    'ablation-layout-tree': 'main_ablation_layout_tree',
}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Generate ArkUI code from UI screenshots; enter your API key in the terminal.')
    parser.add_argument('--mode', choices=MODES, default='InterUI')
    args = parser.parse_args()
    working_dir = Path.cwd()
    ui_root = working_dir / 'UI'
    output_root = working_dir / 'result' / args.mode
    if not ui_root.is_dir():
        parser.error(f'UI directory not found: {ui_root}')
    categories = sorted(p.name for p in ui_root.iterdir() if p.is_dir())
    if not categories:
        parser.error('No category directories found in UI.')
    # Existing prompt readers use paths relative to the project directory.
    os.chdir(root)
    from llm_use_gpt import configure_api
    configure_api()
    module = importlib.import_module(MODES[args.mode])
    module.OUTPUT_ROOT = str(output_root)
    if hasattr(module, 'ARKUI_EXAMPLES_ROOT'):
        module.ARKUI_EXAMPLES_ROOT = str(root / 'arkui_examples')
    for name in categories:
        module.test_category(name, str(ui_root))



if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nRun cancelled.')
        raise SystemExit(130)
