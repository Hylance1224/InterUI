# InterUI: MLLM-Driven Multi-Agent Interactive Declarative UI Code Generation for OpenHarmony

## Installation

Use Python 3.10 or later. Run in PowerShell:

```powershell
cd "InterUI"
pip install -r requirements.txt
python.exe -m nltk.downloader wordnet averaged_perceptron_tagger_eng
```

## Quick Start

```powershell
python.exe run.py --mode InterUI
```

Enter the following values when prompted:

1.  **API Key:** Type or paste your key. Input is hidden and cannot be empty.

2.  **Model name:** Press Enter to retain the original default, `gpt-5.6-sol`, or enter a model available through your provider and account.

The key stays in process memory and is never saved to source or configuration files. Enter it again each time you restart. Use an interactive terminal without pipes or input redirection.

The terminal shows step progress and errors only. Intermediate questions, keywords, candidate examples, and selections are hidden; generated files are still saved under `result/`.

All categories in `./UI` are processed automatically. The API base URL is fixed to `https://api.openai.com/v1`; no endpoint or directory input is required.&#x20;

## Available Modes

| Option                        | Description                             |
| ----------------------------- | --------------------------------------- |
| `--mode InterUI`              | Full pipeline (default)                 |
| `--mode baseline`             | Generate code directly from screenshots |
| `--mode ablation-knowledge`   | Ablation without knowledge retrieval    |
| `--mode ablation-layout-tree` | Ablation without the layout tree        |

Run the command from the project directory. Input is always read from `UI/` under the current working directory, and results are saved under `result/<mode>/` in that same directory. All pages are processed automatically. Input-directory and output-directory arguments are not supported.

## Project and Data Structure

```text
InterUI/
  run.py                       # Recommended entry point
  llm_use_gpt.py                # Terminal API configuration and requests
  generate.py                  # Prompts and generation functions
  common.py                    # Screenshot sorting and file utilities
  main_refactor.py             # Full pipeline
  main_baseline_refactor.py
  main_ablation_RAG.py
  main_ablation_layout_tree.py
  prompt/                      # Prompts for each mode
  arkui_examples/               # Local ArkUI JSON examples
  UI/
    category/
      app/
        page/
          main.jpg             # Initial screenshot
          p1.jpg               # Interaction screenshot
          p2.jpg
  result/                      # Generated results
```

## Outputs

Default output directory: `result/<mode>/<category>/<app>/<page>/`.

## Ablation Experiments

Run without the knowledge retrieval and QA module:

```powershell
python run.py --mode ablation-knowledge
```

Run without the UI layout tree module:

```powershell
python run.py --mode ablation-layout-tree
```
