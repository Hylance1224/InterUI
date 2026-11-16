import os
import re


def get_interaction_images(page_dir):
    """
    获取 page 文件夹中的 p1.jpg、p2.jpg、p3.jpg...
    并按照数字顺序排列
    """
    images = []

    for filename in os.listdir(page_dir):
        match = re.fullmatch(
            r"p(\d+)\.(jpg|jpeg|png)",
            filename,
            re.IGNORECASE
        )

        if match:
            index = int(match.group(1))
            full_path = os.path.join(page_dir, filename)
            images.append((index, full_path))

    images.sort(key=lambda x: x[0])

    return [path for _, path in images]


def save_text(content, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def get_layout_tree_transitions(page_dir, output_dir, *, content=None):
    """Read a multi-predecessor DAG and return all edges in topological order."""
    from pathlib import Path

    page = Path(page_dir)
    images = get_interaction_images(page_dir)
    if not images:
        return []
    path = Path(output_dir) / "transitions.txt"
    if content is None and not path.is_file():
        raise ValueError(f"Missing transitions file: {path}")
    if content is None:
        content = path.read_text(encoding="utf-8-sig")
    predecessors = {}
    source = None
    for number, raw_line in enumerate(content.splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if not separator or key not in {"source", "target"} or not value:
            raise ValueError(f"{path}:{number}: expected source = filename or target = filename")
        if Path(value).name != value or "/" in value or "\\" in value:
            raise ValueError(f"{path}:{number}: use an image filename, not a path")
        if key == "source":
            if source is not None:
                raise ValueError(f"{path}:{number}: previous source has no target")
            source = value
        else:
            if source is None:
                raise ValueError(f"{path}:{number}: target has no source")
            parents = predecessors.setdefault(value, [])
            if source in parents:
                raise ValueError(f"{path}:{number}: duplicate transition {source} -> {value}")
            if source == value:
                raise ValueError(f"{path}:{number}: self-link {source} -> {value}")
            parents.append(source)
            source = None
    if source is not None:
        raise ValueError(f"{path}: last source has no target")

    targets = [Path(image).name for image in images]
    if set(predecessors) != set(targets):
        missing = sorted(set(targets) - set(predecessors))
        extra = sorted(set(predecessors) - set(targets))
        raise ValueError(f"{path}: targets must match interaction images; missing={missing}, extra={extra}")
    for target, parents in predecessors.items():
        for predecessor in parents:
            if predecessor not in {"main.jpg", *targets} or not (page / predecessor).is_file():
                raise ValueError(f"{path}: invalid predecessor {predecessor} for {target}")
    if not (page / "main.jpg").is_file():
        raise ValueError(f"{path}: main.jpg is missing")

    # Stable topological traversal: numeric image order breaks ties.
    visited = {"main.jpg"}
    pending = list(targets)
    ordered = []
    while pending:
        target = next((name for name in pending if all(parent in visited for parent in predecessors[name])), None)
        if target is None:
            raise ValueError(f"{path}: cycle in transitions involving {pending}")
        # Keep every incoming edge, including different sources for one target.
        ordered.extend((str(page / parent), str(page / target))
                       for parent in predecessors[target])
        visited.add(target)
        pending.remove(target)
    return ordered
