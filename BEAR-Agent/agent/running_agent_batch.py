"""Run BEAR task categories, or inspect the discovered calls with --dry-run."""

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = PROJECT_ROOT / "tasks/z_agent/trajectory"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs/z_agent"
CATEGORY = "trajectory"


def iter_tasks(tasks_root, outputs_root):
    """Yield task directories and their category output directories in order."""
    for category in sorted(Path(tasks_root).iterdir()):
        if category.is_dir():
            for entry in sorted(category.iterdir()):
                if entry.is_dir():
                    yield entry, Path(outputs_root) / category.name


def discover_and_run_tasks(tasks_root=None, outputs_root=None, task_type=None):
    """Run each discovered task, continuing after individual failures."""
    from main import run_agent

    tasks_root = TASKS_ROOT if tasks_root is None else tasks_root
    outputs_root = OUTPUTS_ROOT if outputs_root is None else outputs_root
    task_type = CATEGORY if task_type is None else task_type
    failures = 0
    for entry, output in iter_tasks(tasks_root, outputs_root):
        print(f"Running {entry.parent.name}/{entry.name}")
        try:
            run_agent(str(entry), str(output), task_type=task_type)
            print(f"  ✓ Completed: {entry.name}")
        except Exception as exc:
            failures += 1
            print(f"  ✗ Failed {entry.name}: {exc}")
    return failures


def generate_command_strings(tasks_root=None, outputs_root=None, task_type=None):
    """Return valid Python calls without importing the model or vision clients."""
    tasks_root = TASKS_ROOT if tasks_root is None else tasks_root
    outputs_root = OUTPUTS_ROOT if outputs_root is None else outputs_root
    task_type = CATEGORY if task_type is None else task_type
    return [
        f"run_agent({str(entry)!r}, {str(output)!r}, task_type={task_type!r})"
        for entry, output in iter_tasks(tasks_root, outputs_root)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-root", type=Path, default=TASKS_ROOT)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT)
    parser.add_argument("--task-type", default=CATEGORY)
    parser.add_argument("--dry-run", action="store_true", help="Print calls without running agents")
    args = parser.parse_args()
    if not args.tasks_root.is_dir():
        parser.error(f"Task directory does not exist: {args.tasks_root}")
    if args.dry_run:
        for command in generate_command_strings(args.tasks_root, args.outputs_root, args.task_type):
            print(command)
        return 0
    return int(discover_and_run_tasks(args.tasks_root, args.outputs_root, args.task_type) > 0)


if __name__ == "__main__":
    raise SystemExit(main())
