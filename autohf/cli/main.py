"""AutoHF — CLI powered by Typer.

Patterns from:
- Typer: Subcommands, options, help text
- AutoGluon: Presets as CLI options
- Rich: Tables, panels, progress bars

Commands:
  autohf train "sentiment analysis"
  autohf train "NER" --preset best_quality
  autohf search "question answering" --top 10
  autohf info
  autohf predict ./autohf_output/model_imdb --data test.csv
"""

from __future__ import annotations

import sys
from typing import Optional

# Reconfigure stdout/stderr to UTF-8 to prevent encoding crashes on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(
    name="autohf",
    help="🚀 AutoHF — One-line AutoML: from idea to trained model.",
    add_completion=False,
    rich_markup_mode="rich",
)


@app.command()
def train(
    task: str = typer.Argument(
        ...,
        help="Natural-language task description, e.g. 'sentiment analysis'.",
    ),
    preset: str = typer.Option(
        "medium_quality",
        "--preset", "-P",
        help="AutoHF preset: quick_prototype, medium_quality, high_quality, best_quality, optimize_for_deployment.",
    ),
    time_limit: Optional[int] = typer.Option(
        None,
        "--time-limit", "-t",
        help="Training time limit in seconds (overrides preset).",
    ),
    max_rows: Optional[int] = typer.Option(
        None,
        "--max-rows", "-r",
        help="Maximum dataset rows to load (overrides preset).",
    ),
    presets: Optional[str] = typer.Option(
        None,
        "--ag-presets",
        help="AutoGluon presets: best_quality, high_quality, good_quality, medium_quality.",
    ),
    output_dir: str = typer.Option(
        "./autohf_output",
        "--output", "-o",
        help="Directory to save the trained model.",
    ),
    eval_metric: Optional[str] = typer.Option(
        None,
        "--metric", "-m",
        help="Evaluation metric (e.g., accuracy, f1, roc_auc).",
    ),
    router: str = typer.Option(
        "auto",
        "--router",
        help="Routing strategy for task detection: 'auto' (use Gemma/OpenAI if available), 'keyword' (local keywords), 'openai' (force OpenAI), or 'gemma' (force local Gemma model).",
    ),
) -> None:
    """🏋️ Train a model from a task description.

    Examples:
      autohf train "sentiment analysis"
      autohf train "NER" --preset best_quality
      autohf train "spam detection" --time-limit 600 --metric f1
    """
    from autohf.core.autohf import AutoHF

    try:
        # Build overrides
        overrides = {"output_dir": output_dir, "router": router}
        if time_limit is not None:
            overrides["time_limit"] = time_limit
        if max_rows is not None:
            overrides["max_dataset_rows"] = max_rows
        if presets is not None:
            overrides["presets"] = presets
        if eval_metric is not None:
            overrides["eval_metric"] = eval_metric

        hf = AutoHF.from_preset(preset, **overrides)
        result = hf.train(task)

        console.print()
        console.print("[bold green]✅ Done![/bold green] Model saved to:", result.model_path)

    except ImportError as e:
        console.print(f"\n[bold red]❌ Missing dependency:[/bold red] {e}")
        console.print("[dim]Install training dependencies: pip install autohf[train][/dim]")
        raise typer.Exit(code=1)

    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def search(
    task: str = typer.Argument(
        ...,
        help="Natural-language task description.",
    ),
    top: int = typer.Option(
        10,
        "--top", "-n",
        help="Number of top datasets to show.",
    ),
    show_models: bool = typer.Option(
        False,
        "--models", "-m",
        help="Also show top models for this task.",
    ),
    router: str = typer.Option(
        "auto",
        "--router",
        help="Routing strategy for task detection: 'auto', 'keyword', 'openai', or 'gemma'.",
    ),
) -> None:
    """🔍 Search and rank datasets without training.

    Examples:
      autohf search "sentiment analysis"
      autohf search "question answering" --top 20 --models
    """
    from autohf.core.autohf import AutoHF
    from autohf.agents.dataset_agent import find_models

    try:
        from autohf.core.config import AutoHFConfig
        hf = AutoHF(config=AutoHFConfig(router=router))
        ranked = hf.search(task, top_n=top)

        console.print()
        table = Table(
            title=f"Top {len(ranked)} Datasets for '{task}'",
            show_lines=False,
            border_style="dim",
        )
        table.add_column("#", style="dim", width=3)
        table.add_column("Dataset", style="cyan", max_width=50)
        table.add_column("Downloads", justify="right", style="green")
        table.add_column("Likes", justify="right", style="yellow")
        table.add_column("Score", justify="right", style="bold magenta")

        for i, ds in enumerate(ranked, 1):
            table.add_row(
                str(i),
                ds.dataset_id,
                f"{ds.downloads:,}",
                f"{ds.likes:,}",
                f"{ds.score:.4f}",
            )

        console.print(table)

        # Show models if requested
        if show_models:
            from autohf.agents.task_agent import detect_task
            task_info = detect_task(task, router=router)
            models = find_models(task_info.task_type, limit=10)

            if models:
                console.print()
                model_table = Table(
                    title=f"Top Models for '{task_info.task_type}'",
                    border_style="dim",
                )
                model_table.add_column("#", style="dim", width=3)
                model_table.add_column("Model", style="cyan", max_width=50)
                model_table.add_column("Downloads", justify="right", style="green")
                model_table.add_column("Likes", justify="right", style="yellow")

                for i, m in enumerate(models, 1):
                    model_table.add_row(
                        str(i),
                        m["model_id"],
                        f"{m['downloads']:,}",
                        f"{m['likes']:,}",
                    )

                console.print(model_table)

    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def info() -> None:
    """ℹ️ Show AutoHF info, supported tasks, presets, and config."""
    from autohf.agents.task_agent import list_supported_tasks
    from autohf.core.config import AutoHFConfig, PRESET_CONFIGS

    config = AutoHFConfig()

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]🚀 AutoHF v0.1.0[/bold cyan]\n"
            "[dim]One-line AutoML: from idea to trained model[/dim]\n"
            "[dim]Powered by Hugging Face Hub + AutoGluon[/dim]",
            border_style="cyan",
        )
    )

    # Presets
    console.print()
    preset_table = Table(title="Available Presets", border_style="dim")
    preset_table.add_column("Preset", style="cyan")
    preset_table.add_column("Time Limit", justify="right")
    preset_table.add_column("Max Rows", justify="right")
    preset_table.add_column("AG Presets")

    for name, conf in PRESET_CONFIGS.items():
        preset_table.add_row(
            name,
            f"{conf.get('time_limit', 300)}s",
            f"{conf.get('max_dataset_rows', 50000):,}",
            conf.get("presets", "medium_quality"),
        )

    console.print(preset_table)

    # Supported tasks
    console.print()
    table = Table(title="Supported Tasks", border_style="dim")
    table.add_column("Task Type", style="cyan")
    table.add_column("Label", style="bold")

    for task in list_supported_tasks():
        table.add_row(task["task_type"], task["label"])

    console.print(table)

    # Quick examples
    console.print()
    console.print(
        Panel(
            "[bold]Python API:[/bold]\n"
            '  from autohf import AutoHF\n'
            '  result = AutoHF().train("sentiment analysis")\n\n'
            "[bold]CLI:[/bold]\n"
            '  autohf train "sentiment analysis"\n'
            '  autohf train "NER" --preset best_quality\n'
            '  autohf search "question answering" --models\n'
            '  autohf route "Build an AI that detects fake reviews"\n'
            '  autohf version',
            title="📝 Quick Start",
            border_style="dim",
        )
    )


@app.command()
def route(
    task: str = typer.Argument(
        ...,
        help="Natural-language task description.",
    ),
    router: str = typer.Option(
        "auto",
        "--router",
        help="Routing strategy: 'auto', 'keyword', 'openai', or 'gemma'.",
    ),
) -> None:
    """🎯 Route task and detect ML task type, label, and keywords."""
    from autohf.agents.task_agent import detect_task
    try:
        res = detect_task(task, router=router)
        console.print()
        console.print(f"[bold green]✓[/bold green] Detected Task Type: [bold]{res.task_type}[/bold]")
        console.print(f"  Label: {res.task_label}")
        console.print(f"  Confidence: {res.confidence}")
        console.print(f"  Keywords: {', '.join(res.keywords)}")
        console.print(f"  Problem Type: {res.problem_type}")
    except Exception as e:
        console.print(f"\n[bold red]❌ Error:[/bold red] {e}")
        raise typer.Exit(code=1)


@app.command()
def chat(
    prompt: Optional[str] = typer.Argument(
        None,
        help="Single-shot prompt to pass to Gemma. If empty, starts interactive REPL.",
    ),
    model: str = typer.Option(
        "google/gemma-4-E2B-it",
        "--model", "-m",
        help="Gemma model ID to use for chat.",
    ),
    system_prompt: str = typer.Option(
        "You are Gemma, a helpful AI assistant developed by Google.",
        "--system", "-s",
        help="System prompt to guide the conversation.",
    ),
) -> None:
    """💬 Start a chat session or run a single prompt with a local Gemma model.

    Examples:
      autohf chat "Explain quantum computing simply"
      autohf chat --model google/gemma-4-E2B-it
    """
    from autohf.agents.chat_agent import GemmaChatAgent

    console.print()
    if prompt:
        # Single-shot prompt
        try:
            agent = GemmaChatAgent(model_id=model, system_prompt=system_prompt)
            with console.status(f"[bold green]Loading local Gemma model '{model}'...[/bold green]", spinner="dots"):
                agent.load()
            with console.status("[bold yellow]Gemma is thinking...[/bold yellow]", spinner="dots"):
                response = agent.generate(prompt, [])
            console.print(Panel(response, title="[bold green]Gemma[/bold green]", border_style="green"))
        except Exception as e:
            console.print(f"[bold red]❌ Error:[/bold red] {e}")
            raise typer.Exit(code=1)
        return

    # Interactive REPL
    console.print(
        Panel.fit(
            "[bold cyan]💬 AutoHF Gemma Chat[/bold cyan]\n"
            f"[dim]Model: {model}[/dim]\n"
            "[dim]Commands: 'exit' or 'quit' to end session | 'clear' to clear history[/dim]",
            border_style="cyan",
        )
    )

    try:
        agent = GemmaChatAgent(model_id=model, system_prompt=system_prompt)
        with console.status(f"[bold green]Loading local Gemma model '{model}'...[/bold green]", spinner="dots"):
            agent.load()
        console.print("[bold green]✓ Model loaded successfully![/bold green] Let's chat!\n")
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load model:[/bold red] {e}")
        raise typer.Exit(code=1)

    chat_history = []
    while True:
        try:
            user_input = console.input("[bold cyan]You > [/bold cyan]").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break
            if user_input.lower() == "clear":
                chat_history.clear()
                console.print("[dim]Chat history cleared.[/dim]\n")
                continue

            with console.status("[bold yellow]Gemma is thinking...[/bold yellow]", spinner="dots"):
                response = agent.generate(user_input, chat_history)

            console.print()
            console.print(Panel(response, title="[bold green]Gemma[/bold green]", border_style="green"))
            console.print()

            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            console.print("\n[dim]Goodbye![/dim]")
            break
        except Exception as e:
            console.print(f"\n[bold red]❌ Error: {e}[/bold red]\n")


@app.command()
def version() -> None:
    """ℹ️ Show AutoHF version."""
    console.print("[bold cyan]AutoHF v1.0.0[/bold cyan]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()

