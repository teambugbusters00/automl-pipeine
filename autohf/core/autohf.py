"""AutoHF — Main orchestrator.

Patterns extracted from:
- LangGraph: StateGraph with typed state transitions
- AutoGluon: TabularPredictor's fit() orchestration
- AutoGen: Agent collaboration pattern
- OpenHands: Autonomous task execution with retry logic
"""

from __future__ import annotations

import sys
import time
from typing import Optional

# Reconfigure stdout/stderr to UTF-8 to prevent encoding crashes on Windows
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from autohf.core.config import (
    AutoHFConfig,
    DatasetCandidate,
    DatasetProfile,
    PipelineState,
    TaskInfo,
    TrainResult,
    PRESET_CONFIGS,
)
from autohf.agents.task_agent import TaskAgent, detect_task
from autohf.agents.dataset_agent import (
    DatasetAgent,
    find_datasets,
    find_models,
    load_dataset_as_dataframe,
    profile_dataset,
)
from autohf.ranking.dataset_ranker import rank_datasets


console = Console()


class AutoHF:
    """One-line AutoML: from idea to trained model.

    Orchestrates the full pipeline using agent collaboration pattern
    inspired by AutoGen and LangGraph's state machine approach.

    Pipeline States (LangGraph-inspired):
        IDLE → DETECTING_TASK → SEARCHING_DATASETS → RANKING_DATASETS →
        LOADING_DATASET → PROFILING_DATASET → TRAINING → EVALUATING → COMPLETED

    Usage::

        from autohf import AutoHF

        # Quick prototype
        result = AutoHF().train("sentiment analysis")

        # With preset (AutoGluon-inspired)
        result = AutoHF.from_preset("best_quality").train("NER")

        # Custom config
        config = AutoHFConfig(time_limit=600, presets="high_quality")
        result = AutoHF(config=config).train("spam detection")
    """

    def __init__(self, config: Optional[AutoHFConfig] = None) -> None:
        self.config = config or AutoHFConfig()
        self._state = PipelineState.IDLE
        self._task_agent = TaskAgent(router=self.config.router)
        self._dataset_agent = DatasetAgent(self.config)
        self._pipeline_log: list[dict] = []

        logger.info("AutoHF initialised (preset='{}')", self.config.preset)

    @classmethod
    def from_preset(cls, preset: str, **overrides) -> "AutoHF":
        """Create AutoHF from a named preset.

        Inspired by AutoGluon's presets system.

        Presets:
          - quick_prototype: Fast prototyping (~60s)
          - medium_quality: Default balance
          - high_quality: Better results, longer training
          - best_quality: Maximum quality
          - optimize_for_deployment: Small model, fast inference
        """
        config = AutoHFConfig.from_preset(preset, **overrides)
        return cls(config=config)

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def train(self, task_description: str, **kwargs) -> TrainResult:
        """Run the full AutoHF pipeline.

        Follows OpenHands' autonomous task execution pattern:
        each step has retry logic and clear error handling.

        Args:
            task_description: Natural-language description of the ML task.
            **kwargs: Override config values.

        Returns:
            TrainResult with model path, metrics, leaderboard, etc.
        """
        pipeline_start = time.time()

        # Apply any overrides
        if kwargs:
            config_data = self.config.model_dump()
            config_data.update(kwargs)
            self.config = AutoHFConfig(**config_data)

        console.print()
        console.print(
            Panel.fit(
                f"[bold cyan]🚀 AutoHF Pipeline[/bold cyan]\n"
                f"[dim]Task: {task_description}[/dim]\n"
                f"[dim]Preset: {self.config.preset} | "
                f"Time: {self.config.time_limit}s | "
                f"Max rows: {self.config.max_dataset_rows:,}[/dim]",
                border_style="cyan",
            )
        )
        console.print()

        try:
            # Step 1: Detect task
            self._set_state(PipelineState.DETECTING_TASK)
            task_info = self._detect_task(task_description)

            # Step 2: Find datasets
            self._set_state(PipelineState.SEARCHING_DATASETS)
            candidates = self._find_datasets(task_info)

            # Step 3: Rank datasets
            self._set_state(PipelineState.RANKING_DATASETS)
            ranked = self._rank_datasets(candidates, task_info.keywords, task_description)

            # Step 4: Load best dataset (with retry — OpenHands pattern)
            self._set_state(PipelineState.LOADING_DATASET)
            df, text_col, label_col, dataset_id = self._load_best_dataset(ranked)

            # Step 5: Profile dataset
            self._set_state(PipelineState.PROFILING_DATASET)
            ds_profile = profile_dataset(df, dataset_id, text_col, label_col)
            self._display_profile(ds_profile)

            # Step 6: Train
            self._set_state(PipelineState.TRAINING)
            result = self._train_model(
                df, text_col, label_col, task_info, dataset_id, ds_profile
            )

            # Step 7: Display results
            self._set_state(PipelineState.COMPLETED)
            total_time = round(time.time() - pipeline_start, 2)
            self._display_results(result, total_time)

            return result

        except Exception as e:
            self._set_state(PipelineState.FAILED)
            logger.error("Pipeline failed: {}", e)
            raise

    # ------------------------------------------------------------------
    # Search-only mode
    # ------------------------------------------------------------------

    def search(self, task_description: str, top_n: int = 10) -> list[DatasetCandidate]:
        """Discover and rank datasets without training."""
        task_info = self._detect_task(task_description)
        candidates = self._find_datasets(task_info)
        ranked = self._rank_datasets(candidates, task_info.keywords, task_description, show_table=False)
        return ranked[:top_n]

    # ------------------------------------------------------------------
    # Pipeline steps (each mirrors a LangGraph node)
    # ------------------------------------------------------------------

    def _set_state(self, state: PipelineState) -> None:
        """Track pipeline state transitions (LangGraph pattern)."""
        old = self._state
        self._state = state
        self._pipeline_log.append({
            "from": old.value,
            "to": state.value,
            "timestamp": time.time(),
        })
        logger.debug("State: {} → {}", old.value, state.value)

    def _detect_task(self, description: str) -> TaskInfo:
        """Step 1: Detect the ML task type."""
        with console.status("[bold green]🔍 Detecting task type...", spinner="dots"):
            task_info = self._task_agent(description)

        emoji = "✓" if task_info.confidence >= 0.7 else "⚠"
        color = "green" if task_info.confidence >= 0.7 else "yellow"
        console.print(
            f"  [{color}]{emoji}[/{color}] Task detected: "
            f"[bold]{task_info.task_label}[/bold] "
            f"[dim]({task_info.task_type})[/dim] "
            f"[dim]confidence={task_info.confidence}[/dim]"
        )
        if task_info.confidence < 0.7:
            console.print(
                f"    [dim yellow]Tip: Be more specific. Try 'sentiment analysis' instead of '{description}'[/dim yellow]"
            )
        return task_info

    def _find_datasets(self, task_info: TaskInfo) -> list[DatasetCandidate]:
        """Step 2: Search HF Hub (multi-strategy search)."""
        with console.status("[bold green]📦 Searching Hugging Face Hub...", spinner="dots"):
            candidates = self._dataset_agent.search(
                task_type=task_info.task_type,
                keywords=task_info.keywords,
            )

        if not candidates:
            raise RuntimeError(
                f"No datasets found for task '{task_info.task_type}'. "
                "Try a different task description."
            )

        console.print(
            f"  [green]✓[/green] Found [bold]{len(candidates)}[/bold] candidate datasets"
        )

        # Also show top models (Phase 2 prep)
        try:
            models = find_models(task_info.task_type, limit=3)
            if models:
                model_names = ", ".join(m["model_id"].split("/")[-1] for m in models[:3])
                console.print(
                    f"  [dim]📊 Top models for this task: {model_names}[/dim]"
                )
        except Exception:
            pass

        return candidates

    def _rank_datasets(
        self,
        candidates: list[DatasetCandidate],
        keywords: list[str],
        problem_statement: Optional[str] = None,
        show_table: bool = True,
    ) -> list[DatasetCandidate]:
        """Step 3: Rank datasets by quality signals or semantically."""
        with console.status("[bold green]📊 Ranking datasets...", spinner="dots"):
            use_semantic = False
            if problem_statement:
                try:
                    from autohf.ranking import SemanticRanker
                    
                    ranker = SemanticRanker()
                    ranked = ranker.rank(candidates, problem_statement, keywords)
                    use_semantic = True
                except (ImportError, Exception) as e:
                    logger.debug("Failed to use semantic ranking: {}. Falling back to keyword ranker.", e)
                    ranked = rank_datasets(candidates, keywords)
            else:
                ranked = rank_datasets(candidates, keywords)

        if show_table:
            ranking_title = "🏆 Top Datasets (Semantic)" if use_semantic else "🏆 Top Datasets"
            # Display top-5 in a rich table
            table = Table(title=ranking_title, show_lines=False, border_style="dim")
            table.add_column("#", style="dim", width=3)
            table.add_column("Dataset", style="cyan", max_width=45)
            table.add_column("Downloads", justify="right", style="green")
            table.add_column("Likes", justify="right", style="yellow")
            table.add_column("Score", justify="right", style="bold magenta")

            for i, ds in enumerate(ranked[:5], 1):
                table.add_row(
                    str(i),
                    ds.dataset_id,
                    f"{ds.downloads:,}",
                    f"{ds.likes:,}",
                    f"{ds.score:.4f}",
                )

            console.print(table)
        return ranked

    def _load_best_dataset(
        self,
        ranked: list[DatasetCandidate],
    ) -> tuple[pd.DataFrame, str, str, str]:
        """Step 4: Load best dataset with retry (OpenHands pattern).

        Tries top-5 datasets, falling back if one fails.
        """
        max_attempts = min(5, len(ranked))

        for i, candidate in enumerate(ranked[:max_attempts]):
            try:
                console.print(
                    f"  [yellow]⏳[/yellow] Loading: "
                    f"[bold]{candidate.dataset_id}[/bold]"
                    f" [dim](attempt {i+1}/{max_attempts})[/dim]"
                )

                df, text_col, label_col = self._dataset_agent.load(candidate.dataset_id)

                console.print(
                    f"  [green]✓[/green] Loaded [bold]{len(df):,}[/bold] rows "
                    f"[dim](text='{text_col}', label='{label_col}')[/dim]"
                )

                return df, text_col, label_col, candidate.dataset_id

            except Exception as e:
                logger.warning(
                    "Failed to load '{}' (attempt {}/{}): {}",
                    candidate.dataset_id, i + 1, max_attempts, e,
                )
                console.print(
                    f"  [red]✗[/red] Failed: [dim]{str(e)[:80]}[/dim]"
                )
                continue

        raise RuntimeError(
            f"Could not load any of the top-{max_attempts} ranked datasets. "
            "The datasets may be gated, too large, or have incompatible formats."
        )

    def _display_profile(self, profile: DatasetProfile) -> None:
        """Display dataset profile with class distribution."""
        # Build distribution string
        dist_str = ""
        if profile.label_distribution:
            top_classes = list(profile.label_distribution.items())[:5]
            dist_str = " | ".join(f"{k}: {v:,}" for k, v in top_classes)

        console.print()
        console.print(
            Panel(
                f"[bold]Dataset:[/bold] {profile.dataset_id}\n"
                f"[bold]Rows:[/bold] {profile.num_rows:,}\n"
                f"[bold]Classes:[/bold] {profile.num_classes}\n"
                f"[bold]Avg text length:[/bold] {profile.avg_text_length:.0f} chars\n"
                f"[bold]Text col:[/bold] {profile.text_column}\n"
                f"[bold]Label col:[/bold] {profile.label_column}\n"
                f"[bold]Distribution:[/bold] {dist_str}",
                title="📋 Dataset Profile",
                border_style="blue",
            )
        )

    def _train_model(
        self,
        df: pd.DataFrame,
        text_col: str,
        label_col: str,
        task_info: TaskInfo,
        dataset_id: str,
        profile: DatasetProfile,
    ) -> TrainResult:
        """Step 5: Train using AutoGluon."""
        from autohf.training.autogluon_trainer import train_model

        console.print()
        console.print(
            f"  [yellow]🏋️ Training model...[/yellow]\n"
            f"    [dim]time_limit={self.config.time_limit}s | "
            f"presets='{self.config.presets}' | "
            f"eval_metric='{self.config.eval_metric or 'auto'}'[/dim]"
        )

        result = train_model(
            df=df,
            text_column=text_col,
            label_column=label_col,
            task_type=task_info.task_type,
            dataset_id=dataset_id,
            config=self.config,
            dataset_profile=profile,
        )

        return result

    def _display_results(self, result: TrainResult, total_time: float) -> None:
        """Display final results with leaderboard."""
        console.print()

        # Metrics
        metrics_str = "\n".join(
            f"  [bold]{k}:[/bold] {v}" for k, v in result.metrics.items()
        )

        console.print(
            Panel(
                f"[bold green]Task:[/bold green] {result.task_type}\n"
                f"[bold green]Dataset:[/bold green] {result.dataset_id}\n"
                f"[bold green]Best model:[/bold green] {result.best_model_name or 'N/A'}\n"
                f"[bold green]Models trained:[/bold green] {result.num_models_trained}\n"
                f"[bold green]Training time:[/bold green] {result.training_time:.1f}s\n"
                f"[bold green]Total pipeline time:[/bold green] {total_time:.1f}s\n"
                f"[bold green]Model saved to:[/bold green] {result.model_path}\n\n"
                f"[bold cyan]📈 Metrics:[/bold cyan]\n{metrics_str}",
                title="🎉 Training Complete",
                border_style="green",
            )
        )

        if result.leaderboard:
            console.print()
            console.print(
                Panel(
                    result.leaderboard,
                    title="🏆 AutoGluon Leaderboard",
                    border_style="yellow",
                )
            )

        # Show quick-use code
        console.print()
        console.print(
            Panel(
                f'[dim]# Load and predict with your model[/dim]\n'
                f'from autohf.training.autogluon_trainer import load_predictor\n'
                f'predictor = load_predictor("{result.model_path}")\n'
                f'predictions = predictor.predict(your_data)',
                title="📝 Quick Start",
                border_style="dim",
            )
        )
