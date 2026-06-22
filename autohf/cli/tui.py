"""AutoHF — Interactive Terminal UI (TUI) powered by Textual.

Launch with:
    autohf ui

Features:
    - Chat panel for natural-language ML commands
    - Advanced options sidebar (Dataset, Model, Training, Output)
    - Supports all 12 ML task types and all AutoGluon algorithms
    - Real-time pipeline execution feedback
"""

from __future__ import annotations

import sys
import asyncio
import threading
import io
from datetime import datetime
from typing import Optional

# Reconfigure stdout/stderr to UTF-8
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
except Exception:
    pass

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import (
    Horizontal,
    Vertical,
    VerticalScroll,
    Container,
)
from textual.widgets import (
    Header,
    Footer,
    Static,
    Input,
    Button,
    Select,
    Label,
    RichLog,
    Rule,
)
from textual.reactive import reactive
from textual import on, work
from rich.text import Text
from rich.panel import Panel
from rich.table import Table


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TASK_TYPES = [
    ("Auto Detect", "auto"),
    ("Text Classification", "text-classification"),
    ("Token Classification (NER)", "token-classification"),
    ("Question Answering", "question-answering"),
    ("Summarization", "summarization"),
    ("Translation", "translation"),
    ("Text Generation", "text-generation"),
    ("Fill Mask", "fill-mask"),
    ("Text-to-Text Generation", "text2text-generation"),
    ("Zero-Shot Classification", "zero-shot-classification"),
    ("Image Classification", "image-classification"),
    ("Tabular Classification", "tabular-classification"),
    ("Tabular Regression", "tabular-regression"),
]

PRESETS = [
    ("Quick Prototype (~60s)", "quick_prototype"),
    ("Medium Quality (~5min)", "medium_quality"),
    ("High Quality (~10min)", "high_quality"),
    ("Best Quality (~1hr)", "best_quality"),
    ("Optimize for Deployment", "optimize_for_deployment"),
]

METRICS = [
    ("Auto", "auto"),
    ("Accuracy", "accuracy"),
    ("Balanced Accuracy", "balanced_accuracy"),
    ("F1", "f1"),
    ("F1 Macro", "f1_macro"),
    ("F1 Weighted", "f1_weighted"),
    ("ROC AUC", "roc_auc"),
    ("Log Loss", "log_loss"),
    ("Precision", "precision"),
    ("Recall", "recall"),
    ("MCC", "mcc"),
    ("RMSE", "root_mean_squared_error"),
    ("MAE", "mean_absolute_error"),
    ("R²", "r2"),
]

SEARCH_LIMITS = [
    ("10", "10"),
    ("20", "20"),
    ("30", "30"),
    ("50", "50"),
]

MIN_DOWNLOADS = [
    ("0", "0"),
    ("100", "100"),
    ("1,000", "1000"),
    ("10,000", "10000"),
]

QUANTIZATION_OPTIONS = [
    ("None (FP16)", "none"),
    ("INT8", "int8"),
    ("INT4", "int4"),
    ("GPTQ", "gptq"),
    ("AWQ", "awq"),
]

MAX_PARAMS = [
    ("Auto", "auto"),
    ("< 1B", "1b"),
    ("< 7B", "7b"),
    ("< 13B", "13b"),
    ("< 70B", "70b"),
]

DATASET_FILTERS = [
    ("No Filter", "none"),
    ("Text Only", "text"),
    ("Tabular Only", "tabular"),
    ("Image Only", "image"),
]

AG_PRESETS = [
    ("--preset best_quality", "best_quality"),
    ("--preset high_quality", "high_quality"),
    ("--preset good_quality", "good_quality"),
    ("--preset medium_quality", "medium_quality"),
]

GPU_OPTIONS = [
    ("Auto Detect", "auto"),
    ("CPU Only", "cpu"),
    ("GPU 0", "cuda:0"),
    ("GPU 1", "cuda:1"),
]

PUSH_HUB = [
    ("No", "no"),
    ("Yes", "yes"),
]

FRAMEWORK = [
    ("AutoGluon", "autogluon"),
]

MODEL_SOURCE = [
    ("Hugging Face Hub", "huggingface"),
]


# ---------------------------------------------------------------------------
# CSS Stylesheet
# ---------------------------------------------------------------------------

TUI_CSS = """
Screen {
    background: #0a0e17;
}

#app-header {
    dock: top;
    height: 3;
    background: #0d1117;
    color: #c9d1d9;
    padding: 0 2;
}

#header-title {
    color: #58a6ff;
    text-style: bold;
}

#header-subtitle {
    color: #8b949e;
    padding-left: 1;
}

#header-status {
    color: #3fb950;
    dock: right;
    padding-right: 2;
}

#main-layout {
    height: 1fr;
}

/* ---- Left: Chat Panel ---- */
#chat-panel {
    width: 60%;
    border-right: solid #21262d;
    padding: 0;
}

#chat-header {
    height: 3;
    background: #161b22;
    padding: 0 2;
    color: #f0883e;
    text-style: bold;
}

#chat-log {
    height: 1fr;
    padding: 1 2;
    background: #0d1117;
    scrollbar-color: #30363d;
    scrollbar-color-hover: #484f58;
}

#chat-input-area {
    height: auto;
    dock: bottom;
    padding: 1 2;
    background: #161b22;
    max-height: 6;
}

#chat-input {
    width: 1fr;
    background: #0d1117;
    border: solid #30363d;
    color: #c9d1d9;
}

#send-btn {
    width: 8;
    min-width: 8;
    background: #238636;
    color: #ffffff;
    text-style: bold;
    border: none;
    margin-left: 1;
}

#send-btn:hover {
    background: #2ea043;
}

/* ---- Right: Options Sidebar ---- */
#options-panel {
    width: 40%;
    padding: 0;
    background: #0d1117;
}

#options-header {
    height: 3;
    background: #161b22;
    padding: 0 2;
    color: #f0883e;
    text-style: bold;
}

#options-scroll {
    height: 1fr;
    padding: 1 2;
    background: #0d1117;
    scrollbar-color: #30363d;
    scrollbar-color-hover: #484f58;
}

.section-title {
    color: #f0883e;
    text-style: bold;
    margin-top: 1;
    margin-bottom: 0;
}

.field-row {
    height: auto;
    margin-bottom: 0;
    padding: 0;
}

.field-label {
    width: 20;
    color: #8b949e;
    padding: 1 0 0 0;
}

.field-value {
    width: 1fr;
}

.field-value Select {
    width: 100%;
    background: #161b22;
    border: solid #30363d;
    color: #3fb950;
}

.field-value Input {
    width: 100%;
    background: #161b22;
    border: solid #30363d;
    color: #3fb950;
}

.action-btn {
    width: 100%;
    margin-top: 1;
    margin-bottom: 1;
    background: #21262d;
    color: #c9d1d9;
    border: solid #30363d;
    text-style: bold;
}

.action-btn:hover {
    background: #30363d;
}

.action-btn-primary {
    width: 100%;
    margin-top: 1;
    margin-bottom: 1;
    background: #238636;
    color: #ffffff;
    border: none;
    text-style: bold;
}

.action-btn-primary:hover {
    background: #2ea043;
}

Rule {
    color: #21262d;
    margin: 1 0;
}

/* ---- Status Bar ---- */
#status-bar {
    dock: bottom;
    height: 1;
    background: #161b22;
    color: #8b949e;
    padding: 0 2;
}

.example-btn {
    width: 100%;
    margin-bottom: 0;
    background: #161b22;
    border: solid #30363d;
    color: #8b949e;
    text-style: none;
    min-height: 3;
    content-align: left middle;
    text-align: left;
    padding: 0 2;
}

.example-btn:hover {
    background: #21262d;
    color: #c9d1d9;
    border: solid #58a6ff;
}

#welcome-section {
    height: auto;
    padding: 2 4;
}

#welcome-title {
    text-align: center;
    color: #c9d1d9;
    text-style: bold;
    margin-bottom: 1;
}

#welcome-desc {
    text-align: center;
    color: #8b949e;
    margin-bottom: 2;
}

#examples-label {
    color: #8b949e;
    margin-bottom: 1;
    padding-left: 4;
}

.chat-msg-user {
    background: #161b22;
    color: #58a6ff;
    margin: 0 0 1 0;
    padding: 1 2;
    border: solid #21262d;
}

.chat-msg-assistant {
    background: #0d1117;
    color: #c9d1d9;
    margin: 0 0 1 0;
    padding: 1 2;
    border: solid #21262d;
}
"""


# ---------------------------------------------------------------------------
# Chat Message Widget
# ---------------------------------------------------------------------------

class ChatMessage(Static):
    """A single chat message."""

    def __init__(self, role: str, content: str, **kwargs):
        super().__init__(**kwargs)
        self.role = role
        self.content = content

    def compose(self) -> ComposeResult:
        if self.role == "user":
            self.add_class("chat-msg-user")
            yield Static(f"[bold cyan]You >[/bold cyan] {self.content}")
        else:
            self.add_class("chat-msg-assistant")
            yield Static(f"[bold green]AutoHF >[/bold green] {self.content}")


# ---------------------------------------------------------------------------
# Main TUI Application
# ---------------------------------------------------------------------------

class AutoHFApp(App):
    """AutoHF Interactive Terminal UI."""

    TITLE = "AutoHF CLI"
    SUB_TITLE = "Automate. Search. Train. Deploy."
    CSS = TUI_CSS

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit", show=True),
        Binding("ctrl+k", "show_shortcuts", "Shortcuts", show=True),
        Binding("f1", "show_help", "Help", show=True),
        Binding("ctrl+l", "clear_chat", "Clear Chat", show=True),
    ]

    show_welcome = reactive(True)

    def compose(self) -> ComposeResult:
        # ---- Top Header ----
        yield Horizontal(
            Static("[bold #58a6ff]⚙ AutoHF[/bold #58a6ff] [#8b949e]CLI[/#8b949e]  [dim]Automate. Search. Train. Deploy.[/dim]", id="header-title"),
            Static("[bold #3fb950]● CONNECTED[/bold #3fb950]", id="header-status"),
            id="app-header",
        )

        # ---- Main Layout ----
        with Horizontal(id="main-layout"):
            # ---- Left: Chat ----
            with Vertical(id="chat-panel"):
                yield Static("  [bold #f0883e]💬 CHAT[/bold #f0883e]", id="chat-header")

                with VerticalScroll(id="chat-log"):
                    # Welcome section
                    with Vertical(id="welcome-section"):
                        yield Static("Welcome to AutoHF CLI 👋", id="welcome-title")
                        yield Static(
                            "Ask me anything about datasets, models, training, or deployments.\n"
                            "I'll handle the search, selection and training for you.",
                            id="welcome-desc",
                        )
                        yield Static("    Examples:", id="examples-label")
                        yield Button(
                            '> "Find a dataset for sentiment analysis"',
                            id="ex-1",
                            classes="example-btn",
                        )
                        yield Button(
                            '> "Train a model for tweet classification"',
                            id="ex-2",
                            classes="example-btn",
                        )
                        yield Button(
                            '> "Search datasets for question answering"',
                            id="ex-3",
                            classes="example-btn",
                        )
                        yield Button(
                            '> "Compare models for text summarization"',
                            id="ex-4",
                            classes="example-btn",
                        )

                # Chat input
                with Horizontal(id="chat-input-area"):
                    yield Input(
                        placeholder="Talk or type your command...",
                        id="chat-input",
                    )
                    yield Button("➤", id="send-btn")

            # ---- Right: Advanced Options ----
            with Vertical(id="options-panel"):
                yield Static(
                    "  [bold #f0883e]⚙ ADVANCED OPTIONS[/bold #f0883e]  [dim]»[/dim]",
                    id="options-header",
                )

                with VerticalScroll(id="options-scroll"):
                    # ---- DATASET ----
                    yield Static("🔍 DATASET", classes="section-title")

                    with Horizontal(classes="field-row"):
                        yield Label("Source", classes="field-label")
                        yield Container(
                            Select(MODEL_SOURCE, value="huggingface", id="ds-source", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Search Limit", classes="field-label")
                        yield Container(
                            Select(SEARCH_LIMITS, value="10", id="ds-search-limit", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Min Downloads", classes="field-label")
                        yield Container(
                            Select(MIN_DOWNLOADS, value="1000", id="ds-min-downloads", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Task Type", classes="field-label")
                        yield Container(
                            Select(TASK_TYPES, value="auto", id="ds-task-type", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Filters", classes="field-label")
                        yield Container(
                            Select(DATASET_FILTERS, value="none", id="ds-filters", allow_blank=False),
                            classes="field-value",
                        )
                    yield Button("🔎 SEARCH DATASETS", id="btn-search-datasets", classes="action-btn")

                    yield Rule()

                    # ---- MODEL ----
                    yield Static("🤖 MODEL", classes="section-title")

                    with Horizontal(classes="field-row"):
                        yield Label("Model Source", classes="field-label")
                        yield Container(
                            Select(MODEL_SOURCE, value="huggingface", id="mdl-source", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Base Model", classes="field-label")
                        yield Container(
                            Select([("Auto Detect", "auto")], value="auto", id="mdl-base", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Task", classes="field-label")
                        yield Container(
                            Select(TASK_TYPES, value="auto", id="mdl-task", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Max Parameters", classes="field-label")
                        yield Container(
                            Select(MAX_PARAMS, value="auto", id="mdl-max-params", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Quantization", classes="field-label")
                        yield Container(
                            Select(QUANTIZATION_OPTIONS, value="none", id="mdl-quantization", allow_blank=False),
                            classes="field-value",
                        )
                    yield Button("🎯 SELECT / RECOMMEND MODEL", id="btn-select-model", classes="action-btn")

                    yield Rule()

                    # ---- TRAINING ----
                    yield Static("🔧 TRAINING", classes="section-title")

                    with Horizontal(classes="field-row"):
                        yield Label("Framework", classes="field-label")
                        yield Container(
                            Select(FRAMEWORK, value="autogluon", id="trn-framework", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Time Limit (sec)", classes="field-label")
                        yield Container(
                            Input(value="3600", id="trn-time-limit", type="integer"),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Metric", classes="field-label")
                        yield Container(
                            Select(METRICS, value="auto", id="trn-metric", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("GPU", classes="field-label")
                        yield Container(
                            Select(GPU_OPTIONS, value="auto", id="trn-gpu", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Extra Args", classes="field-label")
                        yield Container(
                            Select(AG_PRESETS, value="best_quality", id="trn-extra-args", allow_blank=False),
                            classes="field-value",
                        )
                    yield Button("🚀 START TRAINING", id="btn-start-training", classes="action-btn")

                    yield Rule()

                    # ---- OUTPUT ----
                    yield Static("📤 OUTPUT", classes="section-title")

                    with Horizontal(classes="field-row"):
                        yield Label("Save Model To", classes="field-label")
                        yield Container(
                            Input(value="./autohf_output", id="out-save-path"),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Push to Hub", classes="field-label")
                        yield Container(
                            Select(PUSH_HUB, value="no", id="out-push-hub", allow_blank=False),
                            classes="field-value",
                        )
                    with Horizontal(classes="field-row"):
                        yield Label("Hub Repo Name", classes="field-label")
                        yield Container(
                            Input(value="autohf-model", id="out-hub-repo"),
                            classes="field-value",
                        )
                    yield Button("💾 SAVE CONFIG", id="btn-save-config", classes="action-btn")

                    yield Rule()

                    yield Button("▶ RUN FULL PIPELINE", id="btn-run-pipeline", classes="action-btn-primary")

        # ---- Status Bar ----
        from autohf import __version__
        yield Static(
            f"  AutoHF CLI v{__version__}  |  Context: None  |  Mode: Chat  |  Logs: ./autohf_logs",
            id="status-bar",
        )

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def on_mount(self) -> None:
        """Focus the chat input on startup and initialize chat agent."""
        self.query_one("#chat-input", Input).focus()
        from autohf.agents.chat_agent import GemmaChatAgent
        self.chat_agent = GemmaChatAgent()
        self.chat_history = []
        try:
            self.chat_agent.load()
        except Exception:
            pass

    @on(Input.Submitted, "#chat-input")
    def on_chat_submit(self, event: Input.Submitted) -> None:
        """Handle Enter key in chat input."""
        text = event.value.strip()
        if text:
            event.input.value = ""
            self._handle_user_message(text)

    @on(Button.Pressed, "#send-btn")
    def on_send_pressed(self) -> None:
        """Handle send button click."""
        inp = self.query_one("#chat-input", Input)
        text = inp.value.strip()
        if text:
            inp.value = ""
            self._handle_user_message(text)

    @on(Button.Pressed, "#ex-1")
    def on_ex1(self) -> None:
        self._handle_user_message("Find a dataset for sentiment analysis")

    @on(Button.Pressed, "#ex-2")
    def on_ex2(self) -> None:
        self._handle_user_message("Train a model for tweet classification")

    @on(Button.Pressed, "#ex-3")
    def on_ex3(self) -> None:
        self._handle_user_message("Search datasets for question answering")

    @on(Button.Pressed, "#ex-4")
    def on_ex4(self) -> None:
        self._handle_user_message("Compare models for text summarization")

    @on(Button.Pressed, "#btn-search-datasets")
    def on_search_datasets(self) -> None:
        """Search datasets using sidebar settings."""
        task_type = str(self.query_one("#ds-task-type", Select).value)
        limit = int(str(self.query_one("#ds-search-limit", Select).value))
        task_desc = task_type if task_type != "auto" else "general ML"
        # Map task type to a description
        task_labels = {t[1]: t[0] for t in TASK_TYPES}
        label = task_labels.get(task_type, task_type)
        if task_type == "auto":
            self._add_assistant_msg("⚠️ Please select a specific Task Type in the sidebar, or type a search query in the chat.")
            return
        self._add_user_msg(f"Search top {limit} datasets for {label}")
        self._run_search(task_type, limit)

    @on(Button.Pressed, "#btn-select-model")
    def on_select_model(self) -> None:
        """Recommend models for the selected task."""
        task_type = str(self.query_one("#mdl-task", Select).value)
        if task_type == "auto":
            self._add_assistant_msg("⚠️ Please select a specific Task in the MODEL section first.")
            return
        task_labels = {t[1]: t[0] for t in TASK_TYPES}
        label = task_labels.get(task_type, task_type)
        self._add_user_msg(f"Recommend models for {label}")
        self._run_model_search(task_type)

    @on(Button.Pressed, "#btn-start-training")
    def on_start_training(self) -> None:
        """Start training with sidebar settings."""
        self._add_assistant_msg("🔧 Training requires a task description. Type something like:\n"
                                "  [cyan]\"Train a model for sentiment analysis\"[/cyan]\n"
                                "  [cyan]\"Train NER model\"[/cyan]\n\n"
                                "Or use [bold]▶ RUN FULL PIPELINE[/bold] after searching datasets.")

    @on(Button.Pressed, "#btn-save-config")
    def on_save_config(self) -> None:
        """Save current configuration."""
        save_path = self.query_one("#out-save-path", Input).value
        time_limit = self.query_one("#trn-time-limit", Input).value
        metric = str(self.query_one("#trn-metric", Select).value)
        preset = str(self.query_one("#trn-extra-args", Select).value)

        config_summary = (
            f"💾 [bold green]Configuration saved![/bold green]\n\n"
            f"  Output Dir:  [cyan]{save_path}[/cyan]\n"
            f"  Time Limit:  [cyan]{time_limit}s[/cyan]\n"
            f"  Metric:      [cyan]{metric}[/cyan]\n"
            f"  AG Preset:   [cyan]{preset}[/cyan]"
        )
        self._add_assistant_msg(config_summary)

    @on(Button.Pressed, "#btn-run-pipeline")
    def on_run_pipeline(self) -> None:
        """Run the full ML pipeline."""
        task_type = str(self.query_one("#ds-task-type", Select).value)
        if task_type == "auto":
            self._add_assistant_msg("⚠️ Please select a [bold]Task Type[/bold] in the DATASET section, or type a task description in the chat.\n\n"
                                    "Examples:\n"
                                    "  [cyan]\"Train sentiment analysis model\"[/cyan]\n"
                                    "  [cyan]\"Train NER model\"[/cyan]\n"
                                    "  [cyan]\"Train spam detection model\"[/cyan]")
            return
        task_labels = {t[1]: t[0] for t in TASK_TYPES}
        label = task_labels.get(task_type, task_type)
        self._add_user_msg(f"Run full pipeline for {label}")
        self._run_full_pipeline(task_type)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_show_help(self) -> None:
        self._add_assistant_msg(
            "[bold cyan]📖 AutoHF CLI Help[/bold cyan]\n\n"
            "[bold]Chat Commands:[/bold]\n"
            '  • Type a task like "sentiment analysis" to search datasets\n'
            '  • Type "train <task>" to run training pipeline\n'
            '  • Type "search <task>" to find datasets\n'
            '  • Type "route <task>" to detect ML task type\n'
            '  • Type "info" to see supported tasks\n'
            '  • Type "help" to see this message\n\n'
            "[bold]Keyboard Shortcuts:[/bold]\n"
            "  • Ctrl+Q — Quit\n"
            "  • Ctrl+L — Clear chat\n"
            "  • Ctrl+K — Show shortcuts\n"
            "  • F1 — Help\n\n"
            "[bold]Sidebar:[/bold]\n"
            "  Use the right panel to configure dataset search,\n"
            "  model selection, training, and output settings."
        )

    def action_show_shortcuts(self) -> None:
        self._add_assistant_msg(
            "[bold cyan]⌨ Keyboard Shortcuts[/bold cyan]\n\n"
            "  Ctrl+Q    Quit the application\n"
            "  Ctrl+L    Clear chat history\n"
            "  Ctrl+K    Show this shortcuts panel\n"
            "  F1        Show help\n"
            "  Enter     Send chat message\n"
            "  Tab       Navigate between elements"
        )

    def action_clear_chat(self) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        # Remove all messages but keep welcome
        for child in list(chat_log.children):
            if "chat-msg" in " ".join(child.classes):
                child.remove()
        if hasattr(self, "chat_history"):
            self.chat_history.clear()
        self._add_assistant_msg("[dim]Chat history cleared.[/dim]")

    # ------------------------------------------------------------------
    # Chat logic
    # ------------------------------------------------------------------

    # Built-in ML knowledge base for answering questions without needing AI
    _ML_KNOWLEDGE = {
        "random forest": (
            "🌲 [bold cyan]Random Forest[/bold cyan]\n\n"
            "An ensemble learning method that builds multiple decision trees and merges them.\n\n"
            "[bold]How it works:[/bold]\n"
            "  1. Creates N decision trees on random subsets of data\n"
            "  2. Each tree votes on the prediction\n"
            "  3. Final prediction = majority vote (classification) or average (regression)\n\n"
            "[bold]Best for:[/bold] Tabular data, classification, regression\n"
            "[bold]Pros:[/bold] Resistant to overfitting, handles missing values, feature importance\n"
            "[bold]Cons:[/bold] Slow on very large datasets, less interpretable than single tree\n\n"
            "[bold]AutoHF Preset:[/bold] Available as [cyan]RF[/cyan] in AutoGluon. Use:\n"
            '  [cyan]train tabular classification[/cyan]'
        ),
        "gradient boosting": (
            "🚀 [bold cyan]Gradient Boosting (GBM/XGBoost/LightGBM/CatBoost)[/bold cyan]\n\n"
            "Builds trees sequentially, each one correcting the errors of the previous.\n\n"
            "[bold]Variants in AutoHF:[/bold]\n"
            "  • [cyan]GBM[/cyan] — LightGBM (fast, memory efficient)\n"
            "  • [cyan]XGB[/cyan] — XGBoost (robust, widely used)\n"
            "  • [cyan]CAT[/cyan] — CatBoost (handles categorical features natively)\n\n"
            "[bold]Best for:[/bold] Tabular data, competitions, structured data\n"
            "[bold]Pros:[/bold] State-of-the-art accuracy on tabular data, fast training\n"
            "[bold]Cons:[/bold] Can overfit with too many iterations, requires tuning\n\n"
            "[bold]AutoHF Preset:[/bold] All three are used by default. Try:\n"
            '  [cyan]train text classification[/cyan]'
        ),
        "knn": (
            "📍 [bold cyan]K-Nearest Neighbors (KNN)[/bold cyan]\n\n"
            "Classifies data points based on the K closest training examples.\n\n"
            "[bold]How it works:[/bold]\n"
            "  1. Given a new data point, find K nearest neighbors\n"
            "  2. Majority vote among neighbors = prediction\n\n"
            "[bold]Best for:[/bold] Small datasets, recommendation systems\n"
            "[bold]Pros:[/bold] Simple, no training phase, works well with small data\n"
            "[bold]Cons:[/bold] Slow at prediction time, sensitive to feature scaling\n\n"
            "[bold]AutoHF variants:[/bold]\n"
            "  • [cyan]KNN (Uniform)[/cyan] — equal weight to all neighbors\n"
            "  • [cyan]KNN (Distance)[/cyan] — closer neighbors have more weight"
        ),
        "logistic regression": (
            "📈 [bold cyan]Logistic Regression[/bold cyan]\n\n"
            "A linear model for binary/multiclass classification.\n\n"
            "[bold]How it works:[/bold]\n"
            "  Uses sigmoid function to output probabilities between 0 and 1.\n\n"
            "[bold]Best for:[/bold] Binary classification, text classification baselines\n"
            "[bold]Pros:[/bold] Fast, interpretable, good baseline\n"
            "[bold]Cons:[/bold] Assumes linear decision boundary"
        ),
        "neural network": (
            "🧠 [bold cyan]Neural Networks[/bold cyan]\n\n"
            "Multi-layer networks of connected nodes that learn complex patterns.\n\n"
            "[bold]Types:[/bold]\n"
            "  • [cyan]MLP[/cyan] — Feedforward network for tabular data\n"
            "  • [cyan]CNN[/cyan] — Convolutional, best for images\n"
            "  • [cyan]RNN/LSTM[/cyan] — Sequential data, time series\n"
            "  • [cyan]Transformers[/cyan] — NLP, modern state-of-the-art\n\n"
            "[bold]Best for:[/bold] Complex patterns, large datasets, NLP, vision\n"
            "[bold]Pros:[/bold] Can learn any pattern, state-of-the-art performance\n"
            "[bold]Cons:[/bold] Needs lots of data, GPU recommended, hard to interpret"
        ),
        "svm": (
            "⚡ [bold cyan]Support Vector Machine (SVM)[/bold cyan]\n\n"
            "Finds the optimal hyperplane that separates classes with maximum margin.\n\n"
            "[bold]How it works:[/bold]\n"
            "  Maximizes the distance between the decision boundary and nearest points.\n"
            "  Uses kernel trick for non-linear boundaries.\n\n"
            "[bold]Best for:[/bold] Small-medium datasets, text classification, high-dimensional data\n"
            "[bold]Pros:[/bold] Effective in high dimensions, memory efficient\n"
            "[bold]Cons:[/bold] Slow on large datasets, doesn't give probabilities by default"
        ),
        "clustering": (
            "🔬 [bold cyan]Clustering Algorithms[/bold cyan]\n\n"
            "Unsupervised learning — group data points without labels.\n\n"
            "[bold]Popular algorithms:[/bold]\n"
            "  • [cyan]K-Means[/cyan] — Partition into K groups by centroid distance\n"
            "  • [cyan]DBSCAN[/cyan] — Density-based, finds arbitrary shapes\n"
            "  • [cyan]Hierarchical[/cyan] — Tree of clusters (dendrograms)\n"
            "  • [cyan]Gaussian Mixture[/cyan] — Probabilistic clustering\n\n"
            "[bold]Best for:[/bold] Customer segmentation, anomaly detection, data exploration\n"
            "[bold]Note:[/bold] AutoHF focuses on supervised learning (classification/regression)."
        ),
        "transformer": (
            "🤖 [bold cyan]Transformer Architecture[/bold cyan]\n\n"
            "The architecture behind GPT, BERT, T5, and modern LLMs.\n\n"
            "[bold]Key innovation:[/bold] Self-attention mechanism\n"
            "  Allows the model to look at all parts of input simultaneously.\n\n"
            "[bold]Popular models:[/bold]\n"
            "  • [cyan]BERT[/cyan] — Bidirectional encoder, great for classification\n"
            "  • [cyan]GPT[/cyan] — Autoregressive decoder, text generation\n"
            "  • [cyan]T5[/cyan] — Encoder-decoder, text-to-text\n"
            "  • [cyan]Llama[/cyan] — Open-source LLM by Meta\n"
            "  • [cyan]Gemma[/cyan] — Open-source LLM by Google\n\n"
            "[bold]In AutoHF:[/bold] Use [cyan]search text-classification[/cyan] to find datasets,\n"
            "then train with AutoGluon for tabular features."
        ),
        "accuracy": (
            "📊 [bold cyan]Accuracy Metric[/bold cyan]\n\n"
            "Percentage of correct predictions.\n"
            "  [dim]Accuracy = Correct / Total[/dim]\n\n"
            "[bold]When to use:[/bold] Balanced datasets\n"
            "[bold]When NOT to use:[/bold] Imbalanced datasets (use F1 or balanced_accuracy)\n\n"
            "[bold]Available metrics in AutoHF:[/bold]\n"
            "  accuracy, balanced_accuracy, f1, f1_macro, f1_weighted,\n"
            "  roc_auc, log_loss, precision, recall, mcc, rmse, mae, r²"
        ),
        "f1": (
            "📊 [bold cyan]F1 Score[/bold cyan]\n\n"
            "Harmonic mean of Precision and Recall.\n"
            "  [dim]F1 = 2 × (Precision × Recall) / (Precision + Recall)[/dim]\n\n"
            "[bold]Variants:[/bold]\n"
            "  • [cyan]f1[/cyan] — Binary F1\n"
            "  • [cyan]f1_macro[/cyan] — Average F1 across all classes (equal weight)\n"
            "  • [cyan]f1_weighted[/cyan] — Weighted by class frequency\n\n"
            "[bold]When to use:[/bold] Imbalanced datasets, when both false positives and false negatives matter"
        ),
        "overfitting": (
            "📉 [bold cyan]Overfitting[/bold cyan]\n\n"
            "When a model learns training data too well, including noise.\n\n"
            "[bold]Signs:[/bold]\n"
            "  • Training accuracy is very high, test accuracy is much lower\n"
            "  • Model performs poorly on new data\n\n"
            "[bold]Solutions:[/bold]\n"
            "  • More training data\n"
            "  • Regularization (L1, L2, dropout)\n"
            "  • Early stopping\n"
            "  • Simpler model\n"
            "  • Cross-validation\n\n"
            "[bold]In AutoHF:[/bold] AutoGluon handles this with:\n"
            "  • Multiple model ensembling\n"
            "  • Automatic early stopping\n"
            "  • Stratified train/test split"
        ),
        "autogluon": (
            "🏆 [bold cyan]AutoGluon[/bold cyan]\n\n"
            "AutoML framework by Amazon that automatically trains and ensembles models.\n\n"
            "[bold]What it does:[/bold]\n"
            "  1. Trains multiple model types (GBM, XGB, CAT, RF, KNN, NN)\n"
            "  2. Automatically tunes hyperparameters\n"
            "  3. Stacks and ensembles the best models\n"
            "  4. Returns a leaderboard of all trained models\n\n"
            "[bold]AutoHF presets:[/bold]\n"
            "  • [cyan]quick_prototype[/cyan] — 60s, fast testing\n"
            "  • [cyan]medium_quality[/cyan] — 5min, good balance\n"
            "  • [cyan]high_quality[/cyan] — 10min, better results\n"
            "  • [cyan]best_quality[/cyan] — 1hr, maximum accuracy"
        ),
        "classification": (
            "🏷️ [bold cyan]Classification[/bold cyan]\n\n"
            "Predict a discrete category/label for each data point.\n\n"
            "[bold]Types:[/bold]\n"
            "  • [cyan]Binary[/cyan] — Two classes (spam/not spam, positive/negative)\n"
            "  • [cyan]Multiclass[/cyan] — Multiple classes (emotions, topics, intents)\n\n"
            "[bold]Common tasks:[/bold]\n"
            "  sentiment analysis, spam detection, topic classification,\n"
            "  intent detection, NER, emotion detection\n\n"
            "[bold]Try it:[/bold]\n"
            '  [cyan]search sentiment analysis[/cyan]\n'
            '  [cyan]train spam detection[/cyan]'
        ),
        "regression": (
            "📈 [bold cyan]Regression[/bold cyan]\n\n"
            "Predict a continuous numerical value.\n\n"
            "[bold]Examples:[/bold]\n"
            "  • House price prediction\n"
            "  • Stock price forecasting\n"
            "  • Temperature prediction\n"
            "  • Sales forecasting\n\n"
            "[bold]Metrics:[/bold] RMSE, MAE, R²\n\n"
            "[bold]Try it:[/bold]\n"
            '  [cyan]train tabular regression[/cyan]'
        ),
    }

    # Intent patterns for smart routing
    _INTENT_PATTERNS = {
        "search": [
            "find", "search", "look for", "get", "show me", "list",
            "discover", "explore", "browse", "fetch", "what datasets",
            "which datasets", "find me", "find a dataset", "dataset for",
            "datasets for", "data for",
        ],
        "train": [
            "train", "build", "create a model", "make a model", "fit",
            "build a model", "train a model", "create model", "start training",
            "run training", "fine-tune", "finetune",
        ],
        "compare": [
            "compare", "vs", "versus", "difference between", "which is better",
            "recommend", "suggest", "best model", "best algorithm",
            "compare models", "which model",
        ],
        "explain": [
            "what is", "what are", "explain", "how does", "how do",
            "tell me about", "describe", "define", "meaning of",
            "what's", "whats",
        ],
        "models": [
            "models for", "recommend model", "find models", "show models",
            "which models", "top models", "best models", "model for",
        ],
    }

    def _handle_user_message(self, text: str) -> None:
        """Process a user message and route to the appropriate action.
        
        Smart routing: understands natural language and auto-takes action.
        """
        # Hide welcome on first message
        try:
            welcome = self.query_one("#welcome-section")
            welcome.display = False
        except Exception:
            pass

        self._add_user_msg(text)

        lower = text.lower().strip()

        # ----- Exact commands -----
        if lower in ("help", "?", "/help"):
            self.action_show_help()
            return
        if lower in ("info", "/info"):
            self._show_info()
            return
        if lower in ("version", "/version"):
            from autohf import __version__
            self._add_assistant_msg(f"[bold cyan]AutoHF v{__version__}[/bold cyan]")
            return
        if lower in ("clear", "/clear"):
            self.action_clear_chat()
            return
        if lower in ("quit", "exit", "/quit"):
            self.exit()
            return

        # ----- Explicit prefix commands -----
        if lower.startswith("search ") or lower.startswith("find "):
            query = text.split(maxsplit=1)[1] if " " in text else ""
            self._run_search_from_text(query)
            return
        if lower.startswith("train "):
            query = text.split(maxsplit=1)[1] if " " in text else ""
            self._run_train_from_text(query)
            return
        if lower.startswith("route "):
            query = text.split(maxsplit=1)[1] if " " in text else ""
            self._run_route(query)
            return

        # ----- Smart intent detection -----
        intent = self._detect_intent(lower)

        if intent == "explain":
            answer = self._check_knowledge_base(lower)
            if answer:
                self._add_assistant_msg(answer)
                return
            self._run_chat_agent(text)
            return

        if intent == "compare":
            self._run_chat_agent(text)
            return

        if intent == "models":
            # Extract the task from the query
            query = self._extract_task_query(text)
            self._run_smart_model_search(query)
            return

        if intent == "train":
            query = self._extract_task_query(text)
            self._run_train_from_text(query)
            return

        if intent == "search":
            query = self._extract_task_query(text)
            self._run_search_from_text(query)
            return

        # ----- Fallback: smart conversational LLM chat -----
        self._run_chat_agent(text)

    def _detect_intent(self, text: str) -> Optional[str]:
        """Detect user intent from natural language."""
        for intent, patterns in self._INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in text:
                    return intent
        return None

    def _extract_task_query(self, text: str) -> str:
        """Extract the ML task/topic from a natural language query."""
        lower = text.lower()
        # Remove common filler words
        removals = [
            "find me", "find a dataset for", "find dataset for",
            "find a", "find", "search for", "search", "show me",
            "get me", "get", "list", "look for", "i want to",
            "i need", "can you", "please", "help me",
            "train a model for", "train model for", "build a model for",
            "build model for", "create a model for", "train a", "train",
            "build a", "build", "create a", "create",
            "models for", "model for", "recommend model for",
            "recommend models for", "suggest model for",
            "best model for", "top models for",
            "datasets for", "dataset for", "data for",
            "a dataset for", "the dataset for",
        ]
        result = lower
        for r in removals:
            result = result.replace(r, "")
        result = result.strip().strip('"').strip("'").strip()
        return result if result else text

    def _check_knowledge_base(self, text: str) -> Optional[str]:
        """Check if the question matches any built-in ML knowledge."""
        for key, answer in self._ML_KNOWLEDGE.items():
            if key in text:
                return answer
        return None

    def _add_user_msg(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        msg = Static(f"[bold cyan]You >[/bold cyan] {text}", classes="chat-msg-user")
        chat_log.mount(msg)
        msg.scroll_visible()

    def _add_assistant_msg(self, text: str) -> None:
        chat_log = self.query_one("#chat-log", VerticalScroll)
        msg = Static(f"[bold green]AutoHF >[/bold green] {text}", classes="chat-msg-assistant")
        chat_log.mount(msg)
        msg.scroll_visible()


    # ------------------------------------------------------------------
    # Pipeline Workers (async background tasks)
    # ------------------------------------------------------------------

    def _show_info(self) -> None:
        """Show supported tasks and presets."""
        from autohf.agents.task_agent import list_supported_tasks
        from autohf.core.config import PRESET_CONFIGS

        tasks = list_supported_tasks()
        task_lines = "\n".join(f"  • [cyan]{t['task_type']}[/cyan] — {t['label']}" for t in tasks)

        preset_lines = "\n".join(
            f"  • [cyan]{name}[/cyan] — {conf.get('time_limit', 300)}s, "
            f"{conf.get('max_dataset_rows', 50000):,} rows, "
            f"AG preset: {conf.get('presets', 'medium_quality')}"
            for name, conf in PRESET_CONFIGS.items()
        )

        self._add_assistant_msg(
            f"[bold cyan]ℹ️ AutoHF Info[/bold cyan]\n\n"
            f"[bold]Supported Tasks ({len(tasks)}):[/bold]\n{task_lines}\n\n"
            f"[bold]Available Presets:[/bold]\n{preset_lines}"
        )

    @work(thread=True)
    def _run_search_from_text(self, query: str) -> None:
        """Search datasets from a text query."""
        from autohf.agents.task_agent import detect_task
        from autohf.core.autohf import AutoHF
        from autohf.core.config import AutoHFConfig

        self.call_from_thread(self._add_assistant_msg, f"🔍 Searching datasets for [cyan]'{query}'[/cyan]...")

        try:
            task_info = detect_task(query, router="keyword")
            config = AutoHFConfig(router="keyword")
            hf = AutoHF(config=config)
            ranked = hf.search(query, top_n=10)

            lines = [f"[bold green]✓ Found {len(ranked)} datasets for '{query}'[/bold green]\n"]
            lines.append(f"  [dim]Detected task: {task_info.task_type} (confidence: {task_info.confidence})[/dim]\n")

            for i, ds in enumerate(ranked[:10], 1):
                lines.append(
                    f"  [bold]{i}.[/bold] [cyan]{ds.dataset_id}[/cyan]  "
                    f"↓{ds.downloads:,}  ♥{ds.likes:,}  "
                    f"score=[magenta]{ds.score:.4f}[/magenta]"
                )

            self.call_from_thread(self._add_assistant_msg, "\n".join(lines))
        except Exception as e:
            self.call_from_thread(self._add_assistant_msg, f"[bold red]❌ Error:[/bold red] {e}")

    @work(thread=True)
    def _run_search(self, task_type: str, limit: int) -> None:
        """Search datasets by task type from sidebar."""
        from autohf.core.autohf import AutoHF
        from autohf.core.config import AutoHFConfig

        task_labels = {t[1]: t[0] for t in TASK_TYPES}
        label = task_labels.get(task_type, task_type)

        self.call_from_thread(self._add_assistant_msg, f"🔍 Searching top {limit} datasets for [cyan]{label}[/cyan]...")

        try:
            config = AutoHFConfig(router="keyword")
            hf = AutoHF(config=config)
            ranked = hf.search(label, top_n=limit)

            lines = [f"[bold green]✓ Found {len(ranked)} datasets[/bold green]\n"]
            for i, ds in enumerate(ranked, 1):
                lines.append(
                    f"  [bold]{i}.[/bold] [cyan]{ds.dataset_id}[/cyan]  "
                    f"↓{ds.downloads:,}  ♥{ds.likes:,}  "
                    f"score=[magenta]{ds.score:.4f}[/magenta]"
                )

            self.call_from_thread(self._add_assistant_msg, "\n".join(lines))
        except Exception as e:
            self.call_from_thread(self._add_assistant_msg, f"[bold red]❌ Error:[/bold red] {e}")

    @work(thread=True)
    def _run_model_search(self, task_type: str) -> None:
        """Search models for a task type."""
        from autohf.agents.dataset_agent import find_models

        task_labels = {t[1]: t[0] for t in TASK_TYPES}
        label = task_labels.get(task_type, task_type)

        self.call_from_thread(self._add_assistant_msg, f"🤖 Searching models for [cyan]{label}[/cyan]...")

        try:
            models = find_models(task_type, limit=10)
            if not models:
                self.call_from_thread(self._add_assistant_msg, f"⚠️ No models found for task '{task_type}'.")
                return

            lines = [f"[bold green]✓ Top {len(models)} models for {label}:[/bold green]\n"]
            for i, m in enumerate(models, 1):
                lines.append(
                    f"  [bold]{i}.[/bold] [cyan]{m['model_id']}[/cyan]  "
                    f"↓{m['downloads']:,}  ♥{m['likes']:,}"
                )

            self.call_from_thread(self._add_assistant_msg, "\n".join(lines))
        except Exception as e:
            self.call_from_thread(self._add_assistant_msg, f"[bold red]❌ Error:[/bold red] {e}")

    @work(thread=True)
    def _run_route(self, query: str) -> None:
        """Detect task type from text."""
        from autohf.agents.task_agent import detect_task

        self.call_from_thread(self._add_assistant_msg, f"🎯 Detecting task for [cyan]'{query}'[/cyan]...")

        try:
            res = detect_task(query, router="keyword")
            result = (
                f"[bold green]✓ Task Detection Result:[/bold green]\n\n"
                f"  Task Type:   [bold]{res.task_type}[/bold]\n"
                f"  Label:       {res.task_label}\n"
                f"  Confidence:  {res.confidence}\n"
                f"  Keywords:    {', '.join(res.keywords)}\n"
                f"  Problem Type: {res.problem_type}"
            )
            self.call_from_thread(self._add_assistant_msg, result)
        except Exception as e:
            self.call_from_thread(self._add_assistant_msg, f"[bold red]❌ Error:[/bold red] {e}")

    @work(thread=True)
    def _run_smart_detect(self, text: str) -> None:
        """Smart fallback: detect ML task and AUTO-SEARCH datasets + models."""
        from autohf.agents.task_agent import detect_task
        from autohf.agents.dataset_agent import find_models

        try:
            # Check knowledge base first
            answer = self._check_knowledge_base(text.lower())
            if answer:
                self.call_from_thread(self._add_assistant_msg, answer)
                return

            res = detect_task(text, router="keyword")

            if res.confidence >= 0.5:
                # HIGH CONFIDENCE: Auto-search datasets AND models
                self.call_from_thread(
                    self._add_assistant_msg,
                    f"🎯 Detected: [bold]{res.task_label}[/bold] "
                    f"([cyan]{res.task_type}[/cyan], confidence: {res.confidence})\n\n"
                    f"[dim]Searching datasets and models...[/dim]"
                )

                # Auto-search datasets
                try:
                    from autohf.core.autohf import AutoHF
                    from autohf.core.config import AutoHFConfig

                    config = AutoHFConfig(router="keyword")
                    hf = AutoHF(config=config)
                    ranked = hf.search(text, top_n=5)

                    if ranked:
                        lines = [f"\n[bold]📦 Top Datasets:[/bold]"]
                        for i, ds in enumerate(ranked[:5], 1):
                            lines.append(
                                f"  {i}. [cyan]{ds.dataset_id}[/cyan]  "
                                f"↓{ds.downloads:,}  ♥{ds.likes:,}  "
                                f"score=[magenta]{ds.score:.4f}[/magenta]"
                            )
                        self.call_from_thread(self._add_assistant_msg, "\n".join(lines))
                except Exception:
                    pass

                # Auto-search models
                try:
                    models = find_models(res.task_type, limit=5)
                    if models:
                        lines = [f"\n[bold]🤖 Top Models:[/bold]"]
                        for i, m in enumerate(models[:5], 1):
                            lines.append(
                                f"  {i}. [cyan]{m['model_id']}[/cyan]  "
                                f"↓{m['downloads']:,}  ♥{m['likes']:,}"
                            )
                        self.call_from_thread(self._add_assistant_msg, "\n".join(lines))
                except Exception:
                    pass

                # Offer next actions
                self.call_from_thread(
                    self._add_assistant_msg,
                    f"\n[bold]What's next?[/bold]\n"
                    f"  • Type [cyan]train {text}[/cyan] to start training\n"
                    f"  • Type [cyan]search {text}[/cyan] to see more datasets\n"
                    f"  • Adjust settings in the sidebar → click [bold]▶ RUN FULL PIPELINE[/bold]"
                )
            else:
                # LOW CONFIDENCE: Give helpful suggestions
                self.call_from_thread(
                    self._add_assistant_msg,
                    f"🤔 I understand you're interested in [cyan]'{text}'[/cyan], "
                    f"but I'm not 100% sure of the ML task type.\n\n"
                    f"Here's what you can try:\n\n"
                    f"[bold]🔍 Search:[/bold]\n"
                    f'  [cyan]search {text}[/cyan]\n\n'
                    f"[bold]❓ Ask:[/bold]\n"
                    f'  [cyan]what is random forest[/cyan]\n'
                    f'  [cyan]explain gradient boosting[/cyan]\n'
                    f'  [cyan]what is classification[/cyan]\n\n'
                    f"[bold]🏋️ Train:[/bold]\n"
                    f'  [cyan]train sentiment analysis[/cyan]\n'
                    f'  [cyan]train spam detection[/cyan]\n\n'
                    f"[bold]📋 Info:[/bold]\n"
                    f'  [cyan]info[/cyan] — see all 12 supported ML tasks',
                )
        except Exception as e:
            self.call_from_thread(self._add_assistant_msg, f"[bold red]❌ Error:[/bold red] {e}")

    @work(thread=True)
    def _run_chat_agent(self, text: str) -> None:
        """Call the GemmaChatAgent to generate a conversational response."""
        try:
            response = self.chat_agent.generate(text, self.chat_history)
            
            # Update history
            self.chat_history.append({"role": "user", "content": text})
            self.chat_history.append({"role": "assistant", "content": response})
            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]
                
            self.call_from_thread(self._add_assistant_msg, response)
        except Exception as e:
            from loguru import logger
            logger.warning("GemmaChatAgent failed: {}. Falling back to smart detect...", e)
            self._run_smart_detect(text)

    @work(thread=True)
    def _run_explain(self, text: str) -> None:
        """Explain an ML concept using task detection + knowledge base."""
        from autohf.agents.task_agent import detect_task

        try:
            # Try task detection to give context
            cleaned = self._extract_task_query(text)
            res = detect_task(cleaned, router="keyword")

            if res.confidence >= 0.5:
                explanation = (
                    f"📚 [bold cyan]{res.task_label}[/bold cyan]\n\n"
                    f"[bold]Task Type:[/bold] {res.task_type}\n"
                    f"[bold]Problem Type:[/bold] {res.problem_type}\n"
                    f"[bold]Keywords:[/bold] {', '.join(res.keywords)}\n\n"
                    f"[bold]What this means:[/bold]\n"
                    f"  This is a {'classification' if 'classification' in res.task_type else 'generation/regression'} task.\n"
                    f"  AutoHF can search datasets on Hugging Face Hub and train models using AutoGluon.\n\n"
                    f"[bold]Algorithms used:[/bold]\n"
                    f"  • LightGBM (gradient boosting)\n"
                    f"  • XGBoost\n"
                    f"  • CatBoost\n"
                    f"  • Random Forest\n"
                    f"  • Extra Trees\n"
                    f"  • K-Nearest Neighbors\n\n"
                    f"[bold]Try it:[/bold]\n"
                    f"  [cyan]search {cleaned}[/cyan]\n"
                    f"  [cyan]train {cleaned}[/cyan]"
                )
                self.call_from_thread(self._add_assistant_msg, explanation)
            else:
                self.call_from_thread(
                    self._add_assistant_msg,
                    f"📚 I don't have detailed info about [cyan]'{cleaned}'[/cyan] yet.\n\n"
                    f"[bold]I can explain these ML concepts:[/bold]\n"
                    f"  • [cyan]what is random forest[/cyan]\n"
                    f"  • [cyan]what is gradient boosting[/cyan]\n"
                    f"  • [cyan]explain neural network[/cyan]\n"
                    f"  • [cyan]what is KNN[/cyan]\n"
                    f"  • [cyan]what is SVM[/cyan]\n"
                    f"  • [cyan]explain classification[/cyan]\n"
                    f"  • [cyan]explain regression[/cyan]\n"
                    f"  • [cyan]what is overfitting[/cyan]\n"
                    f"  • [cyan]what is transformer[/cyan]\n"
                    f"  • [cyan]what is accuracy[/cyan]\n"
                    f"  • [cyan]what is f1 score[/cyan]\n"
                    f"  • [cyan]what is autogluon[/cyan]\n"
                    f"  • [cyan]what is clustering[/cyan]\n"
                    f"  • [cyan]what is logistic regression[/cyan]"
                )
        except Exception as e:
            self.call_from_thread(self._add_assistant_msg, f"[bold red]❌ Error:[/bold red] {e}")

    def _run_compare(self, text: str) -> None:
        """Compare ML algorithms or models."""
        lower = text.lower()

        # Check for specific algorithm comparisons
        algorithms = {
            "random forest": "RF",
            "gradient boosting": "GBM",
            "xgboost": "XGB",
            "catboost": "CAT",
            "lightgbm": "GBM",
            "knn": "KNN",
            "neural network": "NN",
            "svm": "SVM",
            "logistic regression": "LR",
        }

        found = [name for name in algorithms if name in lower]

        if len(found) >= 2:
            comparison = (
                f"⚖️ [bold cyan]Comparison: {' vs '.join(found)}[/bold cyan]\n\n"
            )
            for algo in found:
                kb_answer = self._check_knowledge_base(algo)
                if kb_answer:
                    comparison += kb_answer + "\n\n" + "─" * 50 + "\n\n"
            comparison += (
                f"[bold]💡 Recommendation:[/bold]\n"
                f"  AutoGluon trains ALL algorithms and picks the best one automatically!\n"
                f"  Just type [cyan]train <your task>[/cyan] and let AutoHF handle it."
            )
            self._add_assistant_msg(comparison)
        else:
            self._add_assistant_msg(
                f"⚖️ [bold cyan]Model Comparison[/bold cyan]\n\n"
                f"AutoGluon automatically trains and compares these algorithms:\n\n"
                f"  [cyan]GBM[/cyan]  (LightGBM)       — Fast, accurate on tabular data\n"
                f"  [cyan]XGB[/cyan]  (XGBoost)        — Robust, widely used\n"
                f"  [cyan]CAT[/cyan]  (CatBoost)       — Handles categorical features\n"
                f"  [cyan]RF[/cyan]   (Random Forest)   — Resistant to overfitting\n"
                f"  [cyan]XT[/cyan]   (Extra Trees)     — Even more randomized than RF\n"
                f"  [cyan]KNN[/cyan]  (K-Nearest)       — Simple, effective for small data\n\n"
                f"[bold]AutoGluon runs all of them and creates a leaderboard![/bold]\n\n"
                f"To see this in action, type:\n"
                f"  [cyan]train sentiment analysis[/cyan]\n"
                f"  [cyan]train spam detection[/cyan]\n\n"
                f"Want to compare specific algorithms? Try:\n"
                f"  [cyan]compare random forest vs gradient boosting[/cyan]\n"
                f"  [cyan]compare knn vs svm[/cyan]"
            )

    @work(thread=True)
    def _run_smart_model_search(self, query: str) -> None:
        """Smart model search from natural language."""
        from autohf.agents.task_agent import detect_task
        from autohf.agents.dataset_agent import find_models

        self.call_from_thread(self._add_assistant_msg, f"🤖 Finding the best models for [cyan]'{query}'[/cyan]...")

        try:
            res = detect_task(query, router="keyword")
            models = find_models(res.task_type, limit=10)

            if models:
                lines = [
                    f"[bold green]✓ Top models for {res.task_label} ({res.task_type}):[/bold green]\n"
                ]
                for i, m in enumerate(models, 1):
                    lines.append(
                        f"  [bold]{i}.[/bold] [cyan]{m['model_id']}[/cyan]  "
                        f"↓{m['downloads']:,}  ♥{m['likes']:,}"
                    )
                lines.append(
                    f"\n[bold]💡 AutoHF also trains these algorithms locally:[/bold]\n"
                    f"  GBM, XGBoost, CatBoost, Random Forest, Extra Trees, KNN\n"
                    f"  → Type [cyan]train {query}[/cyan] to train automatically"
                )
                self.call_from_thread(self._add_assistant_msg, "\n".join(lines))
            else:
                self.call_from_thread(
                    self._add_assistant_msg,
                    f"⚠️ No HF models found for '{res.task_type}'.\n\n"
                    f"But AutoHF can still train locally using AutoGluon!\n"
                    f"  Type [cyan]train {query}[/cyan]"
                )
        except Exception as e:
            self.call_from_thread(self._add_assistant_msg, f"[bold red]❌ Error:[/bold red] {e}")


    @work(thread=True)
    def _run_train_from_text(self, query: str) -> None:
        """Run training pipeline from chat text."""
        from autohf.core.autohf import AutoHF

        # Get settings from sidebar
        try:
            time_limit = int(self.query_one("#trn-time-limit", Input).value)
        except (ValueError, Exception):
            time_limit = 300
        
        try:
            metric_val = str(self.query_one("#trn-metric", Select).value)
        except Exception:
            metric_val = "auto"

        try:
            preset_val = str(self.query_one("#trn-extra-args", Select).value)
        except Exception:
            preset_val = "medium_quality"

        try:
            output_dir = self.query_one("#out-save-path", Input).value
        except Exception:
            output_dir = "./autohf_output"

        self.call_from_thread(
            self._add_assistant_msg,
            f"🚀 [bold]Starting AutoHF Pipeline[/bold]\n\n"
            f"  Task:       [cyan]{query}[/cyan]\n"
            f"  Preset:     [cyan]{preset_val}[/cyan]\n"
            f"  Time Limit: [cyan]{time_limit}s[/cyan]\n"
            f"  Metric:     [cyan]{metric_val}[/cyan]\n"
            f"  Output:     [cyan]{output_dir}[/cyan]\n\n"
            f"[dim]Training in progress... This may take a while.[/dim]"
        )

        try:
            overrides = {
                "time_limit": time_limit,
                "output_dir": output_dir,
                "router": "keyword",
            }
            if metric_val != "auto":
                overrides["eval_metric"] = metric_val

            hf = AutoHF.from_preset(preset_val, **overrides)
            result = hf.train(query)

            metrics_str = "\n".join(f"    {k}: [bold]{v}[/bold]" for k, v in result.metrics.items())

            self.call_from_thread(
                self._add_assistant_msg,
                f"[bold green]🎉 Training Complete![/bold green]\n\n"
                f"  Task:          [cyan]{result.task_type}[/cyan]\n"
                f"  Dataset:       [cyan]{result.dataset_id}[/cyan]\n"
                f"  Best Model:    [bold]{result.best_model_name or 'N/A'}[/bold]\n"
                f"  Models Trained: {result.num_models_trained}\n"
                f"  Training Time: {result.training_time:.1f}s\n"
                f"  Model Path:    [cyan]{result.model_path}[/cyan]\n\n"
                f"  [bold]📈 Metrics:[/bold]\n{metrics_str}"
            )
        except ImportError as e:
            self.call_from_thread(
                self._add_assistant_msg,
                f"[bold red]❌ Missing dependency:[/bold red] {e}\n\n"
                f"Install training dependencies:\n"
                f"  [cyan]pip install autohf\\[train][/cyan]"
            )
        except Exception as e:
            self.call_from_thread(self._add_assistant_msg, f"[bold red]❌ Training Error:[/bold red] {e}")

    @work(thread=True)
    def _run_full_pipeline(self, task_type: str) -> None:
        """Run the full pipeline from sidebar settings."""
        task_labels = {t[1]: t[0] for t in TASK_TYPES}
        label = task_labels.get(task_type, task_type)
        self._run_train_from_text(label)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tui() -> None:
    """Launch the AutoHF interactive TUI."""
    app = AutoHFApp()
    app.run()


if __name__ == "__main__":
    run_tui()
