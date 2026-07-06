import argparse
import asyncio
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from app.core.config import ModelTokenPrice, cost_settings, llm_settings
from app.core.exceptions import ExtractionError, PDFParseError
from app.core.lifecycle import validate_llm_api_keys
from app.lib.pdf import parse_pdf
from app.models.extraction import ExtractionStatus
from app.schemas.registry import SchemaRegistry
from app.schemas.requests import ExtractionStrategy
from app.services.page_extraction import PageStrategyRunner
from app.services.tool_call_extractor import ToolCallExtractor
from app.services.validation import validate_extraction


@dataclass(frozen=True)
class StrategyEvaluationResult:
    strategy: ExtractionStrategy
    status: str
    source_pages: list[int]
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: Decimal
    extracted_fields: list[str]
    warnings: list[str]


DEFAULT_EVALUATION_STRATEGIES: tuple[ExtractionStrategy, ...] = (
    ExtractionStrategy.FULL,
    ExtractionStrategy.SMART,
    ExtractionStrategy.PAGE_BY_PAGE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare extraction strategies for one PDF."
    )
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--doc-type", required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/evaluation/strategy-comparison.md"),
    )
    args = parser.parse_args()
    valid_doc_types = sorted(SchemaRegistry.list_types())
    if args.doc_type not in valid_doc_types:
        parser.error(
            "--doc-type must be one of: " + ", ".join(valid_doc_types)
        )
    return args


async def main() -> None:
    args = parse_args()
    content = read_pdf(args.pdf_path)
    try:
        parsed = parse_pdf(content, args.pdf_path.name)
    except PDFParseError as exc:
        print(
            f"Could not parse PDF '{args.pdf_path.name}': "
            f"{exc.original_exception}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    page_runner = build_page_runner()
    results = await evaluate_strategies(
        page_runner=page_runner,
        pages=parsed.pages_text,
        doc_type=args.doc_type,
    )
    report = render_strategy_evaluation_markdown(
        pdf_path=args.pdf_path,
        doc_type=args.doc_type,
        results=results,
    )
    write_report(args.output, report)
    print(f"Wrote {args.output}")


def read_pdf(pdf_path: Path) -> bytes:
    try:
        return pdf_path.read_bytes()
    except (FileNotFoundError, PermissionError, OSError) as exc:
        print(
            f"Could not read PDF '{pdf_path.name}': {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def write_report(output_path: Path, report: str) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")
    except (PermissionError, OSError) as exc:
        print(
            f"Could not write report '{output_path.name}': {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


async def evaluate_strategies(
    *,
    page_runner: PageStrategyRunner,
    pages: list[str],
    doc_type: str,
    strategies: Sequence[ExtractionStrategy] = DEFAULT_EVALUATION_STRATEGIES,
    model: str | None = None,
    price_for_model: Callable[
        [str], ModelTokenPrice
    ] = cost_settings.price_for_model,
) -> list[StrategyEvaluationResult]:
    results = []
    for strategy in strategies:
        try:
            extraction = await page_runner.extract_from_pages(
                pages=pages,
                doc_type=doc_type,
                strategy=strategy,
            )
            validation = validate_extraction(doc_type, extraction.raw_output)
            warnings = validation.warnings + extraction.warnings
            extracted_fields = sorted((validation.extracted_data or {}).keys())
            status = validation.status.value
            input_tokens = extraction.input_tokens
            output_tokens = extraction.output_tokens
            source_pages = extraction.source_pages
        except ExtractionError as exc:
            warnings = [str(exc)]
            extracted_fields = []
            status = ExtractionStatus.FAILED.value
            input_tokens = exc.input_tokens
            output_tokens = exc.output_tokens
            source_pages = []

        results.append(
            StrategyEvaluationResult(
                strategy=strategy,
                status=status,
                source_pages=source_pages,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=estimate_cost(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    model=model,
                    price_for_model=price_for_model,
                ),
                extracted_fields=extracted_fields,
                warnings=warnings,
            )
        )

    return results


def estimate_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    model: str | None = None,
    price_for_model: Callable[
        [str], ModelTokenPrice
    ] = cost_settings.price_for_model,
) -> Decimal:
    price = price_for_model(model or llm_settings.EXTRACTION_MODEL)
    return (input_tokens * price.input_token_price_usd) + (
        output_tokens * price.output_token_price_usd
    )


def render_strategy_evaluation_markdown(
    *,
    pdf_path: Path,
    doc_type: str,
    results: Sequence[StrategyEvaluationResult],
) -> str:
    lines = [
        "# Strategy Comparison",
        "",
        f"- PDF: `{pdf_path}`",
        f"- Document type: `{doc_type}`",
        "",
        "| Strategy | Status | Pages | Input tokens | Output tokens | "
        "Estimated cost USD | Fields | Warnings |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]

    for result in results:
        pages = _join_values(result.source_pages)
        fields = _join_values(result.extracted_fields)
        warnings = _join_values(result.warnings)
        lines.append(
            f"| {result.strategy.value} | {result.status} | {pages} | "
            f"{result.input_tokens} | {result.output_tokens} | "
            f"{result.estimated_cost_usd} | {fields} | {warnings} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `full` sends every page in one extraction request.",
            "- `smart` sends selected pages based on the page heuristic.",
            "- `page_by_page` extracts each page separately and merges fields.",
        ]
    )
    return "\n".join(lines) + "\n"


def _join_values(values: Sequence[object]) -> str:
    if not values:
        return "-"

    return ", ".join(str(value).replace("|", "\\|") for value in values)


def build_page_runner() -> PageStrategyRunner:
    from app.integrations.llm.litellm_client import LiteLLMClient

    validate_llm_api_keys()
    llm_client = LiteLLMClient()
    return PageStrategyRunner(
        text_extractor=ToolCallExtractor(llm_client=llm_client)
    )


if __name__ == "__main__":
    asyncio.run(main())
