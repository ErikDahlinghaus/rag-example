from pathlib import Path

import typer

from rag_example.database import get_chromadb_client, query_and_rerank, write_file_to_chromadb
from rag_example.llm import generate_response
from rag_example.settings import settings

app = typer.Typer()


@app.command()
def show_settings() -> None:
    """Print current settings configuration."""
    typer.echo("Current RAG Example Settings:")
    typer.echo("=" * 30)
    for key, value in settings.model_dump().items():
        typer.echo(f"{key}: {value}")


@app.command()
def reprocess() -> None:
    """Delete ChromaDB and reprocess all files from the markdown folder."""
    # Reset ChromaDB
    typer.echo("Resetting ChromaDB...")
    get_chromadb_client().reset()

    # Find all markdown files
    markdown_folder = Path("./markdown")
    if not markdown_folder.exists():
        typer.echo("Error: markdown folder not found")
        raise typer.Exit(1)

    markdown_files = list(markdown_folder.glob("**/*.md"))
    if not markdown_files:
        typer.echo("No markdown files found in ./markdown")
        return

    typer.echo(f"Processing {len(markdown_files)} markdown files...")
    for file_path in markdown_files:
        typer.echo(f"Processing {file_path}")
        write_file_to_chromadb(str(file_path))
    typer.echo("Reprocessing complete!")


@app.command()
def query(search_string: str, n_results: int = 5) -> None:
    """Query the documents with a search string."""
    results = query_and_rerank(search_string, n_results)

    if not results:
        typer.echo("No results found.")
        return

    typer.echo(f"Found {len(results)} results:")
    typer.echo("=" * 50)

    for i, result in enumerate(results, 1):
        typer.echo(
            f"\n{i}. [{result.metadata.get('filename', 'Unknown')}] "
            f"(Score: {result.rerank_score:.4f}) (Distance: {result.distance:.4f})"
        )
        typer.echo(result.document)


@app.command()
def answer(query: str, n_results: int = 5) -> None:
    """Answer a query using RAG (Retrieval-Augmented Generation)."""
    typer.echo("Searching documents...")
    results = query_and_rerank(query, n_results)

    if results:
        typer.echo(f"Found {len(results)} results.")
        context = "\n\n".join([result.document for result in results])
        prompt = f"""Based on the following context, please answer the question. Be as specific as possible.

Context:
{context}

Question: {query}

Answer:"""
    else:
        typer.echo("No relevant documents found.")
        prompt = query

    typer.echo("Generating response...")
    response = generate_response(prompt)

    typer.echo("\n# Prompt:")
    typer.echo(prompt)

    if response.thinking_content:
        typer.echo("\n# Thinking process:")
        typer.echo(response.thinking_content)

    typer.echo("\n# Answer:")
    typer.echo(response.content)


if __name__ == "__main__":
    app()
