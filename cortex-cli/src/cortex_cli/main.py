"""Cortex CLI - Command line interface for Cortex semantic memory service."""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Annotated, Optional

import httpx
import pendulum
import typer
from rich.console import Console

app = typer.Typer(help="Cortex - Semantic memory CLI")
console = Console()


def get_config() -> tuple[str, str]:
    """Get Cortex URL and API key from environment."""
    url = os.environ.get("CORTEX_BASE_URL", "http://alpha-pi:7867")
    api_key = os.environ.get("CORTEX_API_KEY")
    if not api_key:
        console.print("[red]Error: CORTEX_API_KEY not set[/red]")
        raise typer.Exit(1)
    return url, api_key


def get_client() -> tuple[httpx.Client, dict]:
    """Get HTTP client with headers."""
    url, api_key = get_config()
    client = httpx.Client(base_url=url, timeout=30.0)
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    return client, headers


def get_local_timezone() -> str:
    """Get the local timezone name."""
    try:
        import zoneinfo
        # Try to get the system timezone
        tz = datetime.now().astimezone().tzinfo
        if hasattr(tz, 'key'):
            return tz.key
        # Fallback: try to read from /etc/timezone
        try:
            with open("/etc/timezone") as f:
                return f.read().strip()
        except FileNotFoundError:
            pass
        # Another fallback: try TZ env var
        return os.environ.get("TZ", "UTC")
    except Exception:
        return "UTC"


@app.command()
def store(
    content: Annotated[
        Optional[str],
        typer.Argument(help="Memory content (use - for stdin)")
    ] = None,
    tags: Annotated[
        Optional[str],
        typer.Option("--tags", "-t", help="Comma-separated tags")
    ] = None,
):
    """Store a new memory."""
    # Read content from stdin if "-" or no argument
    if content == "-" or (content is None and not sys.stdin.isatty()):
        content = sys.stdin.read().strip()

    if not content:
        console.print("[red]Error: No content provided[/red]")
        raise typer.Exit(1)

    client, headers = get_client()

    payload = {
        "content": content,
        "timezone": get_local_timezone(),
    }
    if tags:
        payload["tags"] = [t.strip() for t in tags.split(",")]

    try:
        response = client.post("/store", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        console.print(f"[green]✓ Memory stored[/green] (id: {data['id']})")
    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print("[red]Error: Could not connect to Cortex server[/red]")
        raise typer.Exit(1)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max results")] = 10,
    include_forgotten: Annotated[
        bool, typer.Option("--include-forgotten", help="Include forgotten memories")
    ] = False,
    exact: Annotated[
        bool, typer.Option("--exact", "-e", help="Exact match (full-text only)")
    ] = False,
    after: Annotated[
        Optional[str], typer.Option("--after", help="Only memories after this date")
    ] = None,
    before: Annotated[
        Optional[str], typer.Option("--before", help="Only memories before this date")
    ] = None,
    date: Annotated[
        Optional[str], typer.Option("--date", "-d", help="Only memories from this date")
    ] = None,
):
    """Search memories."""
    client, headers = get_client()

    payload = {
        "query": query,
        "limit": limit,
        "include_forgotten": include_forgotten,
        "exact": exact,
    }

    # Handle date filters
    if date:
        # --date is shorthand for a single day
        try:
            d = datetime.fromisoformat(date)
            payload["after"] = d.isoformat()
            payload["before"] = (d + timedelta(days=1)).isoformat()
        except ValueError:
            console.print(f"[red]Error: Invalid date format: {date}[/red]")
            raise typer.Exit(1)
    else:
        if after:
            try:
                payload["after"] = datetime.fromisoformat(after).isoformat()
            except ValueError:
                console.print(f"[red]Error: Invalid date format: {after}[/red]")
                raise typer.Exit(1)
        if before:
            try:
                payload["before"] = datetime.fromisoformat(before).isoformat()
            except ValueError:
                console.print(f"[red]Error: Invalid date format: {before}[/red]")
                raise typer.Exit(1)

    try:
        response = client.post("/search", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()

        if not data["memories"]:
            console.print("[dim]No memories found[/dim]")
            return

        for mem in data["memories"]:
            score = mem.get("score")
            score_str = f"[{score:.2f}]" if score else ""
            # Parse UTC timestamp and convert to local time
            dt = pendulum.parse(mem["created_at"]).in_tz(pendulum.local_timezone())
            date_str = dt.format("YYYY-MM-DD")
            console.print(f"[cyan]{score_str}[/cyan] [dim]#{mem['id']}[/dim] ({date_str})")

            # Truncate long content for display
            content = mem["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            console.print(content)
            console.print()

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print("[red]Error: Could not connect to Cortex server[/red]")
        raise typer.Exit(1)


@app.command()
def recent(
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max results")] = 10,
    hours: Annotated[int, typer.Option("--hours", "-h", help="Hours to look back")] = 24,
):
    """Get recent memories."""
    client, headers = get_client()

    try:
        response = client.get(
            "/recent",
            params={"limit": limit, "hours": hours},
            headers=headers,
        )
        response.raise_for_status()
        data = response.json()

        if not data["memories"]:
            console.print("[dim]No recent memories[/dim]")
            return

        for mem in data["memories"]:
            # Parse UTC timestamp and convert to local time
            dt = pendulum.parse(mem["created_at"]).in_tz(pendulum.local_timezone())
            date_str = dt.format("YYYY-MM-DD HH:mm")
            console.print(f"[dim]#{mem['id']}[/dim] ({date_str})")

            content = mem["content"]
            if len(content) > 200:
                content = content[:200] + "..."
            console.print(content)
            console.print()

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print("[red]Error: Could not connect to Cortex server[/red]")
        raise typer.Exit(1)


@app.command()
def health():
    """Check Cortex server health."""
    client, headers = get_client()

    try:
        response = client.get("/health")
        data = response.json()

        status_color = "green" if data["status"] == "healthy" else "red"
        pg_color = "green" if data["postgres"] == "connected" else "red"
        ollama_color = "green" if data["ollama"] == "connected" else "red"

        console.print(f"Status: [{status_color}]{data['status']}[/{status_color}]")
        console.print(f"Postgres: [{pg_color}]{data['postgres']}[/{pg_color}]")
        console.print(f"Ollama: [{ollama_color}]{data['ollama']}[/{ollama_color}]")

        if data["memory_count"] is not None:
            console.print(f"Memories: {data['memory_count']:,}")

    except httpx.ConnectError:
        console.print("[red]Error: Could not connect to Cortex server[/red]")
        raise typer.Exit(1)


@app.command()
def forget(
    memory_id: Annotated[int, typer.Argument(help="Memory ID to forget")],
):
    """Soft-delete a memory."""
    client, headers = get_client()

    try:
        response = client.post("/forget", json={"id": memory_id}, headers=headers)
        response.raise_for_status()
        data = response.json()

        if data["forgotten"]:
            console.print(f"[green]✓ Memory #{memory_id} forgotten[/green]")
        else:
            console.print(f"[yellow]Memory #{memory_id} not found or already forgotten[/yellow]")

    except httpx.HTTPStatusError as e:
        console.print(f"[red]Error: {e.response.status_code} - {e.response.text}[/red]")
        raise typer.Exit(1)
    except httpx.ConnectError:
        console.print("[red]Error: Could not connect to Cortex server[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
