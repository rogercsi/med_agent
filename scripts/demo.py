#!/usr/bin/env python
"""Rich console interactive demo — shows SSE node events and Ragas comparison."""
import asyncio
import json
import sys
from pathlib import Path

import httpx
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

BASE_URL = "http://localhost:8000"
console = Console()


async def chat_stream(patient_id: str, session_id: str, message: str) -> str:
    """Stream a chat message and display SSE events in real time."""
    console.print(f"\n[dim]→ 发送消息: {message[:60]}...[/dim]" if len(message) > 60 else f"\n[dim]→ 发送消息: {message}[/dim]")

    answer = ""
    async with httpx.AsyncClient(timeout=60) as client:
        async with client.stream(
            "POST",
            f"{BASE_URL}/chat/stream",
            json={"patient_id": patient_id, "session_id": session_id, "message": message},
        ) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                    continue

                data_str = line[5:].strip()
                if not data_str:
                    continue

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event = locals().get("current_event", "message")

                if event == "node_start":
                    node = data.get("node", "")
                    console.print(f"  [cyan]▸ 节点:[/cyan] {node}")

                elif event == "emergency":
                    kws = data.get("keywords", [])
                    console.print(
                        f"  [red bold]⚠️  紧急检测触发！关键词: {', '.join(kws)}[/red bold]"
                    )

                elif event == "rag_result":
                    rq = data.get("rewritten_query", "")
                    srcs = data.get("sources", [])
                    tb = data.get("token_before", 0)
                    ta = data.get("token_after", 0)
                    console.print(f"  [yellow]改写查询:[/yellow] {rq}")
                    console.print(f"  [yellow]来源:[/yellow] {', '.join(srcs) or '无'}")
                    if tb and ta:
                        savings = round((1 - ta / tb) * 100)
                        console.print(f"  [yellow]上下文压缩:[/yellow] {tb} → {ta} tokens ({savings}% ↓)")

                elif event == "answer":
                    answer = data.get("text", "")

    if answer:
        console.print(Panel(answer, title="[bold green]回答[/bold green]", border_style="green"))

    return answer


async def show_memory(patient_id: str) -> None:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/chat/memory/{patient_id}")
        if resp.status_code == 200:
            data = resp.json()
            memories = data.get("memories", [])
            if memories:
                table = Table(title=f"患者 {patient_id} 的记忆 ({len(memories)} 条)")
                table.add_column("#", style="dim")
                table.add_column("记忆内容")
                for i, m in enumerate(memories, 1):
                    table.add_row(str(i), m.get("memory", ""))
                console.print(table)
            else:
                console.print("[dim]暂无记忆[/dim]")


async def show_eval_comparison() -> None:
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{BASE_URL}/eval/results")
        if resp.status_code == 200:
            from medical_agent.eval.compare import print_comparison_table
            print_comparison_table(resp.json())
        else:
            console.print(f"[red]评估结果未找到: {resp.text}[/red]")


async def main() -> None:
    console.print(Panel.fit(
        "[bold blue]Medical Knowledge Q&A + Memory Agent[/bold blue]\n"
        "RAG · LangGraph FSM · Mem0 长记忆 · Ragas 评估",
        border_style="blue",
    ))

    patient_id = Prompt.ask("患者ID", default="demo_patient")
    session_id = f"{patient_id}_session_1"

    while True:
        console.print("\n[bold]命令:[/bold] chat / memory / eval / new-session / quit")
        cmd = Prompt.ask("输入命令或直接发送消息", default="chat")

        if cmd == "quit":
            break
        elif cmd == "memory":
            await show_memory(patient_id)
        elif cmd == "eval":
            await show_eval_comparison()
        elif cmd == "new-session":
            import uuid
            session_id = f"{patient_id}_{uuid.uuid4().hex[:8]}"
            console.print(f"[green]新会话: {session_id}[/green]")
        elif cmd == "chat":
            msg = Prompt.ask("您的问题")
            await chat_stream(patient_id, session_id, msg)
        else:
            # Treat as direct message
            await chat_stream(patient_id, session_id, cmd)


if __name__ == "__main__":
    asyncio.run(main())
