"""
Benchmark: AsyncBatchTraceWriter throughput.

Run:
    python scripts/bench_trace_writer.py

Measures single-producer and multi-producer writes/s to validate
the high-throughput claim in observability.py.
"""
import asyncio
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from medical_agent.observability import AsyncBatchTraceWriter

SAMPLE_RECORD = {
    "trace_id": "abc123",
    "span": "node_agent",
    "session_id": "s-001",
    "patient_id": "p-001",
    "tokens_in": 512,
    "tokens_out": 256,
    "latency_ms": 843.2,
    "tool_calls": ["search_medical_knowledge"],
    "timestamp": "2026-06-09T10:00:00Z",
}


async def bench_single(n: int, tmpdir: Path) -> float:
    writer = AsyncBatchTraceWriter(path=tmpdir / "single.jsonl", batch_size=500, flush_interval=0.05)
    await writer.start()
    t0 = time.perf_counter()
    for _ in range(n):
        await writer.write(SAMPLE_RECORD)
    await writer.stop()
    elapsed = time.perf_counter() - t0
    return n / elapsed


async def bench_concurrent(n: int, producers: int, tmpdir: Path) -> float:
    writer = AsyncBatchTraceWriter(path=tmpdir / "concurrent.jsonl", batch_size=500, flush_interval=0.05)
    await writer.start()

    async def producer(count: int) -> None:
        for _ in range(count):
            await writer.write(SAMPLE_RECORD)

    t0 = time.perf_counter()
    await asyncio.gather(*[producer(n // producers) for _ in range(producers)])
    await writer.stop()
    elapsed = time.perf_counter() - t0
    return (n // producers * producers) / elapsed


async def main() -> None:
    N = 100_000

    with tempfile.TemporaryDirectory() as d:
        tmpdir = Path(d)

        wps_single = await bench_single(N, tmpdir)
        print(f"Single-producer:       {wps_single:>10,.0f} writes/s  ({N:,} records)")

        for p in (4, 8):
            wps_conc = await bench_concurrent(N, p, tmpdir)
            print(f"{p}-producer concurrent: {wps_conc:>10,.0f} writes/s  ({N:,} records)")

        # Extrapolate to 10M
        hours_to_10M = 10_000_000 / wps_single / 3600
        print(f"\n10M records @ single-producer rate: {hours_to_10M:.2f} hours sustained")
        print("(production multi-instance deployment would be proportionally faster)")


if __name__ == "__main__":
    asyncio.run(main())
