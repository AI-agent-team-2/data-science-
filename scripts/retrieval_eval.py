from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running the script from any directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.rag.retriever import ChromaRetriever  # noqa: E402


TEST_CASES = [
    {
        "query": "ondo pressure reducer",
        "expected_any": [
            "ondo_pressure_reducer_rag_ready.txt",
            "ondo_pressure_reducer_with_gauge_rag_ready.txt",
            "optima_pressure_reducer_rag_ready.txt",
        ],
    },
    {
        "query": "ondo pump group",
        "expected_any": [
            "ondo_pump_group_rag_ready.txt",
            "ondo_pump_fast_mount_group_rag_ready.txt",
            "ondo_pump_mixing_group_rag_ready.txt",
        ],
    },
    {
        "query": "ondo room thermostat",
        "expected_any": ["ondo_room_thermostat_rag_ready.txt"],
    },
    {
        "query": "ondo thermoelectric actuator",
        "expected_any": ["ondo_thermoelectric_actuator_rag_ready.txt"],
    },
    {
        "query": "ondo zone controller",
        "expected_any": ["ondo_zone_controller_rag_ready.txt"],
    },
    {
        "query": "ondo circulation pump",
        "expected_any": [
            "ondo_circulation_pumps_rag_ready.txt",
            "ondo_pressure_boost_pump_rag_ready.txt",
        ],
    },
    {
        "query": "ondo pressure boost pump",
        "expected_any": ["ondo_pressure_boost_pump_rag_ready.txt"],
    },
    {
        "query": "ondo distribution cabinet",
        "expected_any": ["ondo_distribution_cabinet_rag_ready.txt"],
    },
    {
        "query": "ondo manifold rr",
        "expected_any": ["ondo_manifold_rr_rag_ready.txt"],
    },
    {
        "query": "ondo manifold nr",
        "expected_any": ["ondo_manifold_nr_rag_ready.txt"],
    },
    {
        "query": "ondo manifold airvent",
        "expected_any": ["ondo_manifold_airvent_rag_ready.txt"],
    },
    {
        "query": "ondo pex evoh pipe",
        "expected_any": [
            "ondo_pex_evoh_pipe_rag_ready.txt",
            "stm_pex_evoh_underfloor_pipe_rag_ready.txt",
        ],
    },
    {
        "query": "ondo water hammer compensator",
        "expected_any": [
            "ondo_water_hammer_compensator_rag_ready.txt",
            "ondo_water_hammer_compensator_stainless_rag_ready.txt",
        ],
    },
    {
        "query": "ondo ball valves",
        "expected_any": ["ondo_ball_valves_rag_ready.txt"],
    },
    {
        "query": "atlasplast sewer pipes fittings",
        "expected_any": ["atlasplast_sewer_pipes_fittings_rag_ready.txt"],
    },
    {
        "query": "gas black hose",
        "expected_any": ["gas_black_hose_rag_ready.txt"],
    },
    {
        "query": "stm gas ball valve",
        "expected_any": ["stm_gas_ball_valve_rag_ready.txt", "stm_gas_ball_valves_rag_ready.txt"],
    },
    {
        "query": "stm pressure reducer",
        "expected_any": [
            "stm_pressure_reducer_rag_ready.txt",
            "stm_termo_pressure_reducer_rag_ready.txt",
            "optima_pressure_reducer_rag_ready.txt",
        ],
    },
    {
        "query": "stm safety group",
        "expected_any": [
            "stm_safety_group_rag_ready.txt",
            "ondo_boiler_safety_group_rag_ready.txt",
        ],
    },
    {
        "query": "stm pressure gauge",
        "expected_any": ["stm_pressure_gauges_rag_ready.txt"],
    },
    {
        "query": "stm thermostatic head",
        "expected_any": [
            "stm_thermostatic_head_rag_ready.txt",
            "stm_thermostatic_heads_rag_ready.txt",
        ],
    },
    {
        "query": "stm three way valve",
        "expected_any": [
            "stm_three_way_valve_rag_ready.txt",
            "stm_three_way_ball_valve_appliance_rag_ready.txt",
        ],
    },
    {
        "query": "stm check valve",
        "expected_any": ["stm_check_valve_rag_ready.txt"],
    },
    {
        "query": "stm polypropylene pipes",
        "expected_any": ["stm_polypropylene_pipes_rag_ready.txt"],
    },
    {
        "query": "stm polypropylene fittings",
        "expected_any": ["stm_polypropylene_fittings_rag_ready.txt"],
    },
    {
        "query": "stm threaded fittings",
        "expected_any": ["stm_threaded_fittings_rag_ready.txt"],
    },
    {
        "query": "optima gas ball valves",
        "expected_any": [
            "optima_gas_ball_valves_rag_ready.txt",
            "stm_gas_ball_valves_rag_ready.txt",
        ],
    },
    {
        "query": "optima ball valves",
        "expected_any": [
            "optima_ball_valves_rag_ready.txt",
            "ondo_ball_valves_rag_ready.txt",
        ],
    },
    {
        "query": "rispa collector cabinets",
        "expected_any": ["rispa_collector_cabinets_rag_ready.txt"],
    },
    {
        "query": "roegen wall mixers",
        "expected_any": ["roegen_wall_mixers_rag_ready.txt"],
    },
    {
        "query": "roegen tabletop mixers",
        "expected_any": ["roegen_tabletop_mixers_rag_ready.txt"],
    },
    {
        "query": "roegen heater mixer",
        "expected_any": ["roegen_rt053a_heater_mixer_rag_ready.txt"],
    },
    {
        "query": "roegen garden mixer",
        "expected_any": ["roegen_garden_mixer_rag_ready.txt"],
    },
    {
        "query": "welding machine optima",
        "expected_any": [
            "welding_machine_optima_600w_rag_ready.txt",
            "welding_machine_optima_800w_rag_ready.txt",
        ],
    },
    {
        "query": "welding machine stm",
        "expected_any": ["welding_machine_stm_cpwm215c_rag_ready.txt"],
    },
    {
        "query": "ondo manifold group",
        "expected_any": [
            "ondo_manifold_group_rag_ready.txt",
            "ondo_manifold_rr_rag_ready.txt",
            "ondo_manifold_nr_rag_ready.txt",
            "ondo_manifold_airvent_rag_ready.txt",
        ],
    },
    {
        "query": "boiler safety group ondo",
        "expected_any": [
            "ondo_boiler_safety_group_rag_ready.txt",
            "ondo_boiler_safety_group_pass_through_rag_ready.txt",
        ],
    },
]


def _normalize(value: str) -> str:
    return value.lower().replace("\\", "/").strip()


def evaluate(top_k: int) -> int:
    retriever = ChromaRetriever()

    total = len(TEST_CASES)
    hits = 0
    misses: list[dict] = []

    print(f"Running retrieval eval: {total} cases, top_k={top_k}")
    print("-" * 72)

    for idx, case in enumerate(TEST_CASES, start=1):
        query = case["query"]
        expected = [_normalize(x) for x in case["expected_any"]]

        results = retriever.search(query=query, top_k=top_k)
        sources = [_normalize(str((item.get("metadata") or {}).get("source", ""))) for item in results]
        found = any(any(exp in src for src in sources) for exp in expected)

        if found:
            hits += 1
            status = "HIT"
        else:
            status = "MISS"
            misses.append(
                {
                    "query": query,
                    "expected_any": expected,
                    "got_sources": sources,
                }
            )

        print(f"[{idx:02d}] {status} | query='{query}'")

    hit_rate = hits / total if total else 0.0
    print("-" * 72)
    print(f"hits={hits}/{total} | hit@{top_k}={hit_rate:.2%}")

    if misses:
        print("\nMiss details:")
        for miss in misses:
            print(f"- query: {miss['query']}")
            print(f"  expected_any: {miss['expected_any']}")
            print(f"  got_sources: {miss['got_sources'][:top_k]}")

    # Always return success for now: this script is for lightweight monitoring.
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple hit@k evaluation for RAG retrieval.")
    parser.add_argument("--top-k", type=int, default=6, help="Number of retrieved chunks per query.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    top_k = max(1, int(args.top_k))
    return evaluate(top_k=top_k)


if __name__ == "__main__":
    raise SystemExit(main())
