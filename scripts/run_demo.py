"""
run_demo.py  —  Interactive demo of the MASA-SVD system.

Tests 5 hand-crafted cases covering all hallucination types.
Usage:  python scripts/run_demo.py [--fast]
        --fast  disables NLI model (instant, no download)
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import MASASVDPipeline

DEMO_CASES = [
    {
        "label":    "Known Factual Hallucination",
        "query":    "Who won the 2022 FIFA World Cup?",
        "response": "France won the 2022 FIFA World Cup, defeating Brazil 3-0 in the final held in Qatar.",
        "expected": "HALLUCINATED",
    },
    {
        "label":    "Correct Response",
        "query":    "Who won the 2022 FIFA World Cup?",
        "response": "Argentina won the 2022 FIFA World Cup, defeating France 4-2 on penalties after a 3-3 draw. The final was held at Lusail Stadium in Qatar on December 18, 2022.",
        "expected": "VERIFIED",
    },
    {
        "label":    "Topic Drift (Off-Topic Response)",
        "query":    "Explain how SVD works in linear algebra.",
        "response": "Machine learning is a rapidly growing field with applications in healthcare, finance, and transportation.",
        "expected": "HALLUCINATED",
    },
    {
        "label":    "Fabricated Citation",
        "query":    "What did Einstein say about time?",
        "response": "According to Einstein's 1935 paper 'On the Nature of Temporal Perception' published in Physical Review Volume 48, time is an illusion created by the human brain.",
        "expected": "HALLUCINATED",
    },
    {
        "label":    "Correct Science Explanation",
        "query":    "What is photosynthesis?",
        "response": "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce oxygen and energy in the form of glucose. It occurs in the chloroplasts of plant cells.",
        "expected": "VERIFIED",
    },
]

VERDICT_ICONS = {
    "VERIFIED":     "✅",
    "UNCERTAIN":    "⚠️ ",
    "HALLUCINATED": "❌",
}


def _bar(risk: float, width: int = 30) -> str:
    filled = int(risk * width)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {risk:.2f}"


def run_demo(use_nli: bool) -> None:
    print("\n" + "=" * 65)
    print("  MASA-SVD: Multi-Agent Hallucination Detection System")
    print(f"  Mode: {'Full (SVD + FVA + NLI)' if use_nli else 'Fast (SVD + FVA, no NLI)'}")
    print("=" * 65)

    pipeline = MASASVDPipeline(use_nli=use_nli)

    passed = 0
    for i, case in enumerate(DEMO_CASES, 1):
        print(f"\n{'─'*65}")
        print(f"TEST {i}: {case['label']}")
        print(f"  Query    : {case['query']}")
        print(f"  Response : {case['response'][:80]}{'…' if len(case['response'])>80 else ''}")

        r = pipeline.analyze(case["query"], case["response"], use_nli=use_nli)

        icon    = VERDICT_ICONS.get(r["verdict"], "?")
        correct = "✓" if r["verdict"] == case["expected"] else "✗ (unexpected)"

        print(f"\n  ┌─── RESULT {'─'*44}┐")
        print(f"  │  Verdict     : {icon} {r['verdict']:<30}        │")
        print(f"  │  Risk Score  : {_bar(r['final_risk_score']):<40}│")
        print(f"  │  SDS         : {r['breakdown']['sds']:.4f}  "
              f"FVA: {r['breakdown']['fva_score']:.4f}  "
              f"SCA: {r['breakdown']['sca_score']:.4f}           │")
        print(f"  │  Expected    : {case['expected']:<10}  Match: {correct:<25}│")
        print(f"  └{'─'*59}┘")

        if r["verdict"] == case["expected"]:
            passed += 1

    print(f"\n{'='*65}")
    print(f"  Demo complete: {passed}/{len(DEMO_CASES)} predictions matched expected.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Skip NLI model (faster, no download)")
    args = parser.parse_args()
    run_demo(use_nli=not args.fast)