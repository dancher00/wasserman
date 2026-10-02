"""Freeze the new experiment implementation without changing the existing Marine campaign."""

import shutil

from research_campaign_runtime import BASE, ROOT, read, sha, write

NAMES = [
    "research_campaign_runtime",
    "run_research_core_campaign",
    "run_button_visual_campaign",
    "button_visual_support",
    "button_dp",
    "rollout_button_visual",
    "audit_button_dataset",
    "train_button_act",
    "train_button_dp",
    "publish_button_visual",
    "prepare_research_dataset",
    "rollout_research_visual",
    "train_research_act",
    "train_research_dp",
    "run_research_studies",
    "publish_research_studies",
    "research_conditions",
    "evaluate_water_motor_study",
    "run_water_motor_study",
    "publish_water_motor_study",
    "analyze_marine_policy_failures",
    "with_grounded_presentation",
    "publish_marine_benchmark",
    "verify_valve_visual_results",
    "evaluate_valve_visual",
    "evaluate_hatch_visual",
]


def main():
    target = BASE / "frozen_sources.json"
    if target.exists():
        raise ValueError("Preserve existing freeze")
    for rel, expected in read(ROOT / "artifacts/marine_mechanisms_20260924/acceptance.json")["source_sha256"].items():
        if sha(ROOT / rel) != expected:
            raise ValueError("Existing Marine source changed: " + rel)
    paths = [ROOT / "scripts" / (name + ".py") for name in NAMES]
    paths += [ROOT / "docs/research-core-protocol-20260924.md", ROOT / "scripts/freeze_research_core.py"]
    # Persist source text as well as hashes, since the workspace contains inherited uncommitted work.
    hashes = {}
    for path in paths:
        rel = path.relative_to(ROOT)
        copy = BASE / "frozen_sources" / rel
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copy)
        hashes[str(rel)] = sha(path)
    weights = [ROOT / "checkpoints/wasman_press_button_smooth_seed42.pt"]
    weights += [ROOT / f"logs/{stem}_dp_v1/model_epoch_40.pt" for stem in ["valve", "hatch"]]
    hashes.update({str(p.relative_to(ROOT)): sha(p) for p in weights})
    write(target, hashes)
    print(f"Frozen {len(paths)} sources and {len(weights)} existing teacher/policy checkpoints")


if __name__ == "__main__":
    main()
