"""Execute the user's five-part scope after the existing Marine campaign finishes."""

import argparse
import fcntl
import os
import subprocess
import time

from research_campaign_runtime import BASE, ROOT, frozen_sources, read, run, sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--marine-driver-pid", type=int, default=2597495)
    p.add_argument("--skip-wait", action="store_true", help="Only when Marine completion artifact already exists")
    a = p.parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    lock = (BASE / "campaign.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    wait_start = time.time()
    while True:
        status = read(ROOT / "artifacts/marine_mechanisms_20260924/campaign.json")
        if status["status"] == "complete":
            break
        if a.skip_wait:
            raise ValueError("Marine campaign not complete")
        if status["status"] == "failed":
            raise RuntimeError("Marine campaign failed; diagnose before dependent work")
        os.kill(a.marine_driver_pid, 0)
        if time.time() - wait_start > 48 * 3600:
            raise TimeoutError("Marine completion not received within48h")
        write(
            BASE / "campaign.json", dict(status="waiting", stage="existing_marine_campaign", updated_unix=time.time())
        )
        time.sleep(30)
    frozen_sources()
    for task in ["PushSlider", "PullLever"]:
        run(
            f"{task}_failure_localization",
            ["scripts/analyze_marine_policy_failures.py", "--task", task, "--output-dir", BASE / "final_diagnosis"],
            BASE / "final_diagnosis" / f"{task}_failure_analysis.json",
        )
    write(BASE / "campaign.json", dict(status="running", stage="PressButton", updated_unix=time.time()))
    run("PressButton_campaign", ["scripts/run_button_visual_campaign.py"], BASE / "PressButton/published.json")
    write(BASE / "campaign.json", dict(status="running", stage="water_motor", updated_unix=time.time()))
    run("water_motor_campaign", ["scripts/run_water_motor_study.py"], BASE / "water_motor_published.json")
    write(BASE / "campaign.json", dict(status="running", stage="interface_scaling", updated_unix=time.time()))
    run("interface_scaling", ["scripts/run_research_studies.py"], BASE / "studies_published.json")
    write(BASE / "campaign.json", dict(status="verifying", stage="website", updated_unix=time.time()))
    with (BASE / "website_checks.log").open("x") as log:
        subprocess.run(
            [
                "npm",
                "--prefix",
                "website",
                "run",
                "test",
                "--",
                "tests/benchmark-studies.spec.ts",
                "tests/marine-mechanisms.spec.ts",
                "tests/research-core.spec.ts",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    write(
        BASE / "campaign.json",
        dict(
            status="complete",
            stage="revised_authorized_scope",
            completed_unix=time.time(),
            proofs={
                name: sha(BASE / name)
                for name in ["PressButton/published.json", "water_motor_published.json", "studies_published.json"]
            },
            website="local/private",
            hardware_claim=False,
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write(BASE / "campaign.json", dict(status="failed", error=repr(error), updated_unix=time.time()))
        raise
