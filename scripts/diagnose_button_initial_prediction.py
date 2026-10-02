"""Replay two saved Button first observations on one GPU to separate image and inference differences."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wasman.learning.marine_visual_dataset import normalize_act_batch
from wasman.learning.revision_protocol import configure_inference, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    if config["task"] != "PressButton" or config["model"] not in ("ACT", "BC"):
        raise ValueError("This diagnostic supports the deterministic Button ACT/BC first prediction")
    if config["model"] == "ACT":
        from train_button_act import make_actor

        actor = make_actor(pretrained=False)
    else:
        from wasman.learning.chunk_bc import make_bc

        actor = make_bc("PressButton")
    actor.load_state_dict(checkpoint["state_dict"])
    configure_inference(actor, config)
    actor.cuda().eval()
    stats = checkpoint["statistics"]
    outputs, images, states, recorded, records = [], [], [], [], []
    for folder in (args.left, args.right):
        report = json.loads((folder / "report.json").read_text())
        if report["checkpoint_sha256"] != sha256(args.checkpoint):
            raise ValueError("Checkpoint/report identity mismatch")
        rgb = np.load(folder / "initial_rgb.npz")["rgb"]
        prediction = torch.load(folder / "predictions.pt", weights_only=True, map_location="cpu")[0]
        if prediction["step"] != 0:
            raise ValueError("Missing first prediction")
        images.append(rgb)
        states.append(prediction["input_state"])
        recorded.append(prediction["chunk"])
        batch = normalize_act_batch(
            {
                "observation.images.wrist": torch.from_numpy(rgb).permute(0, 3, 1, 2),
                "observation.state": prediction["input_state"],
            },
            stats,
            "cuda",
        )
        torch.manual_seed(config["rollout_seed"])
        with torch.inference_mode():
            output = (
                actor.predict_action_chunk(batch).float() * stats["action_std"].cuda() + stats["action_mean"].cuda()
            ).cpu()
        outputs.append(output)
        difference = output - prediction["chunk"]
        records.append(
            dict(
                folder=str(folder),
                initial_rgb_sha256=sha256(folder / "initial_rgb.npz"),
                predictions_sha256=sha256(folder / "predictions.pt"),
                recomputed_minus_recorded_max_abs=float(difference.abs().max()),
                recomputed_minus_recorded_mean_abs=float(difference.abs().mean()),
                recomputed_bitwise_equal_to_recorded=bool(torch.equal(output, prediction["chunk"])),
            )
        )
    if not torch.equal(states[0], states[1]):
        raise ValueError("Image-only contrast requires equal initial measured states")
    delta = outputs[0] - outputs[1]
    recorded_delta = recorded[0] - recorded[1]
    result = dict(
        scope="Post hoc first-observation diagnostic on saved inputs; no full-rollout causal attribution",
        checkpoint_sha256=sha256(args.checkpoint),
        gpu=torch.cuda.get_device_name(),
        initial_states_equal=True,
        rgb_mean_absolute_channel_difference=float(np.abs(images[0].astype(float) - images[1].astype(float)).mean()),
        image_only_prediction_max_abs=float(delta.abs().max()),
        image_only_prediction_mean_abs=float(delta.abs().mean()),
        image_contrast_minus_recorded_contrast_max_abs=float((delta - recorded_delta).abs().max()),
        inputs=records,
        script_sha256=sha256(__file__),
    )
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
