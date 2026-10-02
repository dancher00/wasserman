import torch

from wasman.controllers.shell_collection_contract import ShellCollectionContract


def sample(x=0.0, z=0.0125, grasp=False):
    position = torch.tensor([[[x, 0.0, z]]])
    corners = torch.tensor(
        [
            [
                [
                    [-0.0375, -0.0275, -0.0125],
                    [0.0375, 0.0275, 0.0125],
                    [-0.0375, 0.0275, -0.0125],
                    [0.0375, -0.0275, 0.0125],
                ]
            ]
        ]
    )
    return dict(
        positions=position,
        bounds=position[:, :, None, :] + corners,
        linear_velocity=torch.zeros(1, 1, 3),
        angular_velocity=torch.zeros(1, 1, 3),
        finger_forces=torch.tensor([[[[1.0, 0, 0], [-1.0, 0, 0]]]]) * grasp,
        tool_distance=torch.tensor([[0.01 if grasp else 0.20]]),
        hoop_center_xy=torch.zeros(1, 2),
        hoop_inner_radius=0.28,
        floor_z=0.0,
    )


def lifted_and_carried():
    contract = ShellCollectionContract(1, 1, "cpu", 1 / 30)
    contract.update(**sample(x=-0.40, z=0.07, grasp=True))
    contract.update(**sample(x=-0.15, z=0.07, grasp=True))
    return contract


def test_current_drift_or_pushing_into_hoop_does_not_count():
    contract = ShellCollectionContract(1, 1, "cpu", 1 / 30)
    for _ in range(60):
        assert not contract.update(**sample())["success"].item()


def test_two_contacts_on_the_upper_surface_are_not_an_opposing_grasp():
    contract = ShellCollectionContract(1, 1, "cpu", 1 / 30)
    for x in (-0.4, -0.2, 0.0):
        state = sample(x=x, z=0.1, grasp=True)
        state["finger_forces"] = torch.tensor([[[[0.0, 5.0, 4.0], [0.0, -4.0, 6.0]]]])
        result = contract.update(**state)
        assert not result["grasped"].any()
        assert not result["lifted"].any()
        assert not result["transported"].any()


def test_pick_carry_release_and_one_second_rest_pass_then_reset_clears_history():
    contract = lifted_and_carried()
    for _ in range(29):
        assert not contract.update(**sample())["success"].item()
    assert contract.update(**sample())["success"].item()
    contract.reset(torch.tensor([0]))
    assert not contract.transported.any()
    assert not contract.update(**sample())["success"].item()


def test_edge_overlap_suspension_grip_and_instability_do_not_count():
    cases = [sample(x=0.25), sample(z=0.05), sample(grasp=True), sample()]
    cases[-1]["linear_velocity"][0, 0, 0] = 0.03
    for state in cases:
        contract = lifted_and_carried()
        for _ in range(35):
            result = contract.update(**state)
            assert not result["placed"].any()
            assert not result["success"].any()


def test_ungrasped_motion_between_contacts_is_not_carried_distance():
    contract = ShellCollectionContract(1, 1, "cpu", 1 / 30)
    contract.update(**sample(x=-0.40, z=0.07, grasp=True))
    contract.update(**sample(x=-0.20, z=0.07, grasp=False))
    contract.update(**sample(x=0, z=0.07, grasp=True))
    assert not contract.transported.any()


def test_all_objects_must_be_placed_simultaneously():
    contract = ShellCollectionContract(1, 3, "cpu", 1 / 30)
    contract.transported[:] = True  # Isolate the final-arrangement requirement.
    state = sample()
    for key in ("positions", "bounds", "linear_velocity", "angular_velocity", "finger_forces", "tool_distance"):
        state[key] = state[key].expand(1, 3, *state[key].shape[2:]).clone()
    state["bounds"][:, 2, :, 0] += 0.5
    for _ in range(35):
        result = contract.update(**state)
        assert result["placed"].sum() == 2
        assert not result["success"].any()
