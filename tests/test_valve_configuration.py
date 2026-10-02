from wasman.assets.bluerov2_alpha import BLUEROV2_ALPHA_CFG
from wasman.tasks.underwater_panel.agents import ValvePPORunnerCfg
from wasman.tasks.underwater_panel.approach import ApproachRotateValveEnvCfg
from wasman.tasks.underwater_panel.env_cfg import UnderwaterRotateValveT200EnvCfg
from wasman.tasks.underwater_press_button.config.bluerov2_alpha.agents.rsl_rl_ppo_cfg import PPORunnerCfg
from wasman.tasks.underwater_press_button.config.bluerov2_alpha.thruster_cfg import ThrusterPressButtonEnvCfg


def test_valve_actor_configuration_does_not_replace_button_actor():
    button_before = PPORunnerCfg()
    valve = ValvePPORunnerCfg()
    button_after = PPORunnerCfg()
    assert valve.actor.class_name == "wasman.controllers.valve_policy:ValveResidualActor"
    assert button_before.actor.class_name == button_after.actor.class_name
    assert button_after.actor.class_name != valve.actor.class_name
    assert button_after.experiment_name != valve.experiment_name


def test_valve_contact_settings_do_not_mutate_released_button_configuration():
    valve = UnderwaterRotateValveT200EnvCfg()
    button = ThrusterPressButtonEnvCfg()
    assert (valve.action_space, valve.observation_space) == (11, 50)
    assert valve.sim.dt * valve.decimation == button.sim.dt * button.decimation
    assert valve.thruster_command_delay_steps * valve.sim.dt == button.thruster_command_delay_steps * button.sim.dt
    assert (button.action_space, button.observation_space) == (10, 37)
    assert button.station_position_ki[-1] == 7.0
    assert button.station_position_kp == (70.0, 70.0, 90.0)
    assert button.station_rotation_kp == (7.0, 8.0, 10.0)
    assert button.scene.robot.init_state.joint_pos["alpha_axis_b"] == 0.15
    assert "alpha_axis_b" in BLUEROV2_ALPHA_CFG.actuators["alpha_arm"].joint_names_expr
    assert button.scene.robot.actuators["alpha_gripper"].joint_effort_limit == 10
    assert valve.scene.robot.actuators["alpha_wrist"].joint_effort_limit == 0.6
    assert valve.scene.robot.soft_joint_pos_limit_factor == 1.0
    assert button.scene.robot.soft_joint_pos_limit_factor == 0.96
    assert valve.valve_initial_wrist - valve.arm_target_scale[-1] == 0.0
    assert not hasattr(button, "link_hydrodynamics")


def test_valve_swim_in_retains_the_short_start_action_reference():
    approach = ApproachRotateValveEnvCfg()
    short = UnderwaterRotateValveT200EnvCfg()
    assert approach.scene.robot.init_state.pos[0] == -0.85
    assert approach.scene.robot.init_state.joint_pos["alpha_axis_c"] == 0.30
    assert short.scene.robot.init_state.joint_pos["alpha_axis_c"] == 1.62
    assert approach.deployed_arm[-1] == short.valve_initial_wrist
    assert approach.base_target_position == short.base_target_position
    assert approach.action_space == short.action_space == 11
    assert approach.observation_space == short.observation_space == 50
    assert approach.valve_success_contract == short.valve_success_contract == "ambench-angle-170-v1"
