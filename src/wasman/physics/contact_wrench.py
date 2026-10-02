"""Resolved PhysX contact moments for a pinned one-body-per-environment sensor.

Keep normal and friction application points separate. A mean contact point times
an aggregate force loses force couples, including couples with zero net force.
No force, geometry or friction coefficient is modified by this reader.
"""

import warp as wp


@wp.kernel
def sum_contact_wrench(
    normal_magnitudes: wp.array(dtype=wp.float32),
    normal_points: wp.array(dtype=wp.vec3f),
    normal_directions: wp.array(dtype=wp.vec3f),
    normal_counts: wp.array2d(dtype=wp.uint32),
    normal_starts: wp.array2d(dtype=wp.uint32),
    friction_forces: wp.array(dtype=wp.vec3f),
    friction_points: wp.array(dtype=wp.vec3f),
    friction_counts: wp.array2d(dtype=wp.uint32),
    friction_starts: wp.array2d(dtype=wp.uint32),
    references: wp.array(dtype=wp.vec3f),
    force_out: wp.array2d(dtype=wp.vec3f),
    torque_out: wp.array2d(dtype=wp.vec3f),
):
    env, target = wp.tid()
    force = wp.vec3f(0.0)
    torque = wp.vec3f(0.0)
    start = wp.int32(normal_starts[env, target])
    count = wp.int32(normal_counts[env, target])
    for offset in range(count):
        index = start + offset
        f = normal_magnitudes[index] * normal_directions[index]
        force += f
        torque += wp.cross(normal_points[index] - references[env], f)
    start = wp.int32(friction_starts[env, target])
    count = wp.int32(friction_counts[env, target])
    for offset in range(count):
        index = start + offset
        f = friction_forces[index]
        force += f
        torque += wp.cross(friction_points[index] - references[env], f)
    force_out[env, target] = force
    torque_out[env, target] = torque


class ResolvedContactWrench:
    """Reuse the stable PhysX buffers refreshed by the existing contact sensor.

    The pinned backend refreshes these buffers in place on every sensor update.
    Normal counts must come from the sensor-owned staged copy: PhysX's friction
    getter overwrites the raw normal count/start buffers. No duplicate per-step
    PhysX reads or host transfers are needed.
    """

    def __init__(self, sensor):
        self.sensor = sensor
        _ = sensor.data
        magnitudes, points, normals, _, _, _ = sensor.contact_view.get_contact_data(dt=sensor._sim_physics_dt)
        friction, friction_points, counts, _ = sensor.contact_view.get_friction_data(dt=sensor._sim_physics_dt)
        self.magnitudes = magnitudes.flatten()
        self.normal_points = points.view(wp.vec3f)
        self.normals = normals.view(wp.vec3f)
        self.friction = friction.view(wp.vec3f)
        self.friction_points = friction_points.view(wp.vec3f)
        self.shape = tuple(counts.shape)
        self.force = wp.zeros(self.shape, dtype=wp.vec3f, device=counts.device)
        self.torque = wp.zeros(self.shape, dtype=wp.vec3f, device=counts.device)

    def evaluate(self, references):
        _ = self.sensor.data
        if references.shape != (self.shape[0], 3):
            raise ValueError("Expected one sensor body per environment")
        wp.launch(sum_contact_wrench, dim=self.shape, inputs=[
            self.magnitudes, self.normal_points, self.normals,
            self.sensor._contact_counts, self.sensor._contact_start_indices,
            self.friction, self.friction_points,
            self.sensor._friction_counts, self.sensor._friction_start_indices,
            wp.from_torch(references.contiguous(), dtype=wp.vec3f),
        ], outputs=[self.force, self.torque], device=self.force.device)
        return wp.to_torch(self.force), wp.to_torch(self.torque)
