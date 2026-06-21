"""P3 / doc M2: differentiable physics<->imaging coupling; device-as-sensor."""

import pytest

torch = pytest.importorskip("torch")

from lumen.sensors import Sensor
from lumen.sensors.projective import ProjectiveRenderer
from lumen.physics import imaging


def test_renderer_satisfies_sensor_protocol():
    assert isinstance(ProjectiveRenderer(), Sensor)


def test_render_is_differentiable_in_node_positions():
    r = ProjectiveRenderer(height=32, width=32)
    nodes = torch.tensor([[[0.0, 0.0, 18.0], [2.0, 0.0, 22.0]]],
                         dtype=torch.float64, requires_grad=True)
    img = r.render(nodes)
    assert img.shape == (1, 32, 32)
    img.sum().backward()
    assert nodes.grad is not None and torch.any(nodes.grad != 0)


def test_image_moves_with_the_device():
    r = ProjectiveRenderer(height=48, width=48)
    a = r.render(torch.tensor([[[0.0, 0.0, 18.0]]], dtype=torch.float64))
    b = r.render(torch.tensor([[[4.0, 0.0, 18.0]]], dtype=torch.float64))
    # shifting the node in +x moves the bright spot -> the images differ
    assert float(((a - b) ** 2).mean()) > 1e-3


def test_device_as_sensor_recovers_friction_from_image():
    for mu_true in (0.3, 0.6):
        mu_hat, loss = imaging.calibrate_from_image(mu_true=mu_true, steps=35)
        assert abs(mu_hat - mu_true) < 0.04, (mu_true, mu_hat, loss)
