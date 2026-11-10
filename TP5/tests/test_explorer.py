import base64
import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer

import numpy as np
import pytest

from src.data.dataset import Dataset
from src.explorer.__main__ import build_handler
from src.explorer.latent_map import LatentMap
from src.explorer.session import ExplorerSession, InvalidRequest
from src.models.factory import build_model
from src.options import Options

IMAGE_SHAPE = (3, 2)


def test_map_round_trip_recovers_every_sample_code():
    codes = np.random.default_rng(0).standard_normal((30, 6))
    latent_map = LatentMap(codes)
    for code, position in zip(codes, latent_map.points):
        np.testing.assert_allclose(latent_map.lift(position), code, atol=1e-9)


def test_map_between_samples_blends_their_codes():
    codes = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 3.0], [0.0, 2.0, 1.0], [2.0, 2.0, 3.0]])
    latent_map = LatentMap(codes, neighbors=2)
    midpoint = (latent_map.points[0] + latent_map.points[1]) / 2
    np.testing.assert_allclose(latent_map.lift(midpoint), [1.0, 0.0, 2.0], atol=1e-9)


def test_two_dimensional_latent_is_its_own_map():
    codes = np.random.default_rng(1).standard_normal((5, 2))
    latent_map = LatentMap(codes)
    np.testing.assert_allclose(latent_map.points, codes)
    np.testing.assert_allclose(latent_map.lift(np.array([0.3, -0.7])), [0.3, -0.7])


def test_one_dimensional_latent_is_rejected():
    with pytest.raises(ValueError):
        LatentMap(np.zeros((4, 1)))


@pytest.fixture
def session():
    rng = np.random.default_rng(2)
    samples = rng.integers(0, 2, (5, 6)).astype(float)
    dataset = Dataset(samples, list("abcde"), ["x"] * 5, IMAGE_SHAPE, is_binary=True)
    model = build_model(
        Options({"latent_size": 3, "hidden_layers": [4]}, "model"), 6, rng, rng
    )
    return ExplorerSession(model, dataset)


def _image(response):
    return np.frombuffer(base64.b64decode(response["image"]), dtype=np.uint8)


def test_session_describes_points_and_decodes_positions(session):
    description = session.describe()
    assert len(description["points"]) == 5
    assert description["image_shape"] == [3, 2]
    assert description["has_audio"] is False

    first = description["points"][0]
    response = session.decode({"position": [first["x"], first["y"]]})
    assert _image(response).shape == (6,)
    assert response["position"] == pytest.approx([first["x"], first["y"]])
    assert "audio" not in response


def test_session_interpolation_endpoints_match_direct_decoding(session):
    points = session.describe()["points"]
    origin = [points[0]["x"], points[0]["y"]]
    destination = [points[1]["x"], points[1]["y"]]
    at_origin = session.decode({"from": origin, "to": destination, "fraction": 0.0})
    np.testing.assert_array_equal(_image(at_origin), _image(session.decode({"position": origin})))
    at_end = session.decode({"from": origin, "to": destination, "fraction": 1.0})
    np.testing.assert_array_equal(_image(at_end), _image(session.decode({"position": destination})))


@pytest.mark.parametrize(
    "request_body",
    [
        {},
        [],
        {"position": [1.0]},
        {"position": ["a", 2]},
        {"position": [float("nan"), 0.0]},
        {"position": [1e12, 0.0]},
        {"from": [0, 0], "to": [1, 1], "fraction": 2},
        {"random": "yes"},
    ],
)
def test_session_rejects_malformed_requests(session, request_body):
    with pytest.raises(InvalidRequest):
        session.decode(request_body)


def test_http_api_serves_page_state_and_errors(session):
    server = HTTPServer(("127.0.0.1", 0), build_handler(session))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    address = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(address + "/") as response:
            assert b"<canvas" in response.read()
        with urllib.request.urlopen(address + "/api/state") as response:
            assert len(json.load(response)["points"]) == 5

        def post(body: bytes):
            request = urllib.request.Request(address + "/api/decode", data=body, method="POST")
            return urllib.request.urlopen(request)

        with post(b'{"random": true}') as response:
            assert "image" in json.load(response)
        with pytest.raises(urllib.error.HTTPError) as failure:
            post(b"not json")
        assert failure.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as failure:
            post(b" " * 10000)
        assert failure.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
