import json
import os

import requests

from app_gradio import app


os.environ["CUDA_VISIBLE_DEVICES"] = ""


TEST_QUERY = "Two dogs playing in the snow"


def test_local_run():
    """A quick test to make sure we can build the app and ping the API locally."""
    backend = app.PredictorBackend()
    frontend = app.make_frontend(fn=backend.run)

    # run the UI without blocking
    frontend.launch(share=False, prevent_thread_lock=True)
    local_url = frontend.local_url
    get_response = requests.get(local_url)
    assert get_response.status_code == 200

    local_api = f"{local_url}api/predict"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"data": ["data:text," + TEST_QUERY]})
    post_response = requests.post(local_api, data=payload, headers=headers)
    assert "error" not in post_response.json()
    assert "data" in post_response.json()
    print("RESPONSE:", post_response.json())
