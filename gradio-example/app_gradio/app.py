"""Provide a text query describing what you are looking for and get back out images with links!"""
import argparse
import logging
import os
import gradio as gr

from pathlib import Path
from typing import Callable, Dict, List, Tuple
from PIL.Image import Image

print(__file__)
import fashion_aggregator.fashion_aggregator as fa


os.environ["CUDA_VISIBLE_DEVICES"] = ""  # do not use GPU

logging.basicConfig(level=logging.INFO)
DEFAULT_APPLICATION_NAME = "fashion-aggregator"

APP_DIR = Path(__file__).resolve().parent  # what is the directory for this application?
FAVICON = APP_DIR / "t-shirt_1f455.png"  # path to a small image for display in browser tab and social media
README = APP_DIR / "README.md"  # path to an app readme file in HTML/markdown

DEFAULT_PORT = 11700


def main(args):
    predictor = PredictorBackend(url=args.model_url)
    frontend = make_frontend(
        predictor.run,
        flagging=args.flagging,
        gantry=args.gantry,
        app_name=args.application
    )
    frontend.launch(
        server_name="0.0.0.0",  # make server accessible, binding all interfaces  # noqa: S104
        server_port=args.port,  # set a port to bind to, failing if unavailable
        share=True,  # should we create a (temporary) public link on https://gradio.app?
        favicon_path=FAVICON,  # what icon should we display in the address bar?
    )


def make_frontend(
    fn: Callable[[Image], str],
    flagging: bool = False,
    gantry: bool = False,
    app_name: str = "fashion-aggregator"
):
    """Creates a gradio.Interface frontend for text to image search function."""

    allow_flagging = "never"
    readme = _load_readme(with_logging=allow_flagging == "manual")

    # build a basic browser interface to a Python function
    frontend = gr.Interface(
        fn=fn,  # which Python function are we interacting with?
        outputs=gr.Gallery(label="Relevant Items"),
        # what input widgets does it need? we configure an image widget
        inputs=gr.components.Textbox(label="Item Description"),
        title="📝 Text2Image 👕",  # what should we display at the top of the page?
        thumbnail=FAVICON,  # what should we display when the link is shared, e.g. on social media?
        description=__doc__,  # what should we display just above the interface?
        article=readme,  # what long-form content should we display below the interface?
        cache_examples=False,  # should we cache those inputs for faster inference? slows down start
        allow_flagging=allow_flagging,  # should we show users the option to "flag" outputs?
        flagging_options=["incorrect", "offensive", "other"],  # what options do users have for feedback?
    )
    return frontend


class PredictorBackend:
    """Interface to a backend that serves predictions.

    To communicate with a backend accessible via a URL, provide the url kwarg.

    Otherwise, runs a predictor locally.
    """

    def __init__(self, url=None):
        if url is not None:
            self.url = url
            self._predict = self._predict_from_endpoint
        else:
            model = fa.Retriever()
            self._predict = model.predict
            self._search_images = model.search_images

    def run(self, text: str):
        pred, metrics = self._predict_with_metrics(text)
        self._log_inference(pred, metrics)
        return pred

    def _predict_with_metrics(self, text: str) -> Tuple[List[str], Dict[str, float]]:
        paths_and_scores = self._search_images(text)
        metrics = {
            "mean_score": sum(paths_and_scores["score"]) / len(paths_and_scores["score"])
        }
        return paths_and_scores["path"], metrics
    
    def _log_inference(self, pred, metrics):
        for key, value in metrics.items():
            logging.info(f"METRIC {key} {value}")
        logging.info(f"PRED >begin\n{pred}\nPRED >end")


def _load_readme(with_logging=False):
    with open(README) as f:
        lines = f.readlines()
        if not with_logging:
            lines = lines[: lines.index("<!-- logging content below -->\n")]

        readme = "".join(lines)
    return readme


def _make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_url",
        default=None,
        type=str,
        help="Identifies a URL to which to send image data. Data is base64-encoded, converted to a utf-8 string, and then set via a POST request as JSON with the key 'image'. Default is None, which instead sends the data to a model running locally.",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        type=int,
        help=f"Port on which to expose this server. Default is {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--flagging",
        action="store_true",
        help="Pass this flag to allow users to 'flag' model behavior and provide feedback.",
    )
    parser.add_argument(
        "--gantry",
        action="store_true",
        help="Pass --flagging and this flag to log user feedback to Gantry. Requires GANTRY_API_KEY to be defined as an environment variable.",
    )
    parser.add_argument(
        "--application",
        default=DEFAULT_APPLICATION_NAME,
        type=str,
        help=f"Name of the Gantry application to which feedback should be logged, if --gantry and --flagging are passed. Default is {DEFAULT_APPLICATION_NAME}.",
    )
    return parser


if __name__ == "__main__":
    parser = _make_parser()
    args = parser.parse_args()
    main(args)
