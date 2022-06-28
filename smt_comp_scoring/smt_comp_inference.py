"""Anomalib Inferencer Script.

This script performs inference by reading a model config file from
command line, and show the visualization results.
"""

# Copyright (C) 2020 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

import sys
from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.base import Inferencer

sys.path.append("../anomalib_main")  # you should named your anomalib repo as anomalib_main

MODELS: Dict[str, Inferencer] = {}
GROUPS: List[str] = []


def get_args() -> Namespace:
    """Get command line arguments.

    Returns:
        Namespace: List of arguments.
    """
    parser = ArgumentParser()
    # --model_config_path will be deprecated in 0.2.8 and removed in 0.2.9
    parser.add_argument(
        "--smt_comp_model_def", type=Path, required=False, help="Path to JSON file containing the metadata."
    )

    args = parser.parse_args()

    return args


def initialize() -> None:
    """Stream predictions.

    Show/save the output if path is to an image. If the path is a directory, go over each image in the directory.
    """
    # Get the command line arguments, and config from the config.yaml file.
    # This config file is also used for training and contains all the relevant
    # information regarding the data, model, train and inference details.
    args = get_args()
    df_smt_comp_model_def = pd.read_csv(args.smt_comp_model_def)

    groups = df_smt_comp_model_def["Group"].tolist()
    for group in groups:
        GROUPS.append(group)

    for _, row in df_smt_comp_model_def.iterrows():
        group = row["Group"]
        config_path = row["Config_path"]
        weithg_path = row["Weight_path"]
        config = get_configurable_parameters(config_path=config_path)

        # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
        # for the openvino models.
        extension = Path(weithg_path).suffix

        inferencer: Inferencer
        if extension in (".ckpt"):
            module = import_module("anomalib.deploy.inferencers.torch")
            TorchInferencer = getattr(module, "TorchInferencer")  # pylint: disable=invalid-name
            inferencer = TorchInferencer(config=config, model_source=weithg_path, meta_data_path=None)

        elif extension in (".onnx", ".bin", ".xml"):
            module = import_module("anomalib.deploy.inferencers.openvino")
            OpenVINOInferencer = getattr(module, "OpenVINOInferencer")  # pylint: disable=invalid-name
            inferencer = OpenVINOInferencer(config=config, model_source=weithg_path, meta_data_path=None)

        else:
            raise ValueError(
                f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
                f"OpenVINO Inferencer expects either .onnx, .bin or .xml file. Got {extension}"
            )
        MODELS[group] = inferencer


def infer(
    smt_comps_names: Dict, smt_comps_images: Dict
) -> Tuple[
    Dict[str, List],
    Dict[str, List],
    Dict[str, List],
    Dict[str, List],
    Dict[str, List],
    Dict[str, List],
    Dict[str, List],
]:
    """Perform inference on a single image.

    Args:
        smt_comps_names (Dict): group of smt component names
        smt_comps_images (Dict): group of smt component images
    """
    # Perform inference for the given image or image path. if image
    # path is provided, `predict` method will read the image from
    # file for convenience. We set the superimpose flag to True
    # to overlay the predicted anomaly map on top of the input image.

    smt_comp_scocring_image_socres = {}  # type: Dict[str, List]
    smt_comp_scocring_pixel_socres = {}  # type: Dict[str, List]
    smt_comp_image_thresholds = {}  # type: Dict[str, List]
    smt_comp_pixel_thresholds = {}  # type: Dict[str, List]
    smt_comp_scocring_results = {}  # type: Dict[str, List]

    for group in GROUPS:
        inferencer = MODELS[group]
        images = smt_comps_images[group]
        for image in images:
            output = inferencer.predict(image=image, superimpose=True)
            image_threshold = inferencer.get_image_threshold()
            pixel_threshold = inferencer.get_pixel_threshold()

            # Incase both anomaly map and scores are returned add scores to the image.
            if isinstance(output, tuple):
                _, score = output

                if group not in smt_comp_scocring_results:
                    smt_comp_scocring_image_socres[group] = []
                    smt_comp_scocring_pixel_socres[group] = []
                    smt_comp_image_thresholds[group] = []
                    smt_comp_pixel_thresholds[group] = []
                    smt_comp_scocring_results[group] = []

                smt_comp_scocring_image_socres[group].append(score)
                smt_comp_scocring_pixel_socres[group].append(score)
                smt_comp_image_thresholds[group].append(image_threshold)
                smt_comp_pixel_thresholds[group].append(pixel_threshold)
                smt_comp_scocring_results[group].append("defected" if score > pixel_threshold else "good")

    return (
        smt_comps_names,
        smt_comps_images,
        smt_comp_scocring_image_socres,
        smt_comp_scocring_pixel_socres,
        smt_comp_image_thresholds,
        smt_comp_pixel_thresholds,
        smt_comp_scocring_results,
    )


if __name__ == "__main__":
    initialize()
