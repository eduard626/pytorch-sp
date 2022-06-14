from curses.ascii import SYN
from pathlib import Path
from typing import Dict
from settings import SYN_TMPDIR
import logging
from datasets import synthetic_dataset
import numpy as np
from tqdm import tqdm
from utils.opencv_utils import gaussian_blur, resize_image, write_image
import tarfile
import shutil

def dump_primitive_data(primitive : str, tar_path : str, config : Dict):
    """ Dump synthetic drawings of primitives to disk using tar files

    Args:
        primitive (str): primitive type : cube, line, circle, etc
        tar_path (str): path in disk where data is to be stored
        config (Dict): config parameters    
    """

    temp_dir = Path(SYN_TMPDIR, primitive)

    logging.info("Generating tarfile for primitive {}.".format(primitive))
    synthetic_dataset.set_random_state(
        np.random.RandomState(config["generation"]["random_seed"])
    )
    for split, size in config["generation"]["split_sizes"].items():
        im_dir, pts_dir = [Path(temp_dir, i, split) for i in ["images", "points"]]
        im_dir.mkdir(parents=True, exist_ok=True)
        pts_dir.mkdir(parents=True, exist_ok=True)

        for i in tqdm(range(size), desc=split, leave=False):
            image = synthetic_dataset.generate_background(
                config["generation"]["image_size"],
                **config["generation"]["params"]["generate_background"],
            )
            points = np.array(
                getattr(synthetic_dataset, primitive)(
                    image, **config["generation"]["params"].get(primitive, {})
                )
            )
            points = np.flip(points, 1)  # reverse convention with opencv x,y -> y,x

            b = config["preprocessing"]["blur_size"]
            image = gaussian_blur(image, (b, b), 0)
            points = (
                points
                * np.array(config["preprocessing"]["resize"], np.float)
                / np.array(config["generation"]["image_size"], np.float)
            )
            image = resize_image(image, target_size=tuple(config["preprocessing"]["resize"][::-1]))
            image_name = Path(im_dir, "{}.png".format(i))
            # save image
            write_image(image, image_name)
            # save points (corners)
            points_name = Path(pts_dir, "{}.npy".format(i))
            np.save(points_name, points)

    # Pack into a tar file
    tar = tarfile.open(tar_path, mode="w:gz")
    tar.add(temp_dir, arcname=primitive)
    tar.close()
    shutil.rmtree(temp_dir)
    logging.info("Tarfile dumped to {}.".format(tar_path))

def parse_primitives(names, all_primitives):
    p = (
        all_primitives
        if (names == "all")
        else (names if isinstance(names, list) else [names])
    )
    assert set(p) <= set(all_primitives)
    return p