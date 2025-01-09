import os
import random
import shutil
import sys
import natsort
from pathlib import Path
from argparse import ArgumentParser
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from visualize.render_mesh import render2mesh
try:
    import bpy
    bpy.app.debug = 0

    sys.path.append(os.path.dirname(bpy.data.filepath))

    # local packages
    sys.path.append(os.path.expanduser("~/.local/lib/python3.9/site-packages"))
except ImportError:
    raise ImportError(
        "Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender."
    )


# Monkey patch argparse such that
# blender / python / hydra parsing works
def parse_args(self, args=None, namespace=None):
    if args is not None:
        return self.parse_args_bak(args=args, namespace=namespace)
    try:
        idx = sys.argv.index("--")
        args = sys.argv[idx + 1:]  # the list after '--'
    except ValueError as e:  # '--' not in the list:
        args = []
    return self.parse_args_bak(args=args, namespace=namespace)


setattr(ArgumentParser, 'parse_args_bak', ArgumentParser.parse_args)
setattr(ArgumentParser, 'parse_args', parse_args)

from configs.config import parse_args


def render_cli() -> None:
    # parse options
    cfg = parse_args(phase="render")  # parse config file
    cfg.FOLDER = cfg.RENDER.FOLDER
    if cfg.RENDER.INPUT_MODE.lower() == "npy":
        output_dir = Path(os.path.dirname(cfg.RENDER.NPY))
        paths = [cfg.RENDER.NPY]
    elif cfg.RENDER.INPUT_MODE.lower() == "dir":
        output_dir = Path(cfg.RENDER.DIR)
        paths = []
        start = cfg.RENDER.START
        end = cfg.RENDER.END
        file_list = natsort.natsorted(os.listdir(cfg.RENDER.DIR))
        if end != 0:
            if end > len(file_list):
                file_list = file_list[start:]
            else:
                file_list = file_list[start:end]

        # render mesh npy first
        for item in file_list:
            if item.endswith(".npy"):
                paths.append(os.path.join(cfg.RENDER.DIR, item))
    print("begin to render {}".format(paths))

    import numpy as np

    from models.render.blender import render
    from models.render.video import Video

    init = True
    for path in paths:
        print(path)
        file_name = os.path.basename(path.replace(".npy", ""))
        # check existed mp4 or under rendering
        if cfg.RENDER.MODE == "video":
            frames_folder = os.path.join('./data/frames_new/{}'.format(cfg.RENDER.TASK_NAME), file_name+"_frames")
            video_folder = cfg.RENDER.SAVE
            os.makedirs(frames_folder,exist_ok=True)
            os.makedirs(video_folder,exist_ok=True)
            if os.path.exists(os.path.join(video_folder,file_name+".mp4")):
                print(f"npy is rendered or under rendering {path}")
                continue
        else:
            # check existed png
            picture_folder = cfg.RENDER.SAVE
            if os.path.exists(os.path.join(picture_folder, file_name+".png")):
                print(f"npy is rendered or under rendering {path}")
                continue
        # prepare frame folder
        if cfg.RENDER.MODE == "video":
            os.makedirs(frames_folder, exist_ok=True)
        else:
            frames_folder = os.path.join(
                picture_folder,
                file_name+".png")

        try:
            if cfg.RENDER.MESH:
                print("Convert joints to mesh .......")
                mesh_dir = './data/mesh/{}'.format(cfg.RENDER.TASK_NAME)
                os.makedirs(mesh_dir,exist_ok=True)
                data = render2mesh(path, outdir=mesh_dir, device_id=cfg.RENDER.DEVICE)
            else:
                data = np.load(path)
                
                    # 确保数据至少有3个维度注释掉 
            # if data.shape[0] == 1:
            #     data = data[0]
        
            if data.ndim == 2:
                data = data[np.newaxis, ...]
        except FileNotFoundError:
            print(f"{path} not found")
            continue
        

        out = render(
            data,
            frames_folder,
            canonicalize=cfg.RENDER.CANONICALIZE,
            exact_frame=cfg.RENDER.EXACT_FRAME,
            num=cfg.RENDER.NUM,
            mode=cfg.RENDER.MODE,
            model_path=cfg.RENDER.MODEL_PATH,
            faces_path=cfg.RENDER.FACES_PATH,
            downsample=cfg.RENDER.DOWNSAMPLE,
            always_on_floor=cfg.RENDER.ALWAYS_ON_FLOOR,
            oldrender=cfg.RENDER.OLDRENDER,
            res=cfg.RENDER.RES,
            init=init,
            gt=cfg.RENDER.GT,
            accelerator=cfg.ACCELERATOR,
            device=cfg.RENDER.DEVICE,
        )

        init = False

        if cfg.RENDER.MODE == "video":
            shutil.copytree(frames_folder, frames_folder+'_img') 
            if cfg.RENDER.DOWNSAMPLE:
                video = Video(frames_folder, fps=cfg.RENDER.FPS)
            else:
                video = Video(frames_folder, fps=cfg.RENDER.FPS)

            vid_path = os.path.join(video_folder,file_name+".mp4")
            video.save(out_path=vid_path)
            shutil.rmtree(frames_folder)
            print(f"remove tmp fig folder and save video in {vid_path}")

        else:
            print(f"Frame generated at: {out}")


if __name__ == "__main__":
    render_cli()
