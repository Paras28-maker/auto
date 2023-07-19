import pathlib
import traceback

from PIL import Image
import numpy as np
import os

from src import core
from src import backbone
from src.common_constants import GenerationOptions as go


def open_path_as_images(path):
    """Takes the filepath, returns (fps, frames). Every frame is a Pillow Image object"""
    suffix = pathlib.Path(path).suffix
    if suffix == '.gif':
        frames = []
        img = Image.open(path)
        for i in range(img.n_frames):
            img.seek(i)
            frames.append(img.convert('RGB'))
        return 1000 / img.info['duration'], frames
    elif suffix == '.webm':
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(path)
        frames = [Image.fromarray(x) for x in list(clip.iter_frames())]
        # TODO: Wrapping frames into Pillow objects is wasteful
        return clip.fps, frames
    else:
        return 1000, [Image.open(path)]


def frames_to_video(fps, frames, path, name):
    if frames[0].mode == 'I;16':
        print('WARNING! Video will be converted to 24-bit RGB, precision is lost!')
        frames = [frame.point(lambda p: p * 0.0039063096, mode='RGB').convert('RGB') for frame in frames]

    arrs = [np.asarray(frame) for frame in frames]
    from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
    clip = ImageSequenceClip(arrs, fps=fps)

    try:
        clip.write_videofile(os.path.join(path, f"{name}.avi"), codec='png')
    except Exception as e:
        traceback.print_exc()
        try:
            print("Failed to save .avi (png), trying mp4 (rawvideo)")
            clip.write_videofile(os.path.join(path, f"{name}.avi"), codec='rawvideo')
        except Exception as e:
            traceback.print_exc()
            print("Failed to save .avi (rawvideo), trying webm")
            clip.write_videofile(os.path.join(path, f"{name}.webm"), codec='webm')



def launch(video, outpath, inp):
    if inp[go.GEN_SIMPLE_MESH.name.lower()] or inp[go.GEN_INPAINTED_MESH.name.lower()]:
        return 'Creating mesh-videos is not supported. Please split video into frames and use batch processing.'

    fps, input_images = open_path_as_images(os.path.abspath(video.name))
    os.makedirs(backbone.get_outpath(), exist_ok=True)

    needed_keys = [go.COMPUTE_DEVICE, go.MODEL_TYPE, go.BOOST, go.NET_SIZE_MATCH, go.NET_WIDTH, go. NET_HEIGHT]
    needed_keys = [x.name.lower() for x in needed_keys]
    first_pass_inp = {k: v for (k, v) in inp.items() if k in needed_keys}
    first_pass_inp[go.DO_OUTPUT_DEPTH_PREDICTION] = True
    first_pass_inp[go.DO_OUTPUT_DEPTH.name] = False

    print('Generating depthmaps for the video frames')
    gen_obj = core.core_generation_funnel(None, input_images, None, None, first_pass_inp)
    predictions = [x[2] for x in list(gen_obj)]

    print('Processing generated depthmaps')
    # TODO: Smart normalizing (drop 0.001% of top and bottom values from the video/every cut)
    preds_min_value = min([pred.min() for pred in predictions])
    preds_max_value = max([pred.max() for pred in predictions])

    input_depths = []
    for pred in predictions:
        norm = (pred - preds_min_value) / (preds_max_value - preds_min_value)  # normalize to [0; 1]
        input_depths += [norm]
    # TODO: Smoothening between frames (use splines)
    # TODO: Detect cuts and process segments separately

    print('Generating output frames')
    img_results = list(core.core_generation_funnel(None, input_images, input_depths, None, inp))
    gens = list(set(map(lambda x: x[1], img_results)))

    print('Saving generated frames as video outputs')
    for gen in gens:
        imgs = [x[2] for x in img_results if x[1] == gen]
        basename = f'{gen}_video'
        frames_to_video(fps, imgs, outpath, f"{backbone.get_next_sequence_number()}-{basename}")
    print('All done. Video(s) saved!')
    return 'Video generated!'
