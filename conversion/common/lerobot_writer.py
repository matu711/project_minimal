"""One LeRobot v2.1 writer for both robots, with uncompressed temporary PNGs."""
import numpy as np
from PIL import Image
from conversion.common.image_utils import read_rgb


class _PNGWriter:
    def save_image(self, image, fpath):
        Image.fromarray(image).save(fpath, compress_level=0)

    def wait_until_done(self):
        pass

    def stop(self):
        pass


def write(root, repo_id, spec, episodes, timings, resize=None):
    from lerobot.common.datasets import lerobot_dataset as api
    if api.CODEBASE_VERSION != 'v2.1':
        raise RuntimeError('Conversion requires the pinned LeRobot v2.1 writer')
    writer, summaries, shapes = None, [], {}
    try:
        for episode in episodes:
            from conversion.common.validation import check_episode
            check_episode(episode, spec)
            if summaries and episode.fps != summaries[0]['fps']:
                raise ValueError('Conflicting episode FPS')
            for row in range(len(episode.states)):
                with timings.measure('image_decode'):
                    images = {}
                    for camera, values in episode.images.items():
                        value = values[row]
                        try:
                            images[camera] = read_rgb(value, resize)
                        except (OSError, ValueError) as error:
                            raise ValueError(f'{episode.source}: {camera}, row {row}: invalid/missing image: {error}') from error
                        shape = images[camera].shape
                        if camera in shapes and shapes[camera] != shape:
                            raise ValueError(f'{episode.source}: {camera}, row {row}: image shape changed {shapes[camera]} -> {shape}')
                        shapes[camera] = shape
                with timings.measure('write_encode'):
                    if writer is None:
                        features = {f'observation.images.{c}': dict(dtype='video', shape=s, names=['height','width','channel']) for c, s in shapes.items()}
                        features.update({key: dict(dtype='float32', shape=(spec.dim,), names=list(spec.value_names)) for key in ('observation.state','action')})
                        if spec.name == 'bimanual':
                            # Existing trainer requires this one scalar compatibility column.
                            features['quality.frame_valid'] = dict(dtype='bool', shape=(1,), names=['frame_valid'])
                        writer = api.LeRobotDataset.create(repo_id=repo_id, root=root, fps=episode.fps,
                            robot_type=spec.robot_type, features=features, use_videos=True, video_backend='pyav')
                        writer.image_writer = _PNGWriter()
                    frame = {f'observation.images.{c}': image for c, image in images.items()}
                    frame.update({'observation.state': episode.states[row], 'action': episode.actions[row], 'task': episode.task})
                    if spec.name == 'bimanual':
                        frame['quality.frame_valid'] = np.asarray([True], dtype=bool)
                    writer.add_frame(frame)
            with timings.measure('write_encode'):
                writer.save_episode()
            summaries.append(dict(source=episode.source, task=episode.task, frames=len(episode.states), shapes=dict(shapes), fps=episode.fps))
        if writer is None:
            raise ValueError('No convertible episodes')
    finally:
        if writer is not None:
            writer.stop_image_writer()
    return summaries
