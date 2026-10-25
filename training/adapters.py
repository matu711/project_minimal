"""Dual-PiPER adapter for canonical front/left_wrist/right_wrist exports."""

from typing import Literal

from openpi import transforms
from openpi.training import config


def bimanual_data_config(
    repo_id: str,
    *,
    wrist1_side: Literal["left", "right"] = "left",
    exclude_invalid_images: bool = True,
    use_delta_joint_actions: bool = False,
) -> config.LeRobotAlohaDataConfig:
    """Canonical cameras and optional observation-relative joint deltas.

    The dual-arm trainer explicitly enables use_delta_joint_actions. The default
    preserves existing adapter callers. DeltaActions runs before normalization;
    AbsoluteActions restores joint targets after inverse normalization. Both
    grippers stay absolute. Dataset values and State are never rewritten.
    """
    if wrist1_side != "left":
        raise ValueError("Canonical dual-PiPER export has wrist1=left; use left_wrist/right_wrist keys")
    if not isinstance(exclude_invalid_images, bool):
        raise ValueError("exclude_invalid_images must be a boolean")
    if not isinstance(use_delta_joint_actions, bool):
        raise ValueError("use_delta_joint_actions must be a boolean")
    validity_key = "quality.frame_valid" if exclude_invalid_images else None
    repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "images": {
                        "cam_high": "observation.images.front",
                        "cam_left_wrist": "observation.images.left_wrist",
                        "cam_right_wrist": "observation.images.right_wrist",
                    },
                    "state": "observation.state",
                    "actions": "action",
                    "prompt": "prompt",
                }
            )
        ]
    )
    return config.LeRobotAlohaDataConfig(
        repo_id=repo_id,
        base_config=config.DataConfig(prompt_from_task=True, sample_validity_key=validity_key),
        adapt_to_pi=False,
        use_delta_joint_actions=use_delta_joint_actions,
        repack_transforms=repack,
    )


import dataclasses

def single_data_config(repo_id):
    from openpi import transforms as t
    from openpi.policies import libero_policy
    from openpi.training import config

    @dataclasses.dataclass(frozen=True)
    class SingleData(config.DataConfigFactory):
        def create(self, assets_dirs, model_config):
            mask = t.make_bool_mask(6, -1)
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                repack_transforms=t.Group(inputs=[t.RepackTransform({
                    'observation/image': 'observation.images.front',
                    'observation/wrist_image': 'observation.images.wrist',
                    'observation/state': 'observation.state', 'actions': 'action', 'prompt': 'prompt',
                })]),
                data_transforms=t.Group(
                    inputs=[libero_policy.LiberoInputs(model_config.model_type), t.DeltaActions(mask)],
                    outputs=[t.AbsoluteActions(mask), libero_policy.LiberoOutputs()],
                ),
                model_transforms=config.ModelTransformFactory()(model_config),
                action_sequence_keys=('action',),
            )

    return SingleData(repo_id=repo_id, base_config=config.DataConfig(prompt_from_task=True))

