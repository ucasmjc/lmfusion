# Adopted from: https://github.com/haotian-liu/LLaVA/blob/main/llava/train/llava_trainer.py
import os
import logging
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Sampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
    TRAINER_STATE_NAME,
)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_encoder', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "is_alignment", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        # if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
        if torch.distributed.get_rank() == 0:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)
from PIL import Image, ImageDraw, ImageFont
import warnings
from transformers import TrainerCallback
class DatasetStateCallback(TrainerCallback):
    def on_save(self, args, state, control,**kwargs):
        """在保存检查点时触发"""
        try:
            # 获取当前检查点路径
            checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            
            # 确保检查点目录存在
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # 获取并保存数据集状态
            dataset = kwargs.get("train_dataset")
            if dataset and hasattr(dataset, "state_dict"):
                state_dict = dataset.state_dict()
                torch.save(
                    state_dict,
                    os.path.join(checkpoint_dir, "dataset_state.pt")
                )
                #logger.info(f"Saved dataset state at {checkpoint_dir}")
        except Exception as e:
            logger.error(f"Error saving dataset state: {str(e)}")
        return control

    def on_load_checkpoint(self, args, state, control,**kwargs):
        """在加载检查点时触发"""
        try:
            # 获取检查点路径
            checkpoint_dir = state.checkpoint_dir
            
            # 加载数据集状态
            dataset = kwargs.get("train_dataset")
            state_path = os.path.join(checkpoint_dir, "dataset_state.pt")
            
            if dataset and hasattr(dataset, "load_state_dict") and os.path.exists(state_path):
                state_dict = torch.load(state_path)
                dataset.load_state_dict(state_dict)
                logger.info(f"Loaded dataset state from {checkpoint_dir}")
        except Exception as e:
            logger.error(f"Error loading dataset state: {str(e)}")
        return control
def save_combined_image(
    recon_image: torch.Tensor,  # [3, H, W], range [-1, 1]
    gt: torch.Tensor,          # [3, H, W], range [-1, 1]
    noise_image: torch.Tensor, # [3, H, W], range [-1, 1]
    save_path: str = "combined_result.png",
    labels: tuple = ("Recon", "GT", "Noise"),
    gap: int = 10,             # 图片间距
    text_height: int = 30,     # 文字区域高度
    font_size: int = 20,
    font_color: str = "black",
    bg_color: str = "white"
) -> Image.Image:

    def _normalize_to_uint8(tensor: torch.Tensor) -> torch.Tensor:
        tensor = (tensor + 1) * 127.5  # [-1,1] -> [0,255]
        tensor = tensor.clamp(0, 255).to(torch.uint8)
        return tensor

    recon_uint8 = _normalize_to_uint8(recon_image)
    gt_uint8 = _normalize_to_uint8(gt)
    noise_uint8 = _normalize_to_uint8(noise_image)

    def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        return Image.fromarray(tensor.permute(1, 2, 0).cpu().numpy())

    recon_pil = _tensor_to_pil(recon_uint8)
    gt_pil = _tensor_to_pil(gt_uint8)
    noise_pil = _tensor_to_pil(noise_uint8)

    single_width, single_height = recon_pil.width, recon_pil.height
    total_width = single_width * 3 + gap * 2
    total_height = single_height + text_height

    combined_image = Image.new("RGB", (total_width, total_height), color=bg_color)
    draw = ImageDraw.Draw(combined_image)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    def _paste_with_label(img: Image.Image, pos_x: int, label: str):
        # 粘贴图片
        combined_image.paste(img, (pos_x, text_height))
        # 计算文字位置（居中）
        text_x = pos_x + single_width // 2
        text_y = text_height // 2
        draw.text((text_x, text_y), label, fill=font_color, font=font, anchor="mm")

    _paste_with_label(recon_pil, 0, labels[0])
    _paste_with_label(gt_pil, single_width + gap, labels[1])
    _paste_with_label(noise_pil, single_width * 2 + gap * 2, labels[2])

    combined_image.save(save_path)
    #print(f"图片已保存至: {save_path}")
    return combined_image
import wandb
import random
import torch.distributed as dist

from train_files.pipeline import sample
from diffusers import FlowMatchEulerDiscreteScheduler     


class JanusTrainer(Trainer):
    def __init__(self, *args,**kwargs):
        super().__init__(*args, **kwargs)
        self._reset_accumulator()

    def _reset_accumulator(self):
        self.accumulator = {
            "image_under": {"ar_loss": 0.0, "gradnorm": 0.0, "count": 0},
            "image_gen": {"gen_loss": 0.0, "gradnorm": 0.0, "count": 0}
        }
    def update_loss_dict(self, loss, gradnorm, task):
        if self.is_local_process_zero():
            if task == "image_under":
                self.accumulator["image_under"]["ar_loss"] += loss
                self.accumulator["image_under"]["gradnorm"] += gradnorm
                self.accumulator["image_under"]["count"] += 1
                if self.accumulator["image_under"]["count"]%10==0:
                    wandb.log({"ar_loss":self.accumulator["image_under"]["ar_loss"]/10,"ar_norm":self.accumulator["image_under"]["gradnorm"]/10})
                    self.accumulator["image_under"]={"ar_loss": 0.0, "gradnorm": 0.0, "count": 0}
            elif task == "image_gen":
                self.accumulator["image_gen"]["gen_loss"] += loss
                self.accumulator["image_gen"]["gradnorm"] += gradnorm
                self.accumulator["image_gen"]["count"] += 1
                if self.accumulator["image_gen"]["count"]%10==0:
                    wandb.log({"gen_loss":self.accumulator["image_gen"]["gen_loss"]/10,"gen_norm":self.accumulator["image_gen"]["gradnorm"]/10})
                    self.accumulator["image_gen"]={"gen_loss": 0.0, "gradnorm": 0.0, "count": 0}
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        #prob=random.random()
        #prob=torch.tensor(prob).cuda()
        #dist.broadcast(prob, src=0)
        #if prob<0.1:
          #  inputs["input_ids"]=None
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes
        
        if self.args.process_index == 0 and self.state.global_step % 200==1 and inputs["task"]=="image_gen":
            print(self.is_local_process_zero())
            latent=outputs["latent"]#b*16*256*256
            gt=outputs["gt"][0] #b*3*256*256
            noise=outputs["noise"]#b*16*256*256
            recon_image=model.latent_to_image(latent[0].unsqueeze(0))[0]
            noise_image=model.latent_to_image(noise[0].unsqueeze(0))[0]
            image=save_combined_image(
                recon_image, 
                gt, 
                noise_image,
                save_path=f"/work/share/projects/mjc/lmfusion/visual_img/train_res_{self.state.global_step}.png",
                labels=("Prediction", "Ground Truth", "Noisy Input"),
            )
            
            wandb.log({
                "val_image": wandb.Image(image,caption=f"Validation at step {self.state.global_step}")
            })
        return (loss, outputs) if return_outputs else loss

    # def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
    #     eval_dataset=self.eval_dataset
    #     self.model.eval()
    #     total_loss = 0
    #     all_predictions = []
    #     all_labels = []
    #     scheduler=FlowMatchEulerDiscreteScheduler(**{"shift": 3.0})
    #     num_inference_steps= 50
    #     scheduler.set_timesteps(num_inference_steps)
    #     images=[]
    #     for inputs in eval_dataset:
    #         with torch.no_grad():
    #             outputs = sample(self.model,scheduler,inputs["input_ids"].to(device=self.model.device))
    #             print(outputs.shape)
    #             image=outputs.squeeze(0).permute(1, 2, 0).cpu().numpy()
    #             print(image.shape)
    #             images.append(wandb.Image(Image.fromarray(image), caption=inputs["sft_format"]))
    #     wandb.log({"generated_images": images,"global_step": self.state.global_step})
        
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None
        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            optimized_parameters = [(n, p) for n, p in opt_model.named_parameters() if p.requires_grad]
            optimizer_grouped_parameters = []

            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            if self.args.diffusion_lr is not None:
                diffusion_parameters = [
                    name for name, _ in optimized_parameters if "diffusion" in name
                ]
                decay_diffusion_parameters = [name for name in diffusion_parameters if name in decay_parameters]
                nodecay_diffusion_parameters = [name for name in diffusion_parameters if name not in decay_parameters]
                optimizer_grouped_parameters.extend([
                    {
                        "params": [p for n, p in optimized_parameters if n in decay_diffusion_parameters],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.diffusion_lr,
                    },
                    {
                        "params": [p for n, p in optimized_parameters if n in nodecay_diffusion_parameters],
                        "weight_decay": 0.0,
                        "lr": self.args.diffusion_lr,
                    }
                ])
                # print(lm_parameters, '!!!!!!')
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer