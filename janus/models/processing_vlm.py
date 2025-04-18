# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from dataclasses import dataclass
from typing import Dict, List
import copy
import torch
from PIL.Image import Image
from transformers import LlamaTokenizerFast
from transformers.processing_utils import ProcessorMixin

from janus.models.image_processing_vlm import VLMImageProcessor
from janus.utils.conversation import get_conv_template


class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):
        return self.__dict__[item]

    def __setitem__(self, key, value):
        self.__dict__[key] = value


@dataclass
class VLChatProcessorOutput(DictOutput):
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor
    labels: torch.Tensor

    def __len__(self):
        return len(self.input_ids)


@dataclass
class BatchedVLChatProcessorOutput(DictOutput):
    sft_format: List[str]
    input_ids: torch.Tensor
    labels: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_emb_mask: torch.BoolTensor

    def to(self, device, dtype=torch.bfloat16):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.images_emb_mask = self.images_emb_mask.to(device)
        self.pixel_values = self.pixel_values.to(device=device, dtype=dtype)
        return self


class VLChatProcessor(ProcessorMixin):
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast")

    attributes = ["image_processor", "tokenizer"]

    system_prompt = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )

    def __init__(
        self,
        image_processor: VLMImageProcessor,
        tokenizer: LlamaTokenizerFast,
        image_tag: str = "<image_placeholder>",
        image_start_tag: str = "<begin_of_image>",
        image_end_tag: str = "<end_of_image>",
        pad_tag: str = "<｜▁pad▁｜>",
        num_image_tokens: int = 576,
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,

        tokenizer_model_max_length: int = 2048, 
        use_tokenizer_pad: bool = False,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

        image_id = self.tokenizer.vocab.get(image_tag)
        if image_id is None:
            special_tokens = [image_tag]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Add image tag = {image_tag} to the tokenizer")

        self.image_tag = image_tag
        self.image_start_tag = image_start_tag
        self.image_end_tag = image_end_tag
        self.pad_tag = pad_tag

        self.num_image_tokens = num_image_tokens
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id


        self.tokenizer_model_max_length = tokenizer_model_max_length
        self.use_tokenizer_pad = use_tokenizer_pad

        self.generation_prompt_length = len(self.tokenizer.encode("<|Assistant|>:", return_tensors="pt"))

        super().__init__(
            image_processor,
            tokenizer,
            image_tag,
            num_image_tokens,
            add_special_token,
            sft_format,
            mask_prompt,
            ignore_id,
            **kwargs,
        )

    def new_chat_template(self):
        conv = get_conv_template(self.sft_format)
        conv.set_system_message(self.system_prompt)
        return conv

    def apply_sft_template_for_multi_turn_prompts(
        self,
        conversations: List[Dict[str, str]],
        sft_format: str = "deepseek",
        system_prompt: str = "",
    ):
        """
        Applies the SFT template to conversation.

        An example of conversation:
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder> is Figure 1.\n<image_placeholder> is Figure 2.\nWhich image is brighter?",
                "images": [
                    "./multi-images/attribute_comparison_1.png",
                    "./multi-images/attribute_comparison_2.png"
                ]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        Args:
            conversations (List[Dict]): A conversation with a List of Dict[str, str] text.
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """

        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        for message in conversations:
            conv.append_message(message["role"], message["content"].strip())
        sft_prompt = conv.get_prompt().strip()

        return sft_prompt

    @property
    def image_token(self):
        return self.image_tag

    @property
    def image_id(self):
        image_id = self.tokenizer.vocab.get(self.image_tag)
        return image_id

    @property
    def image_start_id(self):
        image_start_id = self.tokenizer.vocab.get(self.image_start_tag)
        return image_start_id

    @property
    def image_end_id(self):
        image_end_id = self.tokenizer.vocab.get(self.image_end_tag)
        return image_end_id

    @property
    def image_start_token(self):
        return self.image_start_tag

    @property
    def image_end_token(self):
        return self.image_end_tag

    @property
    def pad_id(self):
        pad_id = self.tokenizer.vocab.get(self.pad_tag)
        # pad_id = self.tokenizer.pad_token_id
        # if pad_id is None:
        #     pad_id = self.tokenizer.eos_token_id

        return pad_id

    def add_image_token(
        self,
        image_indices: List[int],
        input_ids: torch.LongTensor,
        targets: None,
    ):
        """

        Args:
            image_indices (List[int]): [index_0, index_1, ..., index_j]
            input_ids (torch.LongTensor): [N]

        Returns:
            input_ids (torch.LongTensor): [N + image tokens]
            num_image_tokens (torch.IntTensor): [n_images]
        """

        input_slices = []
        targets_slices = []

        start = 0
        for index in image_indices:
            if self.add_special_token:
                end = index + 1
            else:
                end = index

            # original text tokens
            input_slices.append(input_ids[start:end])

            # add boi, image tokens, eoi and set the mask as False
            input_slices.append(self.image_start_id * torch.ones((1), dtype=torch.long))
            input_slices.append(
                self.image_id * torch.ones((self.num_image_tokens,), dtype=torch.long)
            )
            input_slices.append(self.image_end_id * torch.ones((1), dtype=torch.long))

            if targets is not None:
                targets_slices.append(targets[start:end])
                targets_slices.append(self.ignore_id * torch.ones((1), dtype=torch.long))
                targets_slices.append(
                    self.ignore_id * torch.ones((self.num_image_tokens,), dtype=torch.long)
                )
                targets_slices.append(self.ignore_id * torch.ones((1), dtype=torch.long))

            start = index + 1

        # the left part
        input_slices.append(input_ids[start:])
        
        # concat all slices
        input_ids = torch.cat(input_slices, dim=0)
        num_image_tokens = torch.IntTensor([self.num_image_tokens] * len(image_indices))

        if targets is not None:
            targets_slices.append(targets[start:])
            targets = torch.cat(targets_slices, dim=0)

        return input_ids, num_image_tokens, targets

    def process_one(
        self,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image] = None,
        is_training: bool = False,
        modal = None,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert (
            prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        if modal == "image_gen":
            sft_format = self.apply_sft_template_for_multi_turn_prompts(
                conversations=conversations,
                sft_format=self.sft_format,
                system_prompt="",
                )
        elif prompt is None:
            # apply sft format
            sft_format = self.apply_sft_template_for_multi_turn_prompts(
                conversations=conversations,
                sft_format=self.sft_format,
                system_prompt=self.system_prompt,
            )
        else:
            sft_format = prompt
        # print(self.use_tokenizer_pad,"16546546")

        # =================
        # tokenize  pad for image generation prompt
        if self.use_tokenizer_pad:
            input_ids = self.tokenizer.encode(sft_format,
                                            padding="max_length",  # 填充到最大长度
                                                max_length=self.tokenizer_model_max_length,         # 设置最大长度
                                                truncation=True,       # 截断超过最大长度的部分
                                                # return_tensors="pt"    # 返回 PyTorch 张量, 不选择就返回一个list
                                            )
        else:
            input_ids = self.tokenizer.encode(sft_format,
                                            # padding="max_length",  # 填充到最大长度
                                             max_length=self.tokenizer_model_max_length,         # 设置最大长度
                                             truncation=True,       # 截断超过最大长度的部分
                                                # return_tensors="pt"    # 返回 PyTorch 张量, 不选择就返回一个list
                                            )

        
        # Truncate sequences to max length as image embeddings can make the sequence longer, for image generation
        if self.tokenizer_model_max_length is not None:
            # import ipdb; ipdb.set_trace()
            # print(f'inputs ids length:{len(input_ids)} ==== {self.tokenizer_model_max_length}!')
            if len(input_ids) > self.tokenizer_model_max_length:
                print(f'inputs ids length:{len(input_ids)} is longer than {self.tokenizer_model_max_length}!')
            input_ids =  input_ids[:self.tokenizer_model_max_length]
        # =================


        input_ids = torch.LongTensor(input_ids)

        if is_training and modal == "image_gen":
            targets = torch.full_like(input_ids, self.ignore_id)
            targets[-1] = input_ids[-1]
            num_image_tokens = [576]


        elif is_training:
            training_input_ids_list = []
            targets_list = []
            sample_types_list = []

            for message_idx, message in enumerate(conversations):
                if message_idx == 0:
                    prompt = self.apply_sft_template_for_multi_turn_prompts(
                                conversations=[message],
                                sft_format=self.sft_format,
                                system_prompt=self.system_prompt,
                                )
                else:
                    prompt = message["role"] + ": " + message["content"]

                if message["role"] == "<|Assistant|>":
                    prompt += "<｜end▁of▁sentence｜>"
                else:
                    prompt += "\n\n"
            
                if message_idx != 0:
                    training_input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0][1:]
                else:
                    training_input_ids = self.tokenizer.encode(prompt, return_tensors="pt")[0]

                training_input_ids_list.append(training_input_ids)

                targets = torch.full_like(training_input_ids, self.ignore_id)
                sample_types = torch.full_like(training_input_ids, self.ignore_id)
                if message["role"] == "<|Assistant|>":
                    targets[self.generation_prompt_length+1:] = training_input_ids[self.generation_prompt_length+1:].clone()
                targets_list.append(targets)
                sample_types_list.append(sample_types)

            targets = torch.cat(targets_list)
            sample_types = torch.cat(sample_types_list)
            training_input_ids = torch.cat(training_input_ids_list)


            # truncation
            import torch.nn.functional as F
            def pad_to_max_length(tensor, max_length, pad_value=-100):
                """
                将张量填充到指定长度，前部填充 pad_value。
                """

                
                current_length = tensor.size(0)  # 获取当前长度
                if current_length < max_length:
                    # 计算需要填充的长度
                    pad_length = max_length - current_length
                    # 在前部填充 pad_value
                    padded_tensor = F.pad(tensor, (pad_length, 0), value=pad_value)
                else:
                    # 如果长度已经足够，直接截断
                    
                    padded_tensor = tensor[:max_length]
                return padded_tensor

            # 对 targets, sample_types, training_input_ids 进行填充
            max_length = self.tokenizer_model_max_length


            if self.use_tokenizer_pad:
                targets = pad_to_max_length(targets, max_length, pad_value=self.ignore_id)
                sample_types = pad_to_max_length(sample_types, max_length, pad_value=self.ignore_id)
                training_input_ids = pad_to_max_length(training_input_ids, max_length, pad_value=self.ignore_id)
            else:
                targets =  targets[:self.tokenizer_model_max_length]
                sample_types = sample_types[:self.tokenizer_model_max_length]
                training_input_ids = training_input_ids[:self.tokenizer_model_max_length]

            
            

            
            """if self.tokenizer_model_max_length is not None:
                # import ipdb; ipdb.set_trace()
                print(f'inputs ids length:{len(targets)} ==== {self.tokenizer_model_max_length}!')
                if len(targets) > self.tokenizer_model_max_length:
                    print(f'inputs ids length:{len(targets)} is longer than {self.tokenizer_model_max_length}!')
                targets =  targets[:self.tokenizer_model_max_length]
                sample_types = sample_types[:self.tokenizer_model_max_length]
                training_input_ids = training_input_ids[:self.tokenizer_model_max_length]"""



            types, counts = torch.unique(sample_types[sample_types > -1], return_counts=True)

            if len(types) > 0:
                target_num_samples = counts.amin()

                for type_id, type_count in zip(types, counts):
                    if type_count > target_num_samples:
                        indices = torch.nonzero(sample_types == type_id)[:, 0]
                        random_selector = torch.randperm(indices.size(0))[:-target_num_samples]
                        targets[indices[random_selector]] = self.ignore_id
                        sample_types[indices[random_selector]] = -1
 


            input_ids = training_input_ids

            # add image tokens to the input_ids
            image_token_mask: torch.BoolTensor = input_ids == self.image_id
            image_indices = image_token_mask.nonzero()
            input_ids, num_image_tokens, targets = self.add_image_token(
                image_indices=image_indices,
                input_ids=input_ids,
                targets=targets,
            )

        else:
            targets = None

            # add image tokens to the input_ids
            image_token_mask: torch.BoolTensor = input_ids == self.image_id
            image_indices = image_token_mask.nonzero()
            input_ids, num_image_tokens, targets = self.add_image_token(
                image_indices=image_indices,
                input_ids=input_ids,
                targets=targets,
            )

        # load images
        images_outputs = self.image_processor(images, return_tensors="pt")

        prepare = VLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=input_ids,
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=num_image_tokens,
            labels=targets,
        )

        return prepare

    def __call__(
        self,
        *,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image] = None,
        force_batchify: bool = True,
        is_training: bool = False,
        modal = None,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            force_batchify (bool): force batchify the inputs;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        prepare = self.process_one(
            prompt=prompt, conversations=conversations, images=images, is_training=is_training, modal=modal
        )

        if force_batchify:
            prepare = self.batchify([prepare])

        return prepare

    def batchify(
        self, prepare_list: List[VLChatProcessorOutput]
    ) -> BatchedVLChatProcessorOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            prepare_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.

        Returns:
            BatchedVLChatProcessorOutput: A dictionary of the inputs to use for multimodal inference.
        """

        batch_size = len(prepare_list)
        sft_format = []
        n_images = []
        seq_lens = []
        for prepare in prepare_list:
            n_images.append(len(prepare.num_image_tokens))
            seq_lens.append(len(prepare))

        input_token_max_len = max(seq_lens)
        max_n_images = max(1, max(n_images))
        #max_n_images = max(n_images)

        batched_input_ids = torch.full(
            (batch_size, input_token_max_len), self.pad_id
        ).long()  # FIXME
        batched_labels_ids = torch.full(
            (batch_size, input_token_max_len), self.pad_id
        ).long()  # FIXME

        batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
        batched_pixel_values = torch.zeros(
            (batch_size, max_n_images, *self.image_processor.default_shape)
        ).float()
        batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
        batched_images_emb_mask = torch.zeros(
            (batch_size, max_n_images, self.num_image_tokens)
        ).bool()

        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids
            seq_len = len(prepare)
            n_image = len(prepare.num_image_tokens)
            # left-padding
            batched_attention_mask[i, -seq_len:] = 1
            batched_input_ids[i, -seq_len:] = torch.LongTensor(input_ids)
            batched_images_seq_mask[i, -seq_len:] = input_ids == self.image_id

            if n_image > 0:
                #print(batched_pixel_values.shape,prepare.pixel_values.shape,n_image)
                batched_pixel_values[i, :n_image] = prepare.pixel_values
                for j, n_image_tokens in enumerate(prepare.num_image_tokens):
                    batched_images_emb_mask[i, j, :n_image_tokens] = True

            sft_format.append(prepare.sft_format)

            if prepare.labels is not None:
                # import ipdb; ipdb.set_trace()
                labels = prepare.labels
                batched_labels_ids[i, -seq_len:] = torch.LongTensor(labels)

        # No need to calculate the loss of pad tokens
        batched_labels_ids = torch.where(batched_labels_ids == self.pad_id, -100, batched_labels_ids) # 100015 

        
        batched_prepares = BatchedVLChatProcessorOutput(
            input_ids=batched_input_ids,
            labels=batched_labels_ids,
            attention_mask=batched_attention_mask,
            pixel_values=batched_pixel_values,
            images_seq_mask=batched_images_seq_mask,
            images_emb_mask=batched_images_emb_mask,
            sft_format=sft_format,
        )

        return batched_prepares
