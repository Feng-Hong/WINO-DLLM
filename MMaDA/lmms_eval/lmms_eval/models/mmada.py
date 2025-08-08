import os
import uuid
import warnings
from typing import List, Optional, Tuple, Union

import torch
from accelerate import Accelerator, DistributedType
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model

warnings.simplefilter("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

from loguru import logger as eval_logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from .model_mmada import MAGVITv2, MMadaModelLM
from .model_mmada.prompting_utils import UniversalPrompting
from .model_mmada.utils import get_config, flatten_omega_conf, image_transform

from PIL import Image
import re

@register_model("mmada")
class MMaDA(lmms):
    """
    MMaDA Model
    https://github.com/Gen-Verse/MMaDA
    """

    def __init__(
        self,
        pretrained: str = "Gen-Verse/MMaDA-8B-MixCoT",
        gen_method: str = 'default',
        gen_length: int = 256,
        diff_step: int = 128,
        block_length: int = 128,
        threshold: float = 0.8,
        threshold_back: float = 0.9,
        reasoning: bool = False,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1, #currently only support batch_size=1
        trust_remote_code: Optional[bool] = True,
        use_cache=True,
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device
        self._model = MMadaModelLM.from_pretrained(pretrained, device_map=self.device_map, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code, padding_side="left")
        self._config = self._model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = use_cache
        self._max_length = kwargs.get("max_length", 2048)

        self.uni_prompting = UniversalPrompting(self._tokenizer, max_text_len=512,
                                           special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>", "<|mmu|>", "<|t2v|>",
                                           "<|v2v|>", "<|lvg|>"),
                                           ignore_id=-100, cond_dropout_prob=0.1,
                                           use_reserved_token=True)

        self.vq_model = MAGVITv2.from_pretrained("showlab/magvitv2").to(self._device)
        self.vq_model.requires_grad_(False)
        self.vq_model.eval()

        self.mask_token_id = self._config.mask_token_id
        self.gen_method = gen_method
        self.gen_length = gen_length
        self.diff_step = diff_step
        self.block_length = block_length
        self.threshold = threshold
        self.threshold_back = threshold_back
        self.reasoning = reasoning
        self.reasoning_prompt = "You should first think about the reasoning process in the mind and then provide the user with the answer. The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here\n"

        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for MMaDA")

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list

    def generate_until(self, requests: List[Instance]) -> List[str]:
        res = []

        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tokenizer.encode(x[0])
            return -len(toks), x[0]

        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            gen_kwargs = all_gen_kwargs[0]
            task = task[0]
            split = split[0]
            visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]
            visuals = self.flatten(visuals)
            visual_paths = []
            # save images to ./tmp, name generated by hash function
            for visual in visuals:
                name = uuid.uuid4().hex.upper()[0:6]
                tmp_save_path = f"./tmp/{name}.png"
                os.makedirs('./tmp',exist_ok=True)
                visual.save(tmp_save_path)
                visual_paths.append(tmp_save_path)

            if isinstance(contexts, tuple):
                contexts = list(contexts)

            text_batch = []
            for i in range(len(contexts)):
                if "<image>" in contexts[i]:
                    contexts[i] = contexts[i].replace("<image>", "")
                image_placeholders = re.findall(r"<image \d+>", contexts[i])
                for ph in image_placeholders:
                   contexts[i] = contexts[i].replace(ph, "")
                image_placeholders = re.findall(r"<image\d+>", contexts[i])
                for ph in image_placeholders:
                    contexts[i] = contexts[i].replace(ph, "")

                if self.reasoning:
                    contexts[i] = self.reasoning_prompt + contexts[i]

                text_batch.append('<|start_header_id|>user<|end_header_id|>\n' + contexts[i] + '<eot_id><|start_header_id|>assistant<|end_header_id|>\n')

            input_ids = self.uni_prompting.text_tokenizer(text_batch, padding=True, padding_side="left")['input_ids']
            input_ids = torch.tensor(input_ids).to(self.device)

            input_ids_list = []
            input_ids_list.append((torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|mmu|>']).to(self.device))
            for visual_path in visual_paths:
                image_ori = Image.open(visual_path).convert("RGB")
                image = image_transform(image_ori, resolution=512)
                image = image.unsqueeze(0).to(self.device)
                image_tokens = self.vq_model.get_code(image) + len(self.uni_prompting.text_tokenizer)
                input_ids_list.append((torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|soi|>']).to(self.device))
                input_ids_list.append(image_tokens)
                input_ids_list.append((torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|eoi|>']).to(self.device))
            input_ids_list.append((torch.ones(input_ids.shape[0], 1) * self.uni_prompting.sptids_dict['<|sot|>']).to(self.device))
            input_ids_list.append(input_ids)
            input_ids = torch.cat(input_ids_list, dim=1).long()

            if self.gen_method == 'default':
                output_ids = self.model.mmu_generate(input_ids, max_new_tokens=self.gen_length, steps=self.diff_step, block_length=self.block_length)
                answers = self.uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)
            elif self.gen_method == 'wino':
                output_ids, inference_step = self.model.mmu_generate_wino(
                    input_ids, gen_length=self.gen_length, block_length=self.block_length, threshold=self.threshold, threshold_back=self.threshold_back)
                answers = self.uni_prompting.text_tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)

            for ans, context in zip(answers, contexts):
                res.append(ans)
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)
                pbar.update(1)

            # remove visuals from tmp
            for visual_path in visual_paths:
                try:
                    os.remove(visual_path)
                except:
                    pass
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)

        pbar.close()
        return res

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation")
