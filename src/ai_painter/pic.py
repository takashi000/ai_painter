from diffusers import StableDiffusionPipeline, StableDiffusionControlNetInpaintPipeline, DPMSolverMultistepScheduler, UniPCMultistepScheduler, DDIMScheduler, StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from diffusers.models import AutoencoderKL
from diffusers.callbacks import PipelineCallback, SDCFGCutoffCallback
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sd15
from transformers import pipeline, AutoImageProcessor, UperNetForSemanticSegmentation
from mmpose.apis import MMPoseInferencer
from typing import Any, Dict
import cv2
from PIL import Image
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from accelerate.utils import compile_regions
import gc

import traceback
import re
import sys

torch.set_float32_matmul_precision('high')

ada_palette = np.asarray([
      [0, 0, 0],
      [120, 120, 120],
      [180, 120, 120],
      [6, 230, 230],
      [80, 50, 50],
      [4, 200, 3],
      [120, 120, 80],
      [140, 140, 140],
      [204, 5, 255],
      [230, 230, 230],
      [4, 250, 7],
      [224, 5, 255],
      [235, 255, 7],
      [150, 5, 61],
      [120, 120, 70],
      [8, 255, 51],
      [255, 6, 82],
      [143, 255, 140],
      [204, 255, 4],
      [255, 51, 7],
      [204, 70, 3],
      [0, 102, 200],
      [61, 230, 250],
      [255, 6, 51],
      [11, 102, 255],
      [255, 7, 71],
      [255, 9, 224],
      [9, 7, 230],
      [220, 220, 220],
      [255, 9, 92],
      [112, 9, 255],
      [8, 255, 214],
      [7, 255, 224],
      [255, 184, 6],
      [10, 255, 71],
      [255, 41, 10],
      [7, 255, 255],
      [224, 255, 8],
      [102, 8, 255],
      [255, 61, 6],
      [255, 194, 7],
      [255, 122, 8],
      [0, 255, 20],
      [255, 8, 41],
      [255, 5, 153],
      [6, 51, 255],
      [235, 12, 255],
      [160, 150, 20],
      [0, 163, 255],
      [140, 140, 140],
      [250, 10, 15],
      [20, 255, 0],
      [31, 255, 0],
      [255, 31, 0],
      [255, 224, 0],
      [153, 255, 0],
      [0, 0, 255],
      [255, 71, 0],
      [0, 235, 255],
      [0, 173, 255],
      [31, 0, 255],
      [11, 200, 200],
      [255, 82, 0],
      [0, 255, 245],
      [0, 61, 255],
      [0, 255, 112],
      [0, 255, 133],
      [255, 0, 0],
      [255, 163, 0],
      [255, 102, 0],
      [194, 255, 0],
      [0, 143, 255],
      [51, 255, 0],
      [0, 82, 255],
      [0, 255, 41],
      [0, 255, 173],
      [10, 0, 255],
      [173, 255, 0],
      [0, 255, 153],
      [255, 92, 0],
      [255, 0, 255],
      [255, 0, 245],
      [255, 0, 102],
      [255, 173, 0],
      [255, 0, 20],
      [255, 184, 184],
      [0, 31, 255],
      [0, 255, 61],
      [0, 71, 255],
      [255, 0, 204],
      [0, 255, 194],
      [0, 255, 82],
      [0, 10, 255],
      [0, 112, 255],
      [51, 0, 255],
      [0, 194, 255],
      [0, 122, 255],
      [0, 255, 163],
      [255, 153, 0],
      [0, 255, 10],
      [255, 112, 0],
      [143, 255, 0],
      [82, 0, 255],
      [163, 255, 0],
      [255, 235, 0],
      [8, 184, 170],
      [133, 0, 255],
      [0, 255, 92],
      [184, 0, 255],
      [255, 0, 31],
      [0, 184, 255],
      [0, 214, 255],
      [255, 0, 112],
      [92, 255, 0],
      [0, 224, 255],
      [112, 224, 255],
      [70, 184, 160],
      [163, 0, 255],
      [153, 0, 255],
      [71, 255, 0],
      [255, 0, 163],
      [255, 204, 0],
      [255, 0, 143],
      [0, 255, 235],
      [133, 255, 0],
      [255, 0, 235],
      [245, 0, 255],
      [255, 0, 122],
      [255, 245, 0],
      [10, 190, 212],
      [214, 255, 0],
      [0, 204, 255],
      [20, 0, 255],
      [255, 255, 0],
      [0, 153, 255],
      [0, 41, 255],
      [0, 255, 204],
      [41, 0, 255],
      [41, 255, 0],
      [173, 0, 255],
      [0, 245, 255],
      [71, 0, 255],
      [122, 0, 255],
      [0, 255, 184],
      [0, 92, 255],
      [184, 255, 0],
      [0, 133, 255],
      [255, 214, 0],
      [25, 194, 194],
      [102, 255, 0],
      [92, 0, 255],
  ])

class SDCFGCutoffCallbackTrace(PipelineCallback):
    def __init__(self, cutoff_step_ratio: float = 1, cutoff_step_index: Any | None = None):
        super().__init__(cutoff_step_ratio=cutoff_step_ratio, cutoff_step_index=cutoff_step_index)
        self.images = None
        self.generator = None
        self.callback_func = None
    """
    Callback function for Stable Diffusion Pipelines. After certain number of steps (set by `cutoff_step_ratio` or
    `cutoff_step_index`), this callback will disable the CFG.

    Note: This callback mutates the pipeline by changing the `_guidance_scale` attribute to 0.0 after the cutoff step.
    """
    tensor_inputs = ["prompt_embeds", "latents"]

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        latents = callback_kwargs[self.tensor_inputs[1]]
        
        image, latent = self.decode_latents(latents=latents, pipeline=pipeline)

        if self.images:
            del self.images
            self.images = None

        self.images = (image, latent)
        
        if self.callback_func:
            self.callback_func((image, latent))

        return self.callback_cutoff(pipeline, step_index, timestep, callback_kwargs)
    
    def callback_cutoff(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        cutoff_step_ratio = self.config.cutoff_step_ratio
        cutoff_step_index = self.config.cutoff_step_index

        # Use cutoff_step_index if it's not None, otherwise use cutoff_step_ratio
        cutoff_step = (
            cutoff_step_index if cutoff_step_index is not None else int(pipeline.num_timesteps * cutoff_step_ratio)
        )
        
        if step_index == cutoff_step:
            prompt_embeds = callback_kwargs[self.tensor_inputs[0]]
            prompt_embeds = prompt_embeds[-1:]  # "-1" denotes the embeddings for conditional text tokens.

            pipeline._guidance_scale = 0.0

            callback_kwargs[self.tensor_inputs[0]] = prompt_embeds
        
        return callback_kwargs
    
    def decode_latents(self, pipeline, latents):
        latent = pipeline.image_processor.postprocess(latents, output_type="pil", do_denormalize=[True] * latents.shape[0])[0]
        image_vae = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)
        image = pipeline.image_processor.postprocess(image_vae[0], output_type="pil", do_denormalize=[True] * image_vae[0].shape[0])[0]
        
        del image_vae

        return image, latent
    
    def load_decoder(self, generator, callback_func):
        self.images = None
        self.generator = generator
        self.callback_func = callback_func

class SDCFGControlNetCallback(PipelineCallback):
    def __init__(self):
        super().__init__()
    """
    Callback function for Stable Diffusion Pipelines. After certain number of steps (set by `cutoff_step_ratio` or
    `cutoff_step_index`), this callback will disable the CFG.

    Note: This callback mutates the pipeline by changing the `_guidance_scale` attribute to 0.0 after the cutoff step.
    """
    tensor_inputs = []

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        return {}

class SDCFGControlNetCallbackTrace(PipelineCallback):
    def __init__(self):
        super().__init__()
        self.images = None
        self.generator = None
        self.callback_func = None
    """
    Callback function for Stable Diffusion Pipelines. After certain number of steps (set by `cutoff_step_ratio` or
    `cutoff_step_index`), this callback will disable the CFG.

    Note: This callback mutates the pipeline by changing the `_guidance_scale` attribute to 0.0 after the cutoff step.
    """
    tensor_inputs = ["latents"]

    def callback_fn(self, pipeline, step_index, timestep, callback_kwargs) -> Dict[str, Any]:
        latents = callback_kwargs[self.tensor_inputs[0]]
        
        image, latent = self.decode_latents(latents=latents, pipeline=pipeline)

        if self.images:
            del self.images
            self.images = None

        self.images = (image, latent)
        
        if self.callback_func:
            self.callback_func((image, latent))

        return {}
    
    def decode_latents(self, pipeline, latents):
        latent = pipeline.image_processor.postprocess(latents, output_type="pil", do_denormalize=[True] * latents.shape[0])[0]
        image_vae = pipeline.vae.decode(latents / pipeline.vae.config.scaling_factor, return_dict=False, generator=self.generator)
        image = pipeline.image_processor.postprocess(image_vae[0], output_type="pil", do_denormalize=[True] * image_vae[0].shape[0])[0]
        
        del image_vae

        return image, latent
    
    def load_decoder(self, generator, callback_func):
        self.images = None
        self.generator = generator
        self.callback_func = callback_func

class DiffusersGenerate:
    def __init__(
            self, 
            model="", 
            lora="", 
            loradir="",
            lora_name="",
            lora_weight=0.0,
            vae="",
            model_cnet="",
            model_segmentation=""
        ):
        self.model = model
        self.model_cnet = model_cnet
        self.model_segmentation=model_segmentation
        self.lora = lora
        self.vae = vae
        self.loradir = loradir
        self.lora_name=lora_name
        self.lora_weight=lora_weight
        self.max_memory_mapping = {0: "8GB"}
        self.dtype = torch.bfloat16
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # or cpu
        self.callbacks = [
            SDCFGCutoffCallback(cutoff_step_ratio=None, cutoff_step_index=10),
            SDCFGCutoffCallbackTrace(cutoff_step_ratio=None, cutoff_step_index=10),
            SDCFGControlNetCallback(),
            SDCFGControlNetCallbackTrace()
        ]
        
        self.vae_load, self.pipeline = self.load_model(
            self.model, 
            self.vae, 
            self.lora, 
            self.loradir, 
            self.lora_name, 
            self.lora_weight
        )
        self.controlnet, self.controlnet_pipeline = self.load_controlnet_model(
            self.model, 
            self.model_cnet, 
            self.vae_load, 
            self.lora, 
            self.loradir, 
            self.lora_name, 
            self.lora_weight
        )

    def load_model(self, model, vae, lora, loradir, lora_name, lora_weight):
        vae_load, pipeline = (None, None)
        with torch.device(self.device):
            try:
                if vae:
                    vae_load = AutoencoderKL.from_single_file(vae, torch_dtype=self.dtype, device_map=self.device)
                pipeline = StableDiffusionPipeline.from_single_file(
                    model,
                    torch_dtype=self.dtype,
                    variant="fp16",
                    vae=vae_load,
                    device_map=self.device,
                )
                pipeline.schedular = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config, use_karras_sigmas=True, device_map=self.device)
                if lora:
                    pipeline.load_lora_weights(loradir, weight_name=lora, adapter_name=lora_name)
                    pipeline.set_adapters(lora_name, adapter_weights=lora_weight)
                
                pipeline.enable_model_cpu_offload(device=self.device)
                pipeline.unet = compile_regions(pipeline.unet, mode="reduce-overhead", fullgraph=True)
            except:
                pass

        return vae_load, pipeline

    def load_controlnet_model(self, model, model_cnet, vae_load, lora, loradir, lora_name, lora_weight, cnet_type:str=""):

        controlnet, controlnet_pipeline = (None, None)
        with torch.device(self.device):
            try:
                try:
                    controlnet = ControlNetModel.from_pretrained(model_cnet, torch_dtype=self.dtype, variant="fp16", device_map=self.device)
                except OSError as e:
                    print({e})
                    controlnet = ControlNetModel.from_pretrained(model_cnet, torch_dtype=self.dtype, device_map=self.device)

                match cnet_type:
                    case "inpaint":
                        controlnet_pipeline = StableDiffusionControlNetInpaintPipeline.from_single_file(model, controlnet=controlnet, torch_dtype=self.dtype, vae=vae_load, device_map=self.device)
                        controlnet_pipeline.schedular = DDIMScheduler.from_config(controlnet_pipeline.scheduler.config, use_karras_sigmas=True, device_map=self.device)
                    case "tile":
                        controlnet_pipeline = StableDiffusionPipeline.from_single_file(model, custom_pipeline="stable_diffusion_controlnet_img2img", controlnet=controlnet, torch_dtype=self.dtype, vae=vae_load, device_map=self.device)
                        controlnet_pipeline.schedular = DPMSolverMultistepScheduler.from_config(controlnet_pipeline.scheduler.config, use_karras_sigmas=True, device_map=self.device)
                    case _:
                        controlnet_pipeline = StableDiffusionControlNetPipeline.from_single_file(model, controlnet=controlnet, torch_dtype=self.dtype, vae=vae_load, device_map=self.device)
                        controlnet_pipeline.schedular = UniPCMultistepScheduler.from_config(controlnet_pipeline.scheduler.config, use_karras_sigmas=True, device_map=self.device)
                
                if lora:
                    controlnet_pipeline.load_lora_weights(loradir, weight_name=lora, adapter_name=lora_name)
                    controlnet_pipeline.set_adapters(lora_name, adapter_weights=lora_weight)

                controlnet_pipeline.enable_model_cpu_offload()
                controlnet_pipeline.unet = compile_regions(controlnet_pipeline.unet, mode="reduce-overhead", fullgraph=True)
            except:
                pass
        
        return controlnet, controlnet_pipeline
    
    def load_semantic_segmentation_model(self, model_segnmentation):
        
        image_processor, image_segmentor = (None, None)
        
        try:
            image_processor = AutoImageProcessor.from_pretrained(model_segnmentation, use_fast=True)
            image_segmentor = UperNetForSemanticSegmentation.from_pretrained(model_segnmentation)
        except:
            pass
        
        return image_processor, image_segmentor

    def reload_model(self, model, vae, lora, loradir, lora_name, lora_weight):
        (self.model, self.vae, self.lora, self.loradir, self.lora_name, self.lora_weight) = (
            model, vae, lora, loradir, lora_name, lora_weight
        )
        del self.vae_load, self.pipeline
        self.vae_load, self.pipeline =  self.load_model(
            self.model, 
            self.vae, 
            self.lora, 
            self.loradir, 
            self.lora_name, 
            self.lora_weight
        )
        torch.cuda.empty_cache()

    def reload_controlnet_model(self, model_cnet, cnet_type):
        
        self.model_cnet = model_cnet
        del self.controlnet, self.controlnet_pipeline
        self.controlnet, self.controlnet_pipeline = self.load_controlnet_model(
            model=self.model, 
            model_cnet=self.model_cnet,
            cnet_type=cnet_type,
            vae_load=self.vae_load,
            lora=self.lora,
            loradir=self.loradir,
            lora_name=self.lora_name,
            lora_weight=self.lora_weight
        )
        torch.cuda.empty_cache()

    def generate_image(self, prompt="", neg_prompt="", width=512, height=512, steps=20, guidance=7.5, seed=0, 
                       step_visualize=False, callback_func=None):
        (image, seed_num) = (None, 0)
        
        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                with torch.device(self.device):
                    (positive_embeds, negative_embeds) = get_weighted_text_embeddings_sd15(self.pipeline, prompt=prompt, neg_prompt=neg_prompt)
                    seed_num = seed if seed >= 0 else torch.randint(0, 65535, (1,)).item()
                    generator = torch.Generator(self.device).manual_seed(seed_num)
                    
                    if step_visualize:
                        callback_on_step_end = self.callbacks[1]
                        callback_on_step_end.load_decoder(generator=generator, callback_func=callback_func)
                        output_type="latent"
                    else:
                        callback_on_step_end = self.callbacks[0]
                        output_type="pil"
                    
                    with torch.no_grad():
                        output = self.pipeline(
                            generator=generator,
                            prompt_embeds=positive_embeds,
                            negative_prompt_embeds=negative_embeds,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            width=width,
                            height=height,
                            output_type=output_type,
                            callback_on_step_end=callback_on_step_end,
                        )
            image = callback_on_step_end.images[0] if step_visualize else output.images[0].copy()
        except Exception as e:
            print(e)
            # エラーの時は黒の画像
            image = Image.new("RGB", (width, height), (0, 0, 0))
            pass
        
        del positive_embeds, negative_embeds, output
        gc.collect()
        torch.cuda.empty_cache()

        return image, seed_num

    def apply_bayer_pattern(self, image):
        height, width = image.shape[:2]
        bayer_image = np.zeros((height, width), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                if (x % 2 == 0 and y % 2 == 0):  # Red
                    bayer_image[y, x] = image[y, x, 2]  # Redチャネル (BGR)
                elif (x % 2 == 1 and y % 2 == 0):  # Green
                    bayer_image[y, x] = image[y, x, 1]  # Greenチャネル
                elif (x % 2 == 0 and y % 2 == 1):  # Green
                    bayer_image[y, x] = image[y, x, 1]  # Greenチャネル
                else:  # Blue
                    bayer_image[y, x] = image[y, x, 0]  # Blueチャネル (BGR)

        return bayer_image

    def quantize_image(self, image, num_colors, apply_bayer:bool=False):

        # Bayerパターンを適用するオプションチェック
        if apply_bayer:
            image = self.apply_bayer_pattern(image)
            # 画像をfloat32型に変換
            img = image.astype(np.float32) / 255.0
            Z = img.reshape((-1, 2))  # 画像を1次元に変換
            Z = np.float32(Z)
        else:
            # 画像をfloat32型に変換
            img = image.astype(np.float32) / 255.0
            Z = img.reshape((-1, 3))  # 画像を1次元に変換
            Z = np.float32(Z)

        # k-meansクラスタリングを使って色を量子化
        # k-meansの停止条件
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # k-meansを適用
        _, labels, centers = cv2.kmeans(Z, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # 量子化された画像を作成
        quantized_image = centers[labels.flatten()].reshape(image.shape)
        quantized_image = (quantized_image * 255).astype(np.uint8)  # 0-255にスケーリング

        return quantized_image

    def usual_image(self, image_in=None, image_path:str="", width=512, height=512):
        if image_in is not None:
            image = image_in
        else:
            image = load_image(image_path)
            
        image = image.resize((width, height))

        return image
    
    def canny_image(self, image_in=None, image_path:str="", width=512, height=512, 
                    low_threshold:int=100, high_threshold:int=200, 
                    blur:bool=True, quant:bool=True, ksize=3, num_colors:int=2,
                    apply_bayer:bool=False):

        if image_in is not None:
            image = image_in
        else:
            image = load_image(image_path)

        if ksize % 2 == 0:
            ksize += 1
        
        image = image.resize((width, height))
        image = np.array(image)
        image = cv2.Canny(image, low_threshold, high_threshold)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = cv2.GaussianBlur(image, (ksize, ksize), 0) if blur else image
        image = self.quantize_image(image, num_colors=num_colors, apply_bayer=apply_bayer) if quant else image
        image = Image.fromarray(image)

        return image
    
    def make_inpaint_condition(self, image_in=None, image_path:str="", image_mask=None, width=512, height=512):

        if image_in is not None:
            image = image_in
        else:
            image = load_image(image_path)

        image_init = image.resize((width, height))
        image_mask = image_mask.resize((width, height))

        image = np.array(image_init.convert("RGB")).astype(np.float32) / 255.0
        image_graymask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

        image[image_graymask > 0.5] = -1.0  # set as masked pixel
        image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)

        return image_init, image_mask, image
    
    def openpose_image(self, image_in, image_path:str="", model_alias:str="human",width=512, height=512, radius=4, thickness=2, kpt_thr=0.5):
        if image_in is not None:
            image = image_in
        else:
            image = load_image(image_path)

        skelton_style = "openpose" if model_alias == "human" or model_alias == "wholebody" else "mmpose"

        image_init = np.array(image.resize((width, height)))
        
        with torch.no_grad():
            inferencer = MMPoseInferencer(model_alias)
            result_generator = inferencer(
                image_init,
                radius=radius,
                thickness=thickness,
                kpt_thr=kpt_thr,
                black_background=True,
                skeleton_style=skelton_style,
                return_vis=True)
        result = next(result_generator)
        image = Image.fromarray(result["visualization"][0])

        return image

    def lsd_image(self, image_in=None, image_path:str="", width=512, height=512):
        if image_in is not None:
            image = image_in
        else:
            image = load_image(image_path)
        image = np.array(image.resize((width, height)).convert('L'))
        image_lines = np.zeros((height, width, 3), dtype=np.uint8)

        lsd = cv2.createLineSegmentDetector()
        lines = lsd.detect(image)[0]
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image_lines, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), 1)
        
        image = Image.fromarray(image_lines)

        return image
    
    def depth_image(self, image_in=None, image_path:str="", width=512, height=512):
        if image_in is not None:
            image = image_in
        else:
            image = load_image(image_path)
        
        with torch.device(self.device):
            depth_estimator = pipeline('depth-estimation')
        
        with torch.no_grad():
            image = depth_estimator(image)['depth']

        image = np.array(image.resize((width, height)))
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        control_image = Image.fromarray(image)

        return control_image

    def segment_image(self, image_in=None, image_path:str="", width=512, height=512):
        if image_in is not None:
            image = image_in
        else:
            image = load_image(image_path)
        
        image = image.resize((width, height))
        image_processor, image_segmentor = self.load_semantic_segmentation_model(self.model_segmentation)

        pixel_values = image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = image_segmentor(pixel_values)
        seg = image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
        
        for label, color in enumerate(ada_palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        control_image = Image.fromarray(color_seg)

        return control_image
    
    def shuffle_block_image(self, image, width=512, height=512, block_k=6):
        
        # 比率からブロックサイズの計算
        block_size = 2**block_k

        # ブロックを保持するリスト
        blocks = []
        
        # 画像をブロックに分割
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                block = image[y:y + block_size, x:x + block_size]
                blocks.append(block)
        
        # 配列をシャッフル
        np.random.shuffle(blocks)

        # 新しい画像を作成
        shuffled_image = np.zeros_like(image)
        
        # シャッフルしたブロックを新しい画像に配置
        index = 0
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                if index < len(blocks):
                    shuffled_image[y:y + block_size, x:x + block_size] = blocks[index]
                    index += 1

        return shuffled_image
    
    def shuffle_swirl_effect_image(self, image, width=512, height=512, shuffle_k=5):
        center = (width // 2, height // 2)  # 画像の中心
        radius = np.sqrt(width**2 + height**2) // 2
        
        shuffled_image = np.zeros_like(image)

        for y in range(height):
            for x in range(width):
                # 中心点からの距離を計算
                dx = x - center[0]
                dy = y - center[1]
                distance = np.sqrt(dx**2 + dy**2)

                if distance < radius:
                    # 渦巻きの効果を適用
                    angle = shuffle_k * (radius - distance) / radius
                    new_x = int(center[0] + dx * np.cos(angle) - dy * np.sin(angle))
                    new_y = int(center[1] + dx * np.sin(angle) + dy * np.cos(angle))

                    # 新しい位置が画像の範囲内であるかチェック
                    if 0 <= new_x < width and 0 <= new_y < height:
                        shuffled_image[y, x] = image[new_y, new_x]
                    else:
                    # 範囲外の場合は元の画像のピクセルを使用
                        shuffled_image[y, x] = image[y, x]
                else:
                # 効果が適用されていない部分は元の画像を使用
                    shuffled_image[y, x] = image[y, x]

        return shuffled_image

    def shuffle_image(self, image_in=None, image_path:str="", width=512, height=512, shuffle_k=5, block:bool=False, block_k=6):
        if image_in is not None:
            image = image_in
        else:
            image = load_image(image_path)
        
        image = np.array(image.resize((width, height)).convert('RGB'))
        

        if block:
            image = self.shuffle_block_image(
                image=image,
                width=width,
                height=height,
                block_k=block_k
            )

        shuffuled_image = self.shuffle_swirl_effect_image(
            image=image,
            width=width,
            height=height,
            shuffle_k=shuffle_k)

        control_image = Image.fromarray(shuffuled_image)

        return control_image
    
    def normal_map_image(self, image_in, image_path:str="", width=512, height=512):
        if image_in is not None:
            image = image_in
        else:
            image = load_image(image_path)

        # 画像をグレースケールに変換
        gray_image = np.array(image.resize((width, height)).convert('L'))
        
        # Sobelフィルタを使って勾配を計算
        grad_x = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
        
        # 法線マップを計算
        normal_map = np.zeros((gray_image.shape[0], gray_image.shape[1], 3), dtype=np.float32)
        normal_map[..., 0] = grad_x  # x方向の勾配
        normal_map[..., 1] = grad_y  # y方向の勾配
        normal_map[..., 2] = 1.0      # z方向の法線成分

        # 法線の正規化
        norm = np.sqrt(normal_map[..., 0]**2 + normal_map[..., 1]**2 + normal_map[..., 2]**2)
        
        normal_map[..., 0] /= norm
        normal_map[..., 1] /= norm
        normal_map[..., 2] /= norm

        # 法線マップを0-255の範囲にスケール
        normal_map = ((normal_map + 1) / 2 * 255).astype(np.uint8)
        
        control_image = Image.fromarray(normal_map)

        return control_image

    def process_image(self, image_in=None, image_path:str="", width=512, height=512, 
                      cnet_type:str="canny", low_threshold=100, high_threshold=200,
                      blur:bool=True, quant:bool=True, bayer:bool=False, ksize=3, num_colors:int=2,
                      image_mask=None, openpose_alias:str="human", radius=4, thickness=1, kpt_thr=0.5, shuffle_k=5, block:bool=False, block_k=6):
        init_image, mask_image, image_control = (None, None, None)
        try:
            match cnet_type:
                case "canny" | "lineart" | "lineart_anime" | "scribble" | "softedge":
                    image_control = self.canny_image(
                        image_in=image_in,
                        image_path=image_path, 
                        width=width, 
                        height=height, 
                        low_threshold=low_threshold, 
                        high_threshold=high_threshold,
                        blur=blur,
                        quant=quant,
                        apply_bayer=bayer,
                        ksize=ksize,
                        num_colors=num_colors
                    )
                case "inpaint":
                    init_image, mask_image, image_control = self.make_inpaint_condition(
                        image_in=image_in, 
                        image_path=image_path, 
                        image_mask=image_mask,
                        width=width,
                        height=height)
                case "mlsd":
                    image_control = self.lsd_image(
                        image_in=image_in,
                        image_path=image_path,
                        width=width,
                        height=height
                    )
                case "depth":
                    image_control = self.depth_image(
                        image_in=image_in,
                        image_path=image_path,
                        width=width,
                        height=height
                    )
                case "normalbae":
                    image_control = self.normal_map_image(
                        image_in=image_in,
                        image_path=image_path,
                        width=width,
                        height=height
                    )
                case "seg":
                    image_control = self.segment_image(
                        image_in=image_in,
                        image_path=image_path,
                        width=width,
                        height=height
                    )
                case "openpose":
                    image_control = self.openpose_image(
                        image_in=image_in,
                        image_path=image_path,
                        model_alias=openpose_alias,
                        width=width,
                        height=height,
                        radius=radius,
                        thickness=thickness,
                        kpt_thr=kpt_thr
                    )
                case "shuffle":
                    image_control = self.shuffle_image(
                        image_in=image_in,
                        image_path=image_path,
                        width=width,
                        height=height,
                        shuffle_k=shuffle_k,
                        block=block,
                        block_k=block_k
                    )
                case _:
                    image_control = self.usual_image(
                        image_in=image_in,
                        image_path=image_path,
                        width=width,
                        height=height)
        except:
            # エラーの時は空の画像
            pass
        return init_image, mask_image, image_control
    
    def generate_image_controlnet(self, image_in=None, image_path:str="", prompt="", neg_prompt="", 
                                  width=512, height=512, steps=20, guidance=7.5, eta=1.0, seed=0, strength=1.0,
                                  cnet_type:str="canny", low_threshold=100, high_threshold=200,
                                  blur:bool=True, quant:bool=True, bayer:bool=False, ksize=3, num_colors:int=2,
                                  image_mask=None, openpose_alias:str="human", radius=4, thickness=1, kpt_thr=0.5, 
                                  shuffle_k=5, block:bool=False, block_k=6,
                                  step_visualize:bool=False, callback_func=None):
        (image, seed_num) = (None, 0)
        init_image, mask_image, image_control = self.process_image(
            image_in=image_in,
            image_path=image_path,
            width=width,
            height=height,
            cnet_type=cnet_type,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
            blur=blur,
            ksize=ksize,
            quant=quant,
            bayer=bayer,
            num_colors=num_colors,
            image_mask=image_mask,
            openpose_alias=openpose_alias,
            radius=radius,
            thickness=thickness,
            kpt_thr=kpt_thr,
            shuffle_k=shuffle_k,
            block=block,
            block_k=block_k
        )
        try:
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                with torch.device(self.device):
                    (positive_embeds, negative_embeds) = get_weighted_text_embeddings_sd15(self.pipeline, prompt=prompt, neg_prompt=neg_prompt)
                    seed_num = seed if seed >= 0 else torch.randint(0, 65535, (1,)).item()
                    generator = torch.Generator(self.device).manual_seed(seed_num)
                    
                    if step_visualize:
                        callback_on_step_end = self.callbacks[3]
                        callback_on_step_end.load_decoder(generator=generator, callback_func=callback_func)
                        output_type="latent"
                    else:
                        callback_on_step_end = self.callbacks[2]
                        output_type="pil"

                    with torch.no_grad():
                        if cnet_type == "inpaint":
                            output = self.controlnet_pipeline(
                                generator=generator,
                                image=init_image,
                                mask_image=mask_image,
                                control_image=image_control,
                                prompt_embeds=positive_embeds,
                                negative_prompt_embeds=negative_embeds,
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                eta=eta,
                                width=width,
                                height=height,
                                output_type=output_type,
                                callback_on_step_end=callback_on_step_end
                            )
                        elif cnet_type == "tile":
                            output = self.controlnet_pipeline(
                                generator=generator,
                                image=image_control,
                                controlnet_conditioning_image=image_control,
                                prompt_embeds=positive_embeds,
                                negative_prompt_embeds=negative_embeds,
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                strength=strength,
                                width=width,
                                height=height,
                                output_type=output_type,
                                callback_on_step_end=callback_on_step_end
                            )
                        else:
                            output = self.controlnet_pipeline(
                                generator=generator,
                                image=image_control,
                                prompt_embeds=positive_embeds,
                                negative_prompt_embeds=negative_embeds,
                                num_inference_steps=steps,
                                guidance_scale=guidance,
                                width=width,
                                height=height,
                                output_type=output_type,
                                callback_on_step_end=callback_on_step_end
                            )
            image = callback_on_step_end.images[0] if step_visualize else output.images[0].copy()   
        except Exception as e:
            error_class = type(e)
            error_description = str(e)
            err_msg = '%s: %s' % (error_class, error_description)
            print(err_msg)
            tb = traceback.extract_tb(sys.exc_info()[2])
            trace = traceback.format_list(tb)
            print('---- traceback ----')
            for line in trace:
                if '~^~' in line:
                    print(line.rstrip())
                else:
                    text = re.sub(r'\n\s*', ' ', line.rstrip())
                    print(text)
            print('-------------------')
            # エラーの時は黒の画像
            image = Image.new("RGB", (width, height), (0, 0, 0))
            pass

        del positive_embeds, negative_embeds, output
        gc.collect()
        torch.cuda.empty_cache()

        return image, seed_num

