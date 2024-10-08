import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(now_dir)
import torch
import folder_paths
from PIL import Image
import numpy as np
from huggingface_hub import snapshot_download
from accelerate import Accelerator
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTokenizer
from ip_adapter.ip_adapter_multi import IPAdapterXL, IPAdapterPlusXL

diffusers_dir = os.path.join(folder_paths.models_dir,"diffusers")
RealVisXL_dir = os.path.join(diffusers_dir, "RealVisXL_V1.0")
sdxl_dir = os.path.join(diffusers_dir, "stable-diffusion-xl-base-1.0")
ckpt_dir = os.path.join(now_dir,"checkpoints")
ip_adapter_dir = os.path.join(ckpt_dir,"IP-Adapter")
mip_dir = os.path.join(ckpt_dir,"MIP-Adapter")


def get_token_len(entity_name, tokenizer):
    entity_tokens = tokenizer(entity_name, return_tensors="pt").input_ids[0][1:-1]
    return len(entity_tokens)

class PromptTextNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "dynamicPrompts": True,}), 
            }
        }
    RETURN_TYPES = ("TEXT",)    
    FUNCTION = "encode"

    CATEGORY = "AIFSH_MIP-Adapter"
    def encode(self, text):
        return (text, )

class MIP_AdapterNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "prompt":("TEXT",),
                "reference_image1":("IMAGE",),
                "reference_image2":("IMAGE",),
                "entity1_name":("STRING",{
                    "default": "cat"
                }),
                "entity2_name":("STRING",{
                    "default": "jacket"
                }),
                "height":("INT",{
                    "default": 576,
                }),
                "width":("INT",{
                    "default": 1024,
                }),
                "num_samples":("INT",{
                    "default": 1,
                }),
                "scale":("FLOAT",{
                    "default": 0.6,
                }),
                "is_plus":("BOOLEAN",{
                    "default": False
                }),
                "seed":("INT",{
                    "default": 42
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    #RETURN_NAMES = ("image_output_name",)

    FUNCTION = "gen_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_MIP-Adapter"

    def gen_img(self,prompt,reference_image1,reference_image2,
                entity1_name,entity2_name,height,width,num_samples,
                scale,is_plus,seed):
        
        if not os.path.exists(os.path.join(mip_dir,"model_sdxl_plus.bin")):
            snapshot_download(repo_id="hqhQAQ/MIP-Adapter",local_dir=mip_dir)
        
        if not os.path.exists(os.path.join(ip_adapter_dir,"sdxl_models","image_encoder","model.safetensors")):
            snapshot_download(repo_id="h94/IP-Adapter",
                              allow_patterns=["*.json","*del.safetensors"],
                              local_dir=ip_adapter_dir)

        if is_plus:
            if not os.path.exists(os.path.join(RealVisXL_dir,"unet/diffusion_pytorch_model.fp16.safetensors")):
                snapshot_download(repo_id="SG161222/RealVisXL_V1.0",
                                local_dir=RealVisXL_dir,
                                allow_patterns=["*fp16*","*json","*txt"])
            base_model_path = RealVisXL_dir
            ip_ckpt_path = os.path.join(mip_dir,"model_sdxl_plus.bin")
            image_encoder_path = os.path.join(ip_adapter_dir,"models","image_encoder")
        else:
            if not os.path.exists(os.path.join(sdxl_dir,"unet/diffusion_pytorch_model.fp16.safetensors")):
                snapshot_download(repo_id="stabilityai/stable-diffusion-xl-base-1.0",
                                local_dir=sdxl_dir,
                                allow_patterns=["*fp16*","*json","*txt"])
            base_model_path = sdxl_dir
            ip_ckpt_path = os.path.join(mip_dir,"model_sdxl.bin")
            image_encoder_path = os.path.join(ip_adapter_dir,"sdxl_models","image_encoder")

        # accelerator = Accelerator()
        device = "cuda"
        num_tokens = 4 if not is_plus else 16
        num_objects = 2

        # Load model
        tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer",torch_dtype=torch.float16,)
        # tokenizer_2 = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer_2")

        # Load SDXL pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            add_watermarker=False,
            variant="fp16"
        )
        pipe.enable_vae_slicing()
        pipe.to(device)

        # Load ip-adapter
        ip_params = {
            'num_tokens': num_tokens,
            'num_objects': num_objects,
        }
        cur_model = IPAdapterPlusXL if is_plus else IPAdapterXL
        
        ip_model = cur_model(pipe, image_encoder_path, state_dict=None, ip_ckpt=ip_ckpt_path, device=device, ip_params=ip_params)
        
        reference_image1 = reference_image1.numpy()[0] * 255
        reference_image1 = reference_image1.astype(np.uint8)
        reference_image2 = reference_image2.numpy()[0] * 255
        reference_image2 = reference_image2.astype(np.uint8)

        images = [[Image.fromarray(reference_image1)], [Image.fromarray(reference_image2)] ]
        prompts = [prompt]
        # entity1_name, entity2_name = "img1", "img2"
        entity_names = [[entity1_name, entity2_name]]
        entity_indexes = [[(-1, get_token_len(entity1_name, tokenizer)), (-1, get_token_len(entity2_name, tokenizer))]]

        generated_images = ip_model.generate(pil_images=images, num_samples=num_samples, num_inference_steps=30, seed=seed, prompt=prompts, scale=scale, entity_names=entity_names, entity_indexes=entity_indexes,height=height,width=width)
        img_np = []
        for idx, image in enumerate(generated_images):
            img_np.append(np.array(image)/255)
        out_img = torch.from_numpy(np.stack(img_np))
        # print(out_img.shape)
        return (out_img,)
    
NODE_CLASS_MAPPINGS = {
    "PromptTextNode": PromptTextNode,
    "MIP_AdapterNode":MIP_AdapterNode
}