
import comfy.model_management as mm
import torch
import os
import folder_paths
from PIL import Image, ImageOps, ImageSequence, ImageFile
from PIL.PngImagePlugin import PngInfo
import numpy as np
import base64
from io import BytesIO
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, BitsAndBytesConfig, set_seed
from qwen_vl_utils import process_vision_info
import logging
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Qwen2ModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": (
                [
                    "Qwen2-VL-2B-Instruct-GPTQ-Int4",
                    "Qwen2-VL-2B-Instruct-GPTQ-Int8",
                    "Qwen2-VL-2B-Instruct",
                    "Qwen2-VL-7B-Instruct-GPTQ-Int4",
                    "Qwen2-VL-7B-Instruct-GPTQ-Int8",
                    "Qwen2-VL-7B-Instruct",
                ],
                {"default": "Qwen2-VL-2B-Instruct"},
            ),
            "quantization": (
                ["none", "4bit", "8bit"],
                {"default": "none"},
            ),
            "precision": ([ 'fp16','bf16'],
                {
                "default": 'fp16'
                }),
            "attention": (
                [ 'flash_attention_2', 'sdpa', 'eager'],
                {
                "default": 'sdpa'
                }),
            }
        }

    RETURN_TYPES = ("QWEN2MODEL",)
    RETURN_NAMES = ("qwen2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "Qwen2"

    def loadmodel(self, model, precision, attention, quantization, lora=None):
        device = mm.get_torch_device()
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]

        # Non GPTQ models use this, but GPTQ not.
        # If you try to convert the GPTQ model in 4bit/8bit using bitsandbytes you get error.

        if "GPTQ" in model:
            quantization_config = None
        else:
            quantization_config = {"4bit": BitsAndBytesConfig(load_in_4bit=True), "8bit": BitsAndBytesConfig(load_in_8bit=True), "none": None}[quantization]


        # Check models path
        model_name = model.rsplit('/', 1)[-1]
        model_id = f"qwen/{model}"
        model_path = os.path.join(folder_paths.models_dir, "LLM", model_name)
        
        # Download model if it doesn't exist from Hugging Face Hub
        if not os.path.exists(model_path):
            # print(f"Downloading Qwen2 model to: {model_path}")
            logger.info(f"Downloading Qwen2 model to: {model_path}")

            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id,
                            local_dir=model_path,
                            local_dir_use_symlinks=False)

        # Load model
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, attn_implementation=attention, quantization_config=quantization_config, device_map=device, torch_dtype=dtype, trust_remote_code=True)
        
        qwen2_model = {
            'model_path': model_path, 
            'model': model, 
            #'processor': processor,
            # 'dtype': dtype
            }

        return (qwen2_model, )

class Qwen2ModelRunInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "qwen2_model": ("QWEN2MODEL", ),
                "text_input": ("STRING", {"default": "Describe this image in great detail in one paragraph.", "multiline": True}),
            },
            "optional": {
                "keep_model_loaded": ("BOOLEAN", {"default": False}),
                "always_generate_captions": ("BOOLEAN", {"default": False}),

                "max_new_tokens": (
                    "INT",
                    {
                        "default": 512, 
                        "min": 1,
                        "max": 1000000, 
                        "step": 1, 
                        "tooltip": "Max New Tokens maximum length of the newly generated generated text.If explicitly set to None it will be the model's max context length minus input length."
                    },
                ),


                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0, "max": 2,
                        "step": 0.1,
                        "tooltip": "This setting influences the variety in the model's responses. Lower values lead to more predictable and typical responses, while higher values encourage more diverse and less common responses. At 0, the model always gives the same response for a given input."
                    },
                ),


                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0,
                        "max": 1,
                        "step": 0.1,
                        "tooltip": "This setting limits the model's choices to a percentage of likely tokens: only the top tokens whose probabilities add up to P. A lower value makes the model's responses more predictable, while the default setting allows for a full range of token choices. Think of it like a dynamic Top-K."
                    },
                ),


                "min_p": (
                    "FLOAT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1,
                        "step": 0.1,
                        "tooltip": "Represents the minimum probability for a token to be considered, relative to the probability of the most likely token. (The value changes depending on the confidence level of the most probable token.) If your Min-P is set to 0.1, that means it will only allow for tokens that are at least 1/10th as probable as the best possible option."
                     },
                ),


                "top_k": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2048,
                        "step": 1,
                        "tooltip": "This limits the model's choice of tokens at each step, making it choose from a smaller set. A value of 1 means the model will always pick the most likely next token, leading to predictable results. By default this setting is disabled, making the model to consider all choices."
                    },
                ),


                # "top_a": (
                #     "FLOAT",
                #     {
                #         "default": 0,
                #         "min": 0,
                #         "max": 1,
                #         "step": 0.1,
                #         "tooltip": "Consider only the top tokens with 'sufficiently high' probabilities based on the probability of the most likely token. Think of it like a dynamic Top-P. A lower Top-A value focuses the choices based on the highest probability token but with a narrower scope. A higher Top-A value does not necessarily affect the creativity of the output, but rather refines the filtering process based on the maximum probability."
                #     },
                # ),


                # "frequency_penalty": (
                #     "FLOAT",
                #     {
                #         "default": 0,
                #         "min": -2,
                #         "max": 2,
                #         "step": 0.1,
                #         "tooltip": "This setting aims to control the repetition of tokens based on how often they appear in the input. It tries to use less frequently those tokens that appear more in the input, proportional to how frequently they occur. Token penalty scales with the number of occurrences. Negative values will encourage token reuse."
                #     },
                # ),


                # "presence_penalty": (
                #     "FLOAT",
                #     {
                #         "default": 0,
                #         "min": -2,
                #         "max": 2,
                #         "step": 0.1,
                #         "tooltip": "Adjusts how often the model repeats specific tokens already used in the input. Higher values make such repetition less likely, while negative values do the opposite. Token penalty does not scale with the number of occurrences. Negative values will encourage token reuse."
                #     },
                # ),


                "repetition_penalty": (
                    "FLOAT",
                    {
                        "default": 1,
                        "min": 0.1,
                        "max": 5, # some guides recomments 2 as max
                        "step": 0.1,
                        "tooltip": "Helps to reduce the repetition of tokens from the input. A higher value makes the model less likely to repeat tokens, but too high a value can make the output less coherent (often with run-on sentences that lack small words). Token penalty scales based on original token's probability."
                    },
                ),


                "min_pixels": (
                    "INT",
                    {
                        "default": 256,
                        "min": 4,
                        "max": 1280,
                        "step": 1,
                        "tooltip": "Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels"
                    },
                ),


                "max_pixels": (
                    "INT",
                    {
                        "default": 1280,
                        "min": 4,
                        "max": 16384,
                        "step": 1,
                        "tooltip": "Images will be resized to maintain their aspect ratio within the range of min_pixels and max_pixels"
                    },
                ),


                # "seed": (
                # "INT", 
                #     {
                #         "default": 1,
                #         "min": 1,
                #         "max": 0xffffffffffffffff
                #     }
                # ), # 9223372036854776000 as per docs (https://qwen.readthedocs.io/en/latest/)
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", )
    RETURN_NAMES = ("image", "caption", ) 
    FUNCTION = "inference"
    CATEGORY = "Qwen2"

    def hash_seed(self, seed):
        import hashlib
        # Convert the seed to a string and then to bytes
        seed_bytes = str(seed).encode('utf-8')
        # Create a SHA-256 hash of the seed bytes
        hash_object = hashlib.sha256(seed_bytes)
        # Convert the hash to an integer
        hashed_seed = int(hash_object.hexdigest(), 16)
        # Ensure the hashed seed is within the acceptable range for set_seed
        return hashed_seed % (2**32)

    @classmethod
    def IS_CHANGED(self, **kwargs):
        if 'always_generate_captions' in kwargs and kwargs['always_generate_captions']:
            return float("NaN")
        else:
            return hash(frozenset(kwargs))
        
    def inference(
        self,
        image,
        qwen2_model,
        text_input,
        max_new_tokens,
        temperature,
        top_p,
        min_p,
        top_k,
        # top_a,
        # frequency_penalty,
        # presence_penalty,
        repetition_penalty,
        min_pixels,
        max_pixels,
        # seed,
        keep_model_loaded,
        always_generate_captions
    ):

        #The model supports a wide range of resolution inputs. By default, it uses the native resolution for input, but higher resolutions can enhance performance at the cost of more computation. Users can set the minimum and maximum number of pixels to achieve an optimal configuration for their needs, such as a token count range of 256-1280, to balance speed and memory usage.
        min_pixels = min_pixels * 28 * 28
        max_pixels = max_pixels * 28 * 28

        # if seed:
        #     seedX = self.hash_seed(seed)
        #     set_seed(seedX)

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        model = qwen2_model['model']
        processor = AutoProcessor.from_pretrained(qwen2_model['model_path'], 
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            trust_remote_code=True)
        
        # dtype = qwen2_model['dtype']
        model.to(device)

        with torch.no_grad():
            imageForModel = Image2Base64.image_to_base64(image)

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image", 
                            "image": imageForModel
                        },
                        {
                            "type": "text", 
                            "text": text_input
                        },
                    ],
                }
            ]

            # Preparation for inference
            text_prompt = processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)

            image_inputs, video_inputs = process_vision_info(conversation)
            inputs = processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")

            # Inference: Generation of the output
            generated_ids = model.generate(**inputs, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            # top_a=top_a,
            # frequency_penalty=frequency_penalty,
            # presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            # seed=seed,
            )
            
            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )
        
            if not keep_model_loaded:
                model.to(offload_device)
                mm.soft_empty_cache()
            return (image, result, )

class Image2Base64:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
            },
        }
    
    # Add this helper function in the class
    @staticmethod
    def image_to_base64(image):
        pil_image = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        image_bytes = buffered.getvalue()
        img_str = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("imagebase64",) 
    FUNCTION = "convert"
    CATEGORY = "utilities"

    def convert(
        self,
        image,
    ):
        
        # Add base64 conversion
        base64_image = self.image_to_base64(image)
        return (base64_image,)

NODE_CLASS_MAPPINGS = {
    "Qwen2ModelLoader": Qwen2ModelLoader,
    "Qwen2ModelRunInference": Qwen2ModelRunInference,
    "Image2Base64": Image2Base64,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2ModelLoader": "Qwen2 Model Load / Download",
    "Qwen2ModelRunInference": "Qwen2 Run Inference",
}
