# import sys, os, importlib.util
# import transformers
# from transformers.utils import is_torch_available, is_tf_available

# print("python:", sys.executable)
# print("transformers:", transformers.__version__)
# print("is_torch_available():", is_torch_available())
# print("is_tf_available():", is_tf_available())
# print("find_spec('torch'):", importlib.util.find_spec('torch') is not None)
# print("env TRANSFORMERS_NO_PYTORCH =", os.environ.get("TRANSFORMERS_NO_PYTORCH"))
# print("env TRANSFORMERS_NO_TF      =", os.environ.get("TRANSFORMERS_NO_TF"))

# from transformers import pipeline
# from transformers.image_utils import load_image

# from huggingface_hub import HfApi
# TOKEN = "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"  # 一定是 Value，不是 Name
# api = HfApi()
# print("whoami:", api.whoami(token=TOKEN))
# print("model:", api.model_info("facebook/dinov3-vit7b16-pretrain-lvd1689m", token=TOKEN).modelId)


# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
# image = load_image(url)

# feature_extractor = pipeline(model="facebook/dinov3-vit7b16-pretrain-lvd1689m", task="image-feature-extraction", token=TOKEN,  trust_remote_code=True,)
# features = feature_extractor(image)


# import torch
# from transformers import AutoImageProcessor, AutoModel
# from transformers.image_utils import load_image

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = load_image(url)

# pretrained_model_name = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
# processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
# model = AutoModel.from_pretrained(
#     pretrained_model_name, 
#     device_map="auto", 
# )

# inputs = processor(images=image, return_tensors="pt").to(model.device)
# with torch.inference_mode():
#     outputs = model(**inputs)

# pooled_output = outputs.pooler_output
# print("Pooled output shape:", pooled_output.shape)



import os
from transformers import AutoModel, AutoImageProcessor, pipeline
from transformers.image_utils import load_image

TOKEN = "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"  # 一定是 Value，不是 Name

repo = "facebook/dinov3-vit7b16-pretrain-lvd1689m"

# 先显式把远程自定义代码加载进来（关键：trust_remote_code=True）
processor = AutoImageProcessor.from_pretrained(
    repo, token=TOKEN, trust_remote_code=True
)
model = AutoModel.from_pretrained(
    repo, token=TOKEN, trust_remote_code=True
)

# 再把“已加载好的对象”交给 pipeline（避免再次自动解析 model_type）
pipe = pipeline(
    task="image-feature-extraction",
    model=model,
    image_processor=processor,
    trust_remote_code=True,   # 这行保留无妨
    framework="pt",           # 强制 PyTorch，避免后端误判
)

img = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
)
features = pipe(img)
print(type(features), getattr(features, "shape", None))
