# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# setup.py is the fallback installation script when pyproject.toml does not work
import os
from pathlib import Path

from setuptools import find_packages, setup

version_folder = os.path.dirname(os.path.join(os.path.abspath(__file__)))

with open(os.path.join(version_folder, "verl/version/version")) as f:
    __version__ = f.read().strip()

install_requires = [
    "accelerate",
    "cachetools",
    "codetiming",
    "datasets",
    "dill",
    "fastapi",
    "hydra-core",
    "numpy>=2.0.0",
    "orjson",
    "pandas",
    "peft",
    "pyarrow>=19.0.0",
    "pybind11",
    "pylatexenc",
    "ray[default]>=2.41.0",
    "torchdata",
    "transferqueue==0.1.8",
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    # 5.6.0 ships a broken flash-attention path (crashes on s_aux=None for
    # sink-less models); fixed in 5.6.1. See huggingface/transformers#45588.
    "transformers<5.15.0,!=5.6.0",
    "uvicorn",
    "wandb",
    "packaging>=20.0",
    "tensorboard",
]

TEST_REQUIRES = ["pytest", "pre-commit", "py-spy", "pytest-asyncio", "pytest-rerunfailures"]
# PRIME's local code scorer now uses the standard library module runtime.
# Keep the extra name as a compatibility no-op for existing install commands.
PRIME_REQUIRES = []
GEO_REQUIRES = ["mathruler", "torchvision", "qwen_vl_utils"]
GPU_REQUIRES = ["liger-kernel", "flash-attn"]
QLORA_REQUIRES = ["bitsandbytes>=0.43.3"]
MATH_REQUIRES = ["math-verify"]  # Add math-verify as an optional dependency
VLLM_REQUIRES = [
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "vllm @ git+https://github.com/carbonteq-ai/vllm.git@7817d845727af570352622dc8d58f2d43c76d89d",
]
TRTLLM_REQUIRES = ["tensorrt-llm>=1.2.0rc6"]
SGLANG_REQUIRES = [
    "tensordict>=0.8.0,<=0.10.0,!=0.9.0",
    "sglang[srt,openai]==0.5.8",
    "torch==2.9.1",
]
TRL_REQUIRES = ["trl<=0.9.6"]

extras_require = {
    "test": TEST_REQUIRES,
    "prime": PRIME_REQUIRES,
    "geo": GEO_REQUIRES,
    "gpu": GPU_REQUIRES,
    "qlora": QLORA_REQUIRES,
    "math": MATH_REQUIRES,
    "vllm": VLLM_REQUIRES,
    "sglang": SGLANG_REQUIRES,
    "trl": TRL_REQUIRES,
    "trtllm": TRTLLM_REQUIRES,
}


this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="verl",
    version=__version__,
    package_dir={"": "."},
    packages=find_packages(where="."),
    url="https://github.com/verl-project/verl",
    license="Apache 2.0",
    author="Bytedance - Seed - MLSys",
    author_email="zhangchi.usc1992@bytedance.com, gmsheng@connect.hku.hk",
    description="verl: Volcano Engine Reinforcement Learning for LLM",
    install_requires=install_requires,
    extras_require=extras_require,
    package_data={
        "": ["version/*"],
        "verl": [
            "trainer/config/*.yaml",
            "trainer/config/*/*.yaml",
            "experimental/*/config/*.yaml",
        ],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
