from normal_api import normal_api_request, raw2text
from tqdm import tqdm

configs = [
    {
        "key": "xxx",
        "url": "https://xxx",
        "model": "xxx",
        "concurrence": 200
    },
    {
        "key": "xxx",
        "url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
        "concurrence": 400
    }
]


if __name__ == "__main__":
    for config in tqdm(configs, leave=False, desc="processing model..."):
        zero = normal_api_request(**config, zero_shot=True)
        raw2text(zero)
        few = normal_api_request(**config, zero_shot=False)
        raw2text(few)
