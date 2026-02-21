import re
import yaml
import pandas as pd


def prompt_augmentation(content, augment=True, celebrity=False):
    if not augment:
        return [content]

    if celebrity:
        prompts = [
            "A photo of {}".format(content),
            "A portrait of {}".format(content),
            "A headshot of {}".format(content),
            "A close-up photo of {}".format(content),
            "A professional photo of {}".format(content),
            "{} smiling".format(content),
            "{} looking at the camera".format(content),
            "A candid photo of {}".format(content),
            "{} at a press conference".format(content),
            "{} at an interview".format(content),
            "{} on the red carpet".format(content),
            "{} at a premiere".format(content),
            "{} at an award ceremony".format(content),
            "{} at a public event".format(content),
            "{} giving a speech".format(content),
            "A black and white photo of {}".format(content),
            "A vintage photo of {}".format(content),
            "A studio portrait of {}".format(content),
            "{} in formal attire".format(content),
            "{} in casual clothes".format(content),
            "A magazine cover featuring {}".format(content),
            "A paparazzi photo of {}".format(content),
            "{} waving to fans".format(content),
            "{} signing autographs".format(content),
            "A selfie of {}".format(content),
            "{} on a talk show".format(content),
            "{} being interviewed".format(content),
            "A painting of {}".format(content),
            "An oil portrait of {}".format(content),
            "A digital art of {}".format(content),
            "A sketch of {}".format(content),
            "A caricature of {}".format(content),
            "{} in a movie scene".format(content),
            "{} on set".format(content),
            "A promotional photo of {}".format(content),
            "{} posing for photographers".format(content),
            "An official photo of {}".format(content),
            "{} at a charity event".format(content),
            "{} at a film festival".format(content),
            "A young {}".format(content),
        ]
    else:
        prompts = [
            "{} in a photo".format(content),
            "{} in a snapshot".format(content),
            "A snapshot of {}".format(content),
            "A photograph showcasing {}".format(content),
            "An illustration of {}".format(content),
            "A digital rendering of {}".format(content),
            "A visual representation of {}".format(content),
            "A graphic of {}".format(content),
            "A shot of {}".format(content),
            "A photo of {}".format(content),
            "A black and white image of {}".format(content),
            "A depiction in portrait form of {}".format(content),
            "A scene depicting {} during a public gathering".format(content),
            "{} captured in an image".format(content),
            "A depiction created with oil paints capturing {}".format(content),
            "An image of {}".format(content),
            "A drawing capturing the essence of {}".format(content),
            "An official photograph featuring {}".format(content),
            "A detailed sketch of {}".format(content),
            "{} during sunset/sunrise".format(content),
            "{} in a detailed portrait".format(content),
            "An official photo of {}".format(content),
            "Historic photo of {}".format(content),
            "Detailed portrait of {}".format(content),
            "A painting of {}".format(content),
            "HD picture of {}".format(content),
            "Magazine cover capturing {}".format(content),
            "Painting-like image of {}".format(content),
            "Hand-drawn art of {}".format(content),
            "An oil portrait of {}".format(content),
            "{} in a sketch painting".format(content),
        ]

    return prompts


def load_config(config_path: str) -> tuple:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if len(config) == 1:
        config_name = list(config.keys())[0]
        return config[config_name], config_name
    return config, None


def coerce_prompt(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, (list, tuple, set)):
        return ", ".join(str(x).strip() for x in v if str(x).strip())
    s = str(v).strip()
    s = re.sub(r'^\s*prompt\s*[:\-]?\s*', "", s, flags=re.I)
    m = re.match(r'^\[\s*(.*)\s*\]$', s)
    if m:
        parts = [p.strip() for p in m.group(1).split(",")]
        return ", ".join(p for p in parts if p)
    return s
