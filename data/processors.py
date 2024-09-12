from dataclasses import dataclass
import torch
import random
from typing import  Optional

@dataclass
class Entry:
    """
    This class represents an entry in a batch of image data. Each entry contains information about an image and its associated prompt.

    Attributes:
        is_latent (bool): A flag indicating whether the image is in latent space.
        pixel (torch.Tensor): The pixel data of the image.
        prompt (str): The prompt associated with the image.
        extras (dict): A dictionary to store any extra information associated with the image.
        
    """
    is_latent: bool
    pixel: torch.Tensor
    prompt: str
    original_size: tuple[int, int]  # h, w
    cropped_size: Optional[tuple[int, int]]  # h, w
    dhdw: Optional[tuple[int, int]]  # dh, dw
    extras: dict = None
    # mask: torch.Tensor | None = None

def identical(inputs: Entry):
    return inputs

def shuffle_prompts(e: Entry):
    e.prompt = e.prompt.split(", ")
    random.shuffle(e.prompt)
    e.prompt = ", ".join(e.prompt)
    return e

def dropout_tags(tokens, dropout=0):
    if dropout <= 0:
        return tokens
    l = []
    for token in tokens:
        if random.random() >= dropout:
            l.append(token)
    return l

def shuffle_prompts_sdstyle(e: Entry):
    # constrants
    shuffle_caption = True
    token_warmup_step = 0 # unsupported
    caption_tag_dropout_rate = 0.15
    caption_separator = ","
    keep_tokens_separator = "|||"
    replacements = {}
    
    if keep_tokens_separator not in e.prompt:
        return e
    
    caption = e.prompt
    fixed_part, flex_part = caption.split(keep_tokens_separator, 1)
    fixed_tokens = [t.strip() for t in fixed_part.split(caption_separator) if t.strip()]
    flex_tokens = [t.strip() for t in flex_part.split(caption_separator) if t.strip()]

    if shuffle_caption:
        random.shuffle(flex_tokens)
        
    # dropout flex tags by rate
    flex_tokens = dropout_tags(flex_tokens, caption_tag_dropout_rate)
    caption = ", ".join(fixed_tokens + flex_tokens)

    for str_from, str_to in replacements.items():
        if str_from == "":
            # replace all
            if type(str_to) == list:
                caption = random.choice(str_to)
            else:
                caption = str_to
        else:
            caption = caption.replace(str_from, str_to)
    
    e.prompt = caption
    return e

import random

def shuffle_prompts_dan_native_style(data_entry: Entry, dan_probability: float = 0.7):
    """
    Process an Entry object and return a new Entry object with either 'dan' or 'native' caption.
    If 'native' caption is empty, 'dan' caption is used regardless of probability.
    
    Args:
    data_entry (Entry): The input Entry object.
    dan_probability (float): Probability of choosing 'dan' caption. Default is 0.7.
    
    Returns:
    Entry: Processed Entry object with updated prompt and extras.
    """
    # Check if the data entry has the required fields in extras
    if not data_entry.extras or 'train_caption_dan' not in data_entry.extras or 'train_caption_native' not in data_entry.extras:
        raise ValueError("Missing 'train_caption_dan' or 'train_caption_native' in extras")
    
    # If 'native' caption is empty, use 'dan' caption
    if not data_entry.extras['train_caption_native']:
        new_prompt = data_entry.extras['train_caption_dan']
        caption_type = 'dan'
    else:
        # Randomly choose between 'dan' and 'native' based on probability
        use_dan = random.random() < dan_probability
        new_prompt = data_entry.extras['train_caption_dan'] if use_dan else data_entry.extras['train_caption_native']
        caption_type = 'dan' if use_dan else 'native'
    
    # Create a new extras dictionary with the added caption_type
    new_extras = data_entry.extras.copy()
    new_extras['caption_type'] = caption_type
    
    # Create a new Entry object, inheriting most attributes from the original
    new_entry = Entry(
        is_latent=data_entry.is_latent,
        pixel=data_entry.pixel,
        prompt=new_prompt,
        original_size=data_entry.original_size,
        cropped_size=data_entry.cropped_size,
        dhdw=data_entry.dhdw,
        extras=new_extras
    )
    
    # Apply shuffle_prompts_sdstyle (assuming this function exists)
    return shuffle_prompts_sdstyle(new_entry)