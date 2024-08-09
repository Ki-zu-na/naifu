from dataclasses import dataclass
import torch
import random


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
    extras: dict = None


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

def shuffle_prompts_dan_native_style(data_entry: dict, dan_probability: float = 0.7):
    """
    Process a data entry and return an Entry object with either 'dan' or 'native' caption.
    
    Args:
    data_entry (dict): The input data entry.
    dan_probability (float): Probability of choosing 'dan' caption. Default is 0.7.
    
    Returns:
    Entry: Processed Entry object.
    """
    # Check if the data entry has the required fields
    if not all(key in data_entry for key in ['train_use', 'train_caption_dan', 'train_caption_native', 'file_path', 'train_width', 'train_height']):
        raise ValueError("Invalid data entry format")
    
    # Randomly choose between 'dan' and 'native' based on probability
    use_dan = random.random() < dan_probability
    
    # Create an Entry object
    entry = Entry(
        is_latent=True,  # Assuming it's latent by default
        pixel=torch.zeros((3, data_entry['train_height'], data_entry['train_width'])),  # Placeholder tensor
        prompt=data_entry['train_caption_dan'] if use_dan else data_entry['train_caption_native'],
        extras={
            'file_path': data_entry['file_path'],
            'train_width': data_entry['train_width'],
            'train_height': data_entry['train_height'],
            'caption_type': 'dan' if use_dan else 'native'
        }
    )
    
    # Apply shuffle_prompts_sdstyle
    shuffled_entry = shuffle_prompts_sdstyle(entry)
    
    # Create a new Entry object with the shuffled prompt, keeping other attributes unchanged
    return Entry(
        is_latent=entry.is_latent,
        pixel=entry.pixel,
        prompt=shuffled_entry.prompt,
        extras=entry.extras
    )
