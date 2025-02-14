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
    # constants
    shuffle_caption = True
    token_warmup_step = 0  # unsupported
    caption_tag_dropout_rate = 0.15
    caption_separator = ","
    keep_tokens_separator = "|||"
    replacements = {}
    
    # New parameters for dropping all tags before or after keep_tokens_separator
    drop_all_fixed_prob = 0.05  # Probability to drop all fixed tokens
    drop_all_flex_prob = 0.15    # Probability to drop all flex tokens
    
    # New parameter for dropping the first tag when there's no keep_tokens_separator
    drop_first_tag_prob = 0.05   # Probability to drop the first tag when there's no separator
    drop_second_tag_prob = 0.15   # Probability to drop the first tag when there's no separator

    if keep_tokens_separator not in e.prompt:
        # Handle the case when there's no keep_tokens_separator
        tags = [t.strip() for t in e.prompt.split(caption_separator) if t.strip()]
        
        if tags and random.random() < drop_first_tag_prob:
            # Drop the first tag
            tags = tags[1:]
        
        e.prompt = caption_separator.join(tags)
        return e
    
    caption = e.prompt
    fixed_part, flex_part = caption.split(keep_tokens_separator, 1)
    fixed_tokens = [t.strip() for t in fixed_part.split(caption_separator) if t.strip()]
    flex_tokens = [t.strip() for t in flex_part.split(caption_separator) if t.strip()]

    drop_second_tag = random.random() < drop_second_tag_prob
    if len(fixed_tokens) >= 2 and drop_second_tag:
        fixed_tokens = fixed_tokens[:1] + fixed_tokens[2:] 

    # Decide whether to drop all fixed or flex tokens
    drop_all_fixed = random.random() < drop_all_fixed_prob
    drop_all_flex = random.random() < drop_all_flex_prob

    if drop_all_fixed:
        fixed_tokens = []
    else:
        if shuffle_caption:
            random.shuffle(fixed_tokens) 
        
        # Apply individual token dropout to fixed tokens
        fixed_tokens = dropout_tags(fixed_tokens, caption_tag_dropout_rate)

    if drop_all_flex:
        flex_tokens = []
    else:
        if shuffle_caption:
            random.shuffle(flex_tokens)
        
        # Apply individual token dropout to flex tokens
        flex_tokens = dropout_tags(flex_tokens, caption_tag_dropout_rate)

    caption = caption_separator.join(fixed_tokens + flex_tokens)

    for str_from, str_to in replacements.items():
        if str_from == "":
            # replace all
            if isinstance(str_to, list):
                caption = random.choice(str_to)
            else:
                caption = str_to
        else:
            caption = caption.replace(str_from, str_to)
    
    e.prompt = caption
    return e

def dropout_tags(tokens, dropout_rate):
    return [token for token in tokens if random.random() > dropout_rate]

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

def process_prompts_with_metadata(
    data_entry: Entry,
) -> Entry:

    shuffle_caption = True
    # New parameters for dropping all tags before or after keep_tokens_separator
    drop_all_fixed_prob = 0.05  # Probability to drop all fixed tokens
    drop_all_flex_prob = 0.15    # Probability to drop all flex tokens
    drop_artist_prob = 0.05
    dropout_rate = 0.15
    caption_nl_prob = 0.5
    style_mix_prob = 0.5
    add_fixed_prefix_prob = 0.3

    if not data_entry.extras:
        return data_entry

    extras = data_entry.extras

    fixed_tags = []
    drop_artist = random.random() < drop_artist_prob

    def add_prefix_to_tags(tags_str, prefix):
        if not tags_str:
            return []
        tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
        if random.random() < add_fixed_prefix_prob:
            return [f"{prefix}:{tag}" for tag in tags]
        return tags

    if 'tag_string_artist' in extras and extras['tag_string_artist'] and not drop_artist:
        fixed_tags.extend(add_prefix_to_tags(extras['tag_string_artist'], "artist"))
    if 'tag_string_character' in extras and extras['tag_string_character']:
        fixed_tags.extend(add_prefix_to_tags(extras['tag_string_character'], "character"))
    if 'tag_string_copyright' in extras and extras['tag_string_copyright']:
        copyright_tags = add_prefix_to_tags(extras['tag_string_copyright'], "copyright")
        copyright_tags = [tag for tag in copyright_tags if "original" not in tag]
        fixed_tags.extend(copyright_tags)

    flex_tags = []
    caption_nl = random.random() < caption_nl_prob
    style_mix = random.random() < style_mix_prob
    if 'tag_string_general' in extras and extras['tag_string_general'] and not caption_nl:
        flex_tags = [t.strip() for t in extras['tag_string_general'].split(",") if t.strip()]
        if  'rating' in extras and extras['rating']:
            rating_tags = []
            rating = extras['rating']
            if rating == 'e':
                rating_tags = ["explicit"]
            elif rating == 's':
                rating_tags = ["sensitive"]
            elif rating == 'q':
                rating_tags = ["nsfw"] 
            elif rating == 'g':
                rating_tags = ["safe"]
            flex_tags.extend(add_prefix_to_tags(", ".join(rating_tags), "rating")) # rating 作为一个整体添加前缀
        if 'aes_rating' in extras and extras['aes_rating']:
            flex_tags.append(extras['aes_rating'])
        if 'tag_string_meta' in extras and extras['tag_string_meta']:
            flex_tags.append(extras['tag_string_meta'])

    elif 'regular_summary' in extras and extras['regular_summary'] and caption_nl:
        if 'regular_summary' in extras and extras['regular_summary']:
            flex_tags.append(extras['regular_summary'])
        elif 'brief_summary' in extras and extras['brief_summary'] and style_mix:
            flex_tags.append(extras['brief_summary'])

    # Decide whether to drop all fixed or flex tokens
    drop_all_fixed = random.random() < drop_all_fixed_prob
    drop_all_flex = random.random() < drop_all_flex_prob

    if drop_all_fixed:
        fixed_tags = []

    if drop_all_flex:
        flex_tags = []
    else:
        flex_tags = dropout_tags(flex_tags, dropout_rate)
    if shuffle_caption:
        random.shuffle(fixed_tags)
        random.shuffle(flex_tags)

    new_prompt = ", ".join(fixed_tags + flex_tags)
    new_prompt = new_prompt.replace("_", " ") # 将下划线替换成空格

    return Entry(
        is_latent=data_entry.is_latent,
        pixel=data_entry.pixel,
        prompt=new_prompt,
        original_size=data_entry.original_size,
        cropped_size=data_entry.cropped_size,
        dhdw=data_entry.dhdw,
        extras=extras
    )