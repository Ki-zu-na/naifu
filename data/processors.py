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

def shuffle_prompts_dan_native_style(data_entry: Entry):
    """
    Process an Entry object and return a new Entry object with either 'dan' or 'native' caption.
    If 'native' caption is empty, 'dan' caption is used regardless of probability.
    
    Args:
    data_entry (Entry): The input Entry object.
    dan_probability (float): Probability of choosing 'dan' caption. Default is 0.7.
    
    Returns:
    Entry: Processed Entry object with updated prompt and extras.
    """
    dan_probability: float = 0.7
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
    # Probabilities for various augmentations
    drop_all_fixed_prob = 0.1
    drop_all_flex_prob = 0.2
    drop_artist_prob = 0.05
    dropout_rate = 0.3  # General dropout rate for tags
    caption_nl_prob = 0.5  # Probability of using natural language summaries
    style_mix_prob = 0.5  # Probability of mixing brief/regular summary if both exist
    add_fixed_prefix_prob = 0.3
    add_underline_prob = 0.1

    if not data_entry.extras:
        # Fallback if no extras are provided, process the existing prompt
        train_caption = data_entry.prompt
        train_caption = train_caption.replace("_", " ") # Replace underscores with spaces
        train_caption_tags = [tag.strip() for tag in train_caption.split(",") if tag.strip()]

        if shuffle_caption:
            random.shuffle(train_caption_tags)
        new_prompt = ", ".join(train_caption_tags)
        data_entry.prompt = new_prompt
        return data_entry

    extras = data_entry.extras
    
    # Nested helper functions for tag manipulation
    def add_prefix_to_tags(tags_str, prefix):
        if not tags_str:
            return []
        tags = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
        # Only add prefix if add_fixed_prefix_prob is met, but apply to all tags if so.
        # The original logic seemed to imply prefixing all or none based on one random roll.
        # If prefixing is selective per tag, this needs adjustment.
        # Assuming if prob met, all tags in this group get prefix.
        if random.random() < add_fixed_prefix_prob:
            return [f"{prefix}:{tag}" for tag in tags]
        return tags

    def add_underline_to_tags(tags_list):
        updated_tags = []
        for tag in tags_list:
            if ' ' in tag and random.random() < add_underline_prob:
                updated_tags.append(tag.replace(' ', '_'))
            else:
                updated_tags.append(tag)
        return updated_tags

    # Process fixed tags
    fixed_tags = []
    drop_artist = random.random() < drop_artist_prob

    if 'final_artist_tag' in extras and extras['final_artist_tag'] and not drop_artist:
        fixed_tags.extend(add_prefix_to_tags(extras['final_artist_tag'], "artist"))
    if 'final_character_tag' in extras and extras['final_character_tag']:
        fixed_tags.extend(add_prefix_to_tags(extras['final_character_tag'], "character"))
    if 'final_copyright_tag' in extras and extras['final_copyright_tag']:
        copyright_tags_str = extras['final_copyright_tag']
        # Filter out "original" from copyright tags before prefixing
        raw_copyright_tags = [tag.strip() for tag in copyright_tags_str.split(',') if tag.strip()]
        filtered_copyright_tags = [tag for tag in raw_copyright_tags if "original" not in tag.lower()] # case-insensitive "original" check
        if filtered_copyright_tags:
            fixed_tags.extend(add_prefix_to_tags(",".join(filtered_copyright_tags), "copyright"))
    
    # User's addition: final_features_tag_prefix moved to fixed_tags
    if 'final_features_tag_prefix' in extras and extras['final_features_tag_prefix']:
        fixed_tags.append(extras['final_features_tag_prefix'].strip())

    fixed_tags = add_underline_to_tags(fixed_tags)

    # --- Helper function to get detailed flex tags ---
    def _get_detailed_flex_tags():
        # Uses variables from the outer scope: extras, shuffle_caption, dropout_rate,
        # add_underline_to_tags, add_prefix_to_tags, and the global dropout_tags.
        detailed_tags_list = []
        
        if 'final_features_tag' in extras and extras['final_features_tag']:
            detailed_tags_list.extend([t.strip() for t in extras['final_features_tag'].split(",") if t.strip()])
        
        if 'final_rating_tag' in extras and extras['final_rating_tag']:
            rating_keywords = {"explicit", "sensitive", "nsfw", "general"} # Use a set for faster lookups
            raw_rating_elements = [tag.strip().lower() for tag in extras['final_rating_tag'].split(',') if tag.strip()]
            
            ratings_to_prefix = [elem for elem in raw_rating_elements if elem in rating_keywords]
            other_rating_elements = [elem for elem in raw_rating_elements if elem not in rating_keywords]

            if ratings_to_prefix:
                 detailed_tags_list.extend(add_prefix_to_tags(", ".join(ratings_to_prefix), "rating"))
            detailed_tags_list.extend(other_rating_elements)

        if 'aes_rating' in extras and extras['aes_rating']:
            detailed_tags_list.append(extras['aes_rating'])
        
        if 'additional_tags' in extras and extras['additional_tags']:
            tags_from_additional = [t.strip() for t in extras['additional_tags'].split(',') if t.strip()]
            detailed_tags_list.extend(tags_from_additional)

        # Use 'year_tag' consistent with user's latest change
        if 'year_tag' in extras and extras['year_tag']:
            detailed_tags_list.append(extras['year_tag'])
        if 'year_tag_specific' in extras and extras['year_tag_specific']:
            detailed_tags_list.append(extras['year_tag_specific'])

        if shuffle_caption:
            random.shuffle(detailed_tags_list)
        
        detailed_tags_list = dropout_tags(detailed_tags_list, dropout_rate) # dropout_tags is global
        detailed_tags_list = add_underline_to_tags(detailed_tags_list)
        
        return detailed_tags_list
    # --- End of helper function ---

    flex_tags = []
    caption_nl = random.random() < caption_nl_prob
    style_mix = random.random() < style_mix_prob

    if caption_nl:
        # Try to use summaries
        summary_text_to_use = None
        has_regular_summary = 'regular_summary' in extras and extras['regular_summary'] and extras['regular_summary'].strip()
        has_brief_summary = 'brief_summary' in extras and extras['brief_summary'] and extras['brief_summary'].strip()
        
        if has_regular_summary and has_brief_summary:
            summary_text_to_use = extras['brief_summary'].strip() if style_mix else extras['regular_summary'].strip()
        elif has_regular_summary:
            summary_text_to_use = extras['regular_summary'].strip()
        elif has_brief_summary:
            summary_text_to_use = extras['brief_summary'].strip()
        
        if summary_text_to_use: # If a non-empty summary was found
            flex_tags.append(summary_text_to_use)
            # Summaries are typically single strings and not further processed with dropout/underline here.
    
    # If flex_tags is still empty (i.e., caption_nl was false, or it was true but no valid summary was found),
    # then populate with detailed tags.
    if not flex_tags:
        flex_tags.extend(_get_detailed_flex_tags())

    # Apply drop_all probabilities
    if random.random() < drop_all_fixed_prob:
        fixed_tags = []
    if random.random() < drop_all_flex_prob:
        flex_tags = []
    
    if shuffle_caption:
        random.shuffle(fixed_tags)
        # Flex tags from _get_detailed_flex_tags are already shuffled if shuffle_caption is true.
        # Flex tags from summaries (single strings) are not shuffled.

    all_tags = fixed_tags + flex_tags
    # Filter out any potentially empty strings before joining
    all_tags = [tag for tag in all_tags if tag and tag.strip()] 
    new_prompt = ", ".join(all_tags)

    return Entry(
        is_latent=data_entry.is_latent,
        pixel=data_entry.pixel,
        prompt=new_prompt,
        original_size=data_entry.original_size,
        cropped_size=data_entry.cropped_size,
        dhdw=data_entry.dhdw,
        extras=extras
    )