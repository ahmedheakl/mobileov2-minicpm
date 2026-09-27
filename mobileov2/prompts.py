"""Prompt templates and token-id construction -- byte-identical to what SFT trained on.

If any of this drifts the model still produces an image, just a worse one, with no error.
That is why the templates are copied verbatim from the training code rather than rewritten:
  GENERATION_SYSTEM_PROMPT / build_ids       <- mobileo/train/sample_callback.py
  EDITING_SYSTEM_PROMPT / preprocess_editing <- mobileo/train/train_minicpm.py
  tokenizer_image_token                      <- mobileo/mm_utils.py
  process_und_image                          <- mobileo/minicpm_image.py
"""
import torch

IMAGE_TOKEN_INDEX = -200
IGNORE_INDEX = -100

GENERATION_SYSTEM_PROMPT = (
    "Describe the image by detailing the color, quantity, text, shape, size, texture, "
    "spatial relationships of the objects and background: "
)

EDITING_SYSTEM_PROMPT = (
    "Describe the key features of the input image (color, shape, size, texture, objects, background), "
    "then explain how the user's text instruction should alter or modify the image. "
    "Generate a new image that meets the user's requirements while maintaining consistency "
    "with the original input where appropriate."
)

EDIT_USER_PROMPT = "Please edit the provided image according to the following description: {}"

# MUST match --und_max_slices of the checkpoint. MiniCPM slices the source image into this
# many 448px tiles plus an overview, each through the 27-layer ViT. Inference must use the
# SAME value as training or the model is conditioned differently than it was taught.
UND_MAX_SLICES = 9


def build_ids(tokenizer, prompt, max_len=256):
    """Token ids for one generation prompt."""
    text = (
        f"<|im_start|>system\n{GENERATION_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nPlease generate image based on the following caption: {prompt}"
        f"<|im_end|>\n<|im_start|>assistant\n<|im_end|>\n"
    )
    return tokenizer.encode(text, add_special_tokens=False)[:max_len]


def tokenizer_image_token(
    prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None
):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split("<image>")]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if (
        len(prompt_chunks) > 0
        and len(prompt_chunks[0]) > 0
        and prompt_chunks[0][0] == tokenizer.bos_token_id
    ):
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids


def preprocess_qwen_editing(sources, tokenizer, system_message="You are a helpful assistant."):
    """Tokenize editing conversations, replacing <image> with IMAGE_TOKEN_INDEX."""
    roles = {"human": "user", "gpt": "assistant"}
    ignore_index = IGNORE_INDEX
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    eol_id = 198  # newline token

    input_ids_list = []
    labels_list = []

    for i, source in enumerate(sources):
        if not isinstance(source, list):
            continue

        tokens = []
        labels = []

        # System message
        sys_text = f'<|im_start|>system\n{system_message}'
        sys_ids = tokenizer.encode(sys_text, add_special_tokens=False)
        tokens.extend(sys_ids)
        tokens.append(im_end_id)
        tokens.append(eol_id)
        labels.extend([ignore_index] * (len(sys_ids) + 2))

        for msg in source:
            role = roles.get(msg["from"], msg["from"])
            text = f'<|im_start|>{role}\n{msg["value"]}'

            if "<image>" in text:
                # Handle <image> token: split and replace with IMAGE_TOKEN_INDEX
                msg_ids = tokenizer_image_token(text, tokenizer)
            else:
                msg_ids = tokenizer.encode(text, add_special_tokens=False)

            tokens.extend(msg_ids)
            if msg["from"] == "human":
                labels.extend([ignore_index] * len(msg_ids))
            else:
                labels.extend(msg_ids)

            if msg["from"] != "human":
                tokens.append(im_end_id)
                tokens.append(eol_id)
                labels.append(im_end_id)
                labels.append(eol_id)

        input_ids_list.append(torch.tensor(tokens, dtype=torch.long))
        labels_list.append(torch.tensor(labels, dtype=torch.long))

    input_ids = torch.stack(input_ids_list, dim=0)
    labels = torch.stack(labels_list, dim=0)
    return dict(input_ids=input_ids, labels=labels)


def build_edit_ids(tokenizer, instruction):
    """Token ids for one edit, identical to what the editing SFT dataset builds."""
    conv = [{"from": "human", "value": f"<image>\n{EDIT_USER_PROMPT.format(instruction)}"},
            {"from": "gpt", "value": ""}]
    ids = preprocess_qwen_editing([conv], tokenizer, system_message=EDITING_SYSTEM_PROMPT)["input_ids"][0]
    n_img = int((ids == IMAGE_TOKEN_INDEX).sum())
    assert n_img == 1, f"{n_img} image tokens for one source image; template drifted"
    return ids


def process_und_image(
    pil_image,
    image_processor,
    max_slice_nums: int = 9,
    downsample_mode: str = "16x",
):
    """Run the MiniCPM image processor on one PIL image.

    Args:
        pil_image: a single PIL.Image (RGB conversion is handled by the processor).
        image_processor: a ``MiniCPMV4_6ImageProcessor`` (or its Pil variant).
        max_slice_nums: cap on high-res slices; match what the *processor* used to
            build text placeholders so token counts stay aligned.
        downsample_mode: "16x" (default) or "4x". Must match the value passed to the
            model's ``encode_images`` / ``config.downsample_mode``.
    """
    out = image_processor(
        pil_image,
        return_tensors="pt",
        max_slice_nums=max_slice_nums,
        downsample_mode=downsample_mode,
    )

    pixel_values = out["pixel_values"]            # [1, C, patch_size, L]
    target_sizes = out["target_sizes"]            # [num_patches, 2]  (int32)
    num_patches = int(out["num_patches_per_image"][0])

    return {
        "und_pixel_values": pixel_values[0].contiguous(),   # [C, patch_size, L]
        "und_target_sizes": target_sizes,                   # [num_patches, 2]
        "und_num_patches": num_patches,                     # int
    }


def embed_edit_row(model, ids, und, device):
    """Splice the source image's features into the token sequence, for ONE row.

    One row at a time means no padding at all, so this is the unpadded reference that
    every padding convention reduces to at real positions.
    """
    feats = model.encode_images(und["und_pixel_values"].unsqueeze(0).to(device),
                                und["und_target_sizes"].to(device),
                                [und["und_num_patches"]])
    pos = (ids[0] == IMAGE_TOKEN_INDEX).nonzero().flatten()
    assert len(pos) == len(feats) == 1, (
        f"{len(pos)} image tokens and {len(feats)} image features; one of each expected")
    embed = model.get_model().get_input_embeddings()
    i = int(pos[0])
    parts = [embed(ids[0, :i]), feats[0].to(embed.weight.dtype), embed(ids[0, i + 1:])]
    out = torch.cat(parts, dim=0).unsqueeze(0)
    return out, torch.ones(out.shape[:2], dtype=torch.bool, device=device)
