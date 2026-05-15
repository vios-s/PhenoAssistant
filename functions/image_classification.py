import os
import glob
import json
import copy
import random
import requests
from typing import Annotated, Literal, List, Dict, Optional, Tuple, Any
import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import torchvision.transforms as T
# from torchvision.transforms import (
#     CenterCrop,
#     Compose,
#     Normalize,
#     RandomHorizontalFlip,
#     RandomResizedCrop,
#     Resize,
#     ToTensor,
# )
from albumentations.pytorch import ToTensorV2
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, 
    Transpose, ShiftScaleRotate, HueSaturationValue,
    RandomResizedCrop, RandomBrightnessContrast, Compose, Normalize, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize)

import datasets
from datasets import load_dataset
import accelerate
import evaluate
import albumentations as A

import transformers
from transformers import (
    AutoImageProcessor, 
    AutoModelForImageClassification, 
    AutoConfig,
    AutoModelForImageSegmentation, 
    AutoModelForObjectDetection,
    Mask2FormerImageProcessor, 
    Mask2FormerModel, 
    Mask2FormerForUniversalSegmentation, 
    Mask2FormerConfig,
    PretrainedConfig,
    TrainingArguments, 
    Trainer,
)

import peft
from peft import PeftConfig, PeftModel, LoraConfig, get_peft_model
from huggingface_hub import login, HfApi, model_info, dataset_info, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# My functions
from .generic_tools import (
    set_env_vars, 
    set_random_seed, 
    print_trainable_parameters, 
    handle_grayscale_image, 
    download_hffile, 
    load_images, 
    update_model_zoo,
)

def compute_metrics(eval_pred):
    """Computes accuracy on a batch of predictions"""
    metric = evaluate.load("accuracy")
    predictions = np.argmax(eval_pred.predictions, axis=1)
    return metric.compute(predictions=predictions, references=eval_pred.label_ids)

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def finetune_image_classification(
        # dataset_name: Annotated[str, "Dataset name for finetuning."] = "winter-wheat_fertiliser-identification_dnd-ww2020",
        dataset_name: Annotated[str, "Dataset name for finetuning."],
        ft_method: Annotated[Literal['lora', 'fullft'], "Finetuning method ('lora' or 'fullft')."] = 'lora',
        train_bs: Annotated[int, "Training batch size."] = 32,
        val_bs: Annotated[int, "Validation batch size."] = 1,
        img_w: Annotated[int, "Image width for training."] = 512,
        img_h: Annotated[int, "Image height for training."] = 512,
        lr: Annotated[float, "Learning rate for training."] = 1e-4,
        epochs: Annotated[int, "Number of training epochs."] = 1,
        ) -> str:
    '''
    Finetune a vision model for image classification on a custom dataset.
    Returns:
    - str: Best validation accuracy achieved during training.
    '''
    # move some hyperparameters here
    pretrained_model = "facebook/dinov2-base"
    pretrained_model_short = "dino2b"
    auto_find_batch_size = False
    img_size = (img_w, img_h)
    
    # environment variables
    # set_env_vars('../.env.yaml')
    # set_random_seed()
    # login(token=os.getenv('HF_TOKEN')) # login to Hugging Face
    user_name = os.getenv('HF_USER')
    repo_id = f"{user_name}/{dataset_name}"
    model_name = '_'.join((dataset_name, pretrained_model_short, ft_method))
    # model_name = '_'.join(('image-classification', ft_method, pretrained_model.split('/')[-1], dataset_name)) # output_dir
    # check if model name exists
    try:
        model_info(f'{user_name}/{model_name}')  # Try fetching model info
        return f"Model '{model_name}' exists. You can proceed to infer the model."
    except HfHubHTTPError:
        print(f"Model '{model_name}' does not exist. Proceed to train the model.")

    lora_modules_to_save = ["classifier"]

    dataset = load_dataset(repo_id,)['train'] # for dnd, test does not have labels    
    label2id, id2label = dict(), dict()
    for i, label in enumerate(dataset.features['label'].names):
        label2id[label] = i
        id2label[i] = label

    image_processor = AutoImageProcessor.from_pretrained(pretrained_model,
                                                         do_resize = False,
                                                         do_rescale = False,
                                                         do_normalize = False,
                                                         do_center_crop = False,
                                                         do_convert_rgb = False,
                                                         size = img_size,
                                                        )
    image_processor.label2id = label2id
    image_processor.id2label = id2label

    # if input is RGB/RGBA/P/L, ToTensor will do scaling
    image_processor.image_mean = [0.485, 0.456, 0.406]
    image_processor.image_std = [0.229, 0.224, 0.225]
    
    # # torch version
    # normalize = T.Normalize(mean=image_processor.image_mean, 
    #                       std=image_processor.image_std) # for google vit they use 0.5 all the time
    # train_transforms = Compose([
    #     T.Resize((img_size[0], img_size[1])),
    #     # T.RandomResizedCrop
    #     T.RandomHorizontalFlip(),
    #     T.ToTensor(),
    #     normalize,
    # ])
    # val_transforms = Compose([
    #     T.Resize((img_size[0], img_size[1])),
    #     T.ToTensor(),
    #     normalize,
    # ])
    # def preprocess_train(examples):
    #     """Apply transforms across a batch."""
    #     examples["pixel_values"] = [train_transforms(image.convert("RGB")) for image in examples["image"]]
    #     return examples
    # def preprocess_val(examples):
    #     """Apply transforms across a batch."""
    #     examples["pixel_values"] = [val_transforms(image.convert("RGB")) for image in examples["image"]]
    #     return examples

    # albumentations version
    def get_train_transforms():
        return Compose([
                RandomResizedCrop(img_size[0], img_size[1]),
                Transpose(p=0.5), # 90 degree rotation + vertical flip
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5), # Randomly apply affine transforms: translate, scale and rotate the input
                # HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                # RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                Normalize(mean=image_processor.image_mean, std=image_processor.image_std, max_pixel_value=255.0, p=1.0),
                CoarseDropout(p=0.5),
                ToTensorV2(p=1.0),
            ], p=1.)
    train_transforms = get_train_transforms()
    def get_val_transforms():
        return Compose([
                # HorizontalFlip(p=0.5),
                # VerticalFlip(p=0.5),
                # CenterCrop(config['img_size'], config['img_size'], p=1.),
                Resize(img_size[0], img_size[1]),
                Normalize(mean=image_processor.image_mean, std=image_processor.image_std, max_pixel_value=255.0, p=1.0),
                ToTensorV2(p=1.0),
            ], p=1.)
    val_transforms = get_val_transforms()
    def preprocess_train(examples):
        """Apply transforms across a batch."""
        examples["pixel_values"] = [train_transforms(image=np.array(image.convert("RGB")))['image'] for image in examples["image"]]
        return examples
    def preprocess_val(examples):
        """Apply transforms across a batch."""
        examples["pixel_values"] = [val_transforms(image=np.array(image.convert("RGB")))['image'] for image in examples["image"]]
        return examples

    splits = dataset.train_test_split(test_size=0.1) # in this dataset the dataset["test"] does not contaion labels
    train_dataset = splits["train"]
    val_dataset = splits["test"]
    train_dataset.set_transform(preprocess_train)
    val_dataset.set_transform(preprocess_val)

    model = AutoModelForImageClassification.from_pretrained(
        pretrained_model,
        label2id=label2id, # need this to tell the model how many classes to predict
        id2label=id2label,
        ignore_mismatched_sizes=True, # warning on mismatched weights
    )
    print_trainable_parameters(model)

    # model to lora_model
    if ft_method == 'lora': # else full finetune
        config = LoraConfig(
            r=8,
            lora_alpha=8,
            target_modules=["query", "value"],
            # layers_to_transform=["xxx"], # where to apply lora
            lora_dropout=0.1,
            bias="lora_only",
            use_rslora=True,
            modules_to_save=lora_modules_to_save,
        )
        model = get_peft_model(model, config)
        print_trainable_parameters(model)

    # training
    training_args = TrainingArguments(
        learning_rate = lr,
        num_train_epochs = epochs,
        auto_find_batch_size = auto_find_batch_size,
        per_device_train_batch_size = train_bs,
        per_device_eval_batch_size = val_bs,
        save_total_limit = 1, # total number of models saved (not incl. best model)
        label_names = ["labels"],
        metric_for_best_model = "accuracy", # 'loss' by default
        greater_is_better = True,
        # lr_scheduler_type = "constant", # default linear
        # batch_eval_metrics = True,
        # eval_do_concat_batches = False, # default is True

        hub_token = os.getenv('HF_TOKEN'),
        push_to_hub = True,
        hub_private_repo = True,
        output_dir = model_name,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        logging_steps = 5,
        fp16 = False,
        dataloader_num_workers = 0,
        load_best_model_at_end = True,
        disable_tqdm=True,

        ## other useful args
        remove_unused_columns = False, # default is True
        # gradient_accumulation_steps=4,
        # eval_on_start = True, # make sure the evaluation works before training
        # eval_steps # defualt to logging_steps, but only when xxx = "steps" works
        # fp16_full_eval=True, # for evaluation use fp16 as well
        # dataloader_prefetch_factor = 2,
        # save_safetensors # default is True
        # seed # default is 42
        # include_inputs_for_metrics
        # optim # defaults to "adamw_torch"
        # resume_from_checkpoint # local path to checkpoint
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        tokenizer=image_processor,
    )
    trainer.train()
    
    # evaluate_results = trainer.evaluate() # this is the best model if load_best_model_at_end = True
    # print(evaluate_results)
    # best_model_checkpoint = trainer.state.best_model_checkpoint
    metric_for_best_model = trainer.args.metric_for_best_model
    best_metric = trainer.state.best_metric
    
    # Write model card and push to hub
    kwargs = {
        "finetuned_from": pretrained_model,
        "dataset": dataset_name,
        "tags": ["image-classification", "vision"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    # update model_zoo
    update_model_zoo('image-classification',
                     f'{user_name}/{model_name}')

    # read model card
    model_card = hf_hub_download(f'{user_name}/{model_name}', 'README.md')
    with open(model_card, "r") as f:
        detailed_model_info = f.read()

    # return trainer
    return f"Best validation {metric_for_best_model} is {best_metric}.\n The trained model is saved as {user_name}/{model_name} in model zoo.\nFull details of the trained model can be found in the model card:\n{detailed_model_info}."

def infer_image_classification(
        image_urls: Annotated[Optional[List[str]], "List of image paths"] = None,
        file_path: Annotated[Optional[str], "Path to CSV/JSON file with a 'file_name' column/key containing image URLs"] = None,
        checkpoint: Annotated[str, "Classification model checkpoint identifier."] = None,
        batch_size: Annotated[int, "Number of images to process per batch."] = 1,
        device: Annotated[str, "Device to use for inference ('cuda' or 'cpu')."] = "cuda",
        output_dir: Annotated[str, "Directory path to save results."] = "./results") -> str:
    """
    Perform image classification on a list of images using a specified deep learning model.
    Please provide either image_urls or file_path as input.

    Returns:
    - str: Path to a CSV file containing image classification results.
    """
    # raise errors
    if (image_urls and file_path) or (not image_urls and not file_path):
        return "Error: Provide either 'image_urls' or 'file_path', but not both."

    if file_path:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            if 'file_name' not in df.columns:
                return "Error: CSV file must contain a 'file_name' column."
            image_urls = df['file_name'].tolist()
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'file_name' not in data:
                return "Error: JSON file must contain a 'file_name' key."
            image_urls = data['file_name']
        else:
            return "Error: Unsupported file format. Use CSV or JSON."

    if not image_urls:
        return "Error: No image URLs found."
    
    # model_zoo = build_model_zoo()
    # if checkpoint not in model_zoo:
    #     # raise ValueError(f"Model {checkpoint} not found in the model zoo.")
    #     return {"Error": f"Model {checkpoint} not found in the model zoo."}
    # raw_images = load_images(image_urls) # return a list of PIL.RGB images
    
    dataset_name = image_urls[0].split('/')[-2]
    # save_dir = os.path.join(output_dir, dataset_name)
    save_dir = output_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "img_classification_results.csv")

    # image processor
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)
    image_processor.do_resize = True
    image_processor.do_normalize = True
    image_processor.do_rescale = True # 255 -> 1

    # model
    if 'lora' in checkpoint:
        config = PeftConfig.from_pretrained(checkpoint)
        pretrained_model = AutoModelForImageClassification.from_pretrained(
            config.base_model_name_or_path,
            label2id=image_processor.label2id,
            id2label=image_processor.id2label,
            ignore_mismatched_sizes=True,
        )
        model = PeftModel.from_pretrained(pretrained_model, 
                                          checkpoint, 
                                        #   device_map=device,
                                          )
    else:
        model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            label2id=image_processor.label2id,
            id2label=image_processor.id2label,
            ignore_mismatched_sizes=False,
            # device_map=device,
        )

    # inference
    model.to(device)
    model.eval()
    # batch inference in case of large number of images
    all_images = []
    class_id_list = []
    class_score_list = []
    for i in range(0, len(image_urls), batch_size):
        batch_urls = image_urls[i:i+batch_size]
        raw_images = load_images(batch_urls) # return a list of PIL.RGB images
        inputs = image_processor(images=raw_images, return_tensors="pt").to(device)
        with torch.no_grad():
            batch_outputs = model(**inputs)
        batch_logits = batch_outputs.logits
        # class_id_list = logits.argmax(-1).tolist()
        batch_class_score = (F.softmax(batch_logits, dim=-1)).max(-1)
        
        all_images.extend(raw_images)
        class_id_list.extend(batch_class_score.indices.tolist())
        class_score_list.extend(batch_class_score.values.tolist())
    if isinstance(list(image_processor.id2label.keys())[0], str):
        class_label_list = [image_processor.id2label[str(class_id)] for class_id in class_id_list]
    else:
        class_label_list = [image_processor.id2label[class_id] for class_id in class_id_list]

    # Build the result dictionary
    result_dict = {
        "file_name": image_urls,
        "class_ids": class_id_list,
        "class_labels": class_label_list,
        "class_scores": class_score_list
    }

    df = pd.DataFrame(result_dict)
    df.to_csv(save_path, index=False)

    return f"The image classification results are saved at {save_path}."

# example usage
if __name__ == "__main__":
    set_env_vars(env_file='../.env.yaml')
    finetune_image_classification(dataset_name = "winter-wheat_fertiliser-identification_dnd-ww2020",
                                #   ft_method = 'fullft', # support lora and fullft
                                  ft_method = 'lora',
                                  pretrained_model = "facebook/dinov2-base", # tiny/small/large
                                  img_size = (512, 512),
                                  auto_find_batch_size = False,
                                  train_bs = 32,
                                  val_bs = 1,
                                  lr = 1e-4,
                                  epochs = 100,
                                  )

# ============================================================
# CUSTOM DINOv2 FULL FINE-TUNE INFERENCE
# Contributor: Narendran (IIT Bombay, Dr. Soumyashree Kar)
# Models: Rice / Wheat / Maize nutrient deficiency classifiers
# ============================================================



# ============================================================
# CUSTOM DINOv2 FULL FINE-TUNE INFERENCE
# Contributor: Narendren S V (IIT Bombay, Dr. Soumyashree Kar)
# Models: Rice / Wheat / Maize nutrient deficiency classifiers
# ============================================================

class DINOClassifier(torch.nn.Module):
    """
    DINOv2 ViT-S/14 backbone with linear classification head.
    Used for full fine-tune nutrient deficiency classification.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", verbose=False
        )
        self.classifier = torch.nn.Linear(
            self.backbone.embed_dim, num_classes
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))


def infer_image_classification_dinov2(
        image_urls: Annotated[Optional[List[str]], "List of image paths"] = None,
        file_path: Annotated[Optional[str], "Path to CSV/JSON file with a 'file_name' column/key containing image URLs"] = None,
        checkpoint: Annotated[str, "HuggingFace repo id for DINOv2 fullft classifier (e.g. Naren1704/rice_nutrient-deficiency_rgbdataset_dinov2_fullft)."] = None,
        batch_size: Annotated[int, "Number of images to process per batch."] = 16,
        device: Annotated[str, "Device to use for inference ('cuda' or 'cpu')."] = "cpu",
        output_dir: Annotated[str, "Directory path to save results."] = "./results") -> str:
    """
    Perform nutrient deficiency classification using DINOv2 ViT-S/14
    full fine-tuned models for rice (4-class), wheat (5-class),
    and maize (6-class) datasets.

    Contributed by Narendren S V (IIT Bombay internship,
    Dr. Soumyashree Kar) as part of PhenoAssistant vision model zoo.

    Please provide either image_urls or file_path as input.

    Returns:
    - str: Path to a CSV file containing classification results.
    """
    if (image_urls and file_path) or (not image_urls and not file_path):
        return "Error: Provide either 'image_urls' or 'file_path', not both."

    if file_path:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
            if 'file_name' not in df.columns:
                return "Error: CSV file must contain a 'file_name' column."
            image_urls = df['file_name'].tolist()
        elif file_path.endswith(".json"):
            with open(file_path, 'r') as f:
                data = json.load(f)
            if 'file_name' not in data:
                return "Error: JSON file must contain a 'file_name' key."
            image_urls = data['file_name']
        else:
            return "Error: Unsupported file format. Use CSV or JSON."

    if not image_urls:
        return "Error: No image URLs found."

    try:
        config_path = hf_hub_download(repo_id=checkpoint, filename="config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
    except Exception as e:
        return f"Error loading config from {checkpoint}: {str(e)}"

    label2id    = config["label2id"]
    id2label    = {int(k): v for k, v in config["id2label"].items()}
    num_classes = config["num_classes"]

    model_filename = checkpoint.split("/")[-1] + ".pth"
    try:
        weights_path = hf_hub_download(
            repo_id=checkpoint, filename=model_filename
        )
    except Exception as e:
        return f"Error loading weights from {checkpoint}: {str(e)}"

    model = DINOClassifier(num_classes=num_classes)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "img_classification_results.csv")

    class_id_list    = []
    class_label_list = []
    class_score_list = []

    for i in range(0, len(image_urls), batch_size):
        batch_urls    = image_urls[i:i + batch_size]
        raw_images    = load_images(batch_urls)
        batch_tensors = torch.stack(
            [val_tf(img) for img in raw_images]
        ).to(device)

        with torch.no_grad():
            logits = model(batch_tensors)

        probs = F.softmax(logits, dim=-1)
        scores, indices = probs.max(dim=-1)

        class_id_list.extend(indices.cpu().tolist())
        class_score_list.extend(scores.cpu().tolist())
        class_label_list.extend(
            [id2label[idx] for idx in indices.cpu().tolist()]
        )

    result_df = pd.DataFrame({
        "file_name":    image_urls,
        "class_ids":    class_id_list,
        "class_labels": class_label_list,
        "class_scores": class_score_list,
    })
    result_df.to_csv(save_path, index=False)

    return (
        f"Nutrient deficiency classification complete.\n"
        f"Model: {checkpoint}\n"
        f"Images processed: {len(image_urls)}\n"
        f"Results saved at: {save_path}"
    )
