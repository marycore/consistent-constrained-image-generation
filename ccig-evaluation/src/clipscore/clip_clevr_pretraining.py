import os
import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from torch.utils.data import DataLoader
import torch
from transformers import CLIPModel
from tqdm import tqdm
from torch.utils.data import Subset

from transformers import CLIPTokenizer
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from collections import defaultdict
import random

device = "cuda" if torch.cuda.is_available() else "cpu"
print('Device:', device)
model_name = "zer0int/LongCLIP-GmP-ViT-L-14" #openai/clip-vit-base-patch32

tokenizer = AutoTokenizer.from_pretrained(model_name)

print('Max len:', tokenizer.model_max_length)#77 for normal clip, 248 for long clip

#json_file = '/users/sbsh670/data/ccig_finetune/clip_pretraining_clevr_train.json'
image_dir = '/users/sbsh670/data/clevr/CLEVR_v1.0/image_generative_model_training/images/train'

clip_train = '/users/sbsh670/data/ccig_finetune/clip_train_clevr.json'
clip_val = '/users/sbsh670/data/ccig_finetune/clip_val_clevr.json'

save_path= '/users/sbsh670/data/ccig-models/clip-clevr'
save_model = save_path+'/longclip_clevr_best'

'''
with open(json_file) as f:
    data = json.load(f)

# group all prompts by image
image_to_samples = defaultdict(list)

for sample in data:
    image_to_samples[sample["image_id"]].append(sample)

image_ids = list(image_to_samples.keys())

random.seed(42)
random.shuffle(image_ids)

split_ratio = 0.9

split_idx = int(len(image_ids) * split_ratio)

train_ids = set(image_ids[:split_idx])
val_ids = set(image_ids[split_idx:])

train_samples = []
val_samples = []

for image_id, samples in image_to_samples.items():

    if image_id in train_ids:
        train_samples.extend(samples)
    else:
        val_samples.extend(samples)


print("Train images:", len(train_ids))
print("Val images:", len(val_ids))

print("Train samples:", len(train_samples))
print("Val samples:", len(val_samples))

with open(clip_train,"w") as f:
    json.dump(train_samples,f)

with open(clip_val,"w") as f:
    json.dump(val_samples,f)


with open(json_file) as f:
    data = json.load(f)

lengths = []

for sample in tqdm(data):
    tokens = tokenizer(
        sample["prompt"],
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]

    lengths.append(len(tokens))

print("Maximum tokens:", max(lengths)) #94
print("Average tokens:", sum(lengths)/len(lengths)) #36.55
'''

def evaluate(model, loader, device):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in loader:

            batch = {
                k:v.to(device)
                for k,v in batch.items()
            }

            outputs = model(
                **batch,
                return_loss=True
            )

            total_loss += outputs.loss.item()


    return total_loss / len(loader)

class CLEVRClipDataset(Dataset):
    def __init__(self, json_file, image_dir):

        with open(json_file, "r") as f:
            self.samples = json.load(f)

        self.image_dir = image_dir

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        sample = self.samples[idx]

        image_path = os.path.join(
            self.image_dir,
            sample["image_filename"]
        )

        image = Image.open(image_path).convert("RGB")

        text = sample["prompt"]

        return image, text, sample["image_id"], sample["n_objects"]

from collections import defaultdict
import random
from torch.utils.data import BatchSampler, Sampler


class NObjectsBatchSampler(Sampler):

    def __init__(self, dataset, batch_size):

        self.batch_size = batch_size

        self.groups = defaultdict(list)

        for idx, sample in enumerate(dataset.samples):
            self.groups[sample["n_objects"]].append(idx)

        self.n_values = list(self.groups.keys())


    def __iter__(self):

        pools = {
            k: random.sample(v, len(v))
            for k, v in self.groups.items()
        }

        while True:

            batch = []

            # cycle through different n_objects
            while len(batch) < self.batch_size:

                random.shuffle(self.n_values)

                added = False

                for n in self.n_values:

                    if pools[n]:

                        batch.append(
                            pools[n].pop()
                        )

                        added = True

                    if len(batch) == self.batch_size:
                        break

                if not added:
                    break

            if len(batch) == 0:
                break

            yield batch


    def __len__(self):

        total = sum(
            len(v)
            for v in self.groups.values()
        )

        return total // self.batch_size

def collate_fn(batch):

    images = [b[0] for b in batch]
    texts = [b[1] for b in batch]
    image_ids = [b[2] for b in batch]
    n_objects = [b[3] for b in batch]
    
    return processor(
        images=images,
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=248
    )

def main():
    train_dataset = CLEVRClipDataset(
    json_file=clip_train,
    image_dir=image_dir
    )

    val_dataset = CLEVRClipDataset(
    json_file=clip_val,
    image_dir=image_dir
    )
    val_indices = random.sample(
    range(len(val_dataset)),
    10000
    )
    val_subset = Subset(
    val_dataset,
    val_indices
    )

    sampler = NObjectsBatchSampler(train_dataset, 8)

    train_loader = DataLoader(
    train_dataset,
    batch_sampler=sampler,
    collate_fn=collate_fn
    #num_workers=4,
    #pin_memory=True,
    #persistent_workers=True
    )

    val_loader = DataLoader(
    val_subset,
    batch_size=8,
    shuffle=False,
    collate_fn=collate_fn
    #num_workers=4,
    #pin_memory=True,
    #persistent_workers=True

    )

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-6,
    weight_decay=0.01
    )

    epochs = 1
    step = 0
    validate_every = 50000

    best_val_loss = float("inf")

    for epoch in range(epochs):

        model.train()
        running_loss = 0
        for batch in tqdm(train_loader):

            batch = {
            k:v.to(device)
            for k,v in batch.items()
            }


            outputs = model(
            **batch,
            return_loss=True
            )

            loss = outputs.loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            step += 1
            if step % validate_every == 0:

                train_loss = running_loss / validate_every

                val_loss = evaluate(
                model,
                val_loader,
                device
                )


                print(
                f"""
                Step: {step}
                Train loss: {train_loss:.4f}
                Validation loss: {val_loss:.4f}
                """
                )


                if val_loss < best_val_loss:

                    best_val_loss = val_loss

                    model.save_pretrained(
                    "/users/sbsh670/data/ccig-models/clip-clevr/longclip_clevr_best"
                    )

                    processor.save_pretrained(
                    "/users/sbsh670/data/ccig-models/clip-clevr/longclip_clevr_best"
                    )

                    print("Saved best model")


                running_loss = 0
                model.train()


if __name__ == "__main__":
    main()



