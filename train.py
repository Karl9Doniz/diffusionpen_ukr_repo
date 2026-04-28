import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, random_split
import torchvision
from tqdm import tqdm
from torch import optim
import copy
import argparse
import uuid
import time
import json
from diffusers import AutoencoderKL, DDIMScheduler
import random
from unet import UNetModel
import wandb
from torchvision import transforms
from feature_extractor import ImageEncoder
from torch.utils.tensorboard import SummaryWriter
from utils.iam_dataset import IAMDataset
from utils.GNHK_dataset import GNHK_Dataset
from utils.ukr_dataset import UkrDataset, UkrWordDataset
from utils.auxilary_functions import *
from torchvision.utils import save_image
from torch.nn import DataParallel
import torch.nn.functional as F
from transformers import CanineModel, CanineTokenizer
from utils.word_dataset import (
    char_classes as WORD_CHAR_CLASSES,
    index2letter as WORD_INDEX2LETTER,
    letter2index as WORD_LETTER2INDEX,
    tokens as WORD_TOKENS,
)

torch.cuda.empty_cache()
OUTPUT_MAX_LEN = 95 #+ 2  # <GO>+groundtruth+<END>
IMG_WIDTH = 256
IMG_HEIGHT = 64

letter2index = WORD_LETTER2INDEX
index2letter = WORD_INDEX2LETTER
char_classes = WORD_CHAR_CLASSES
tokens = WORD_TOKENS
num_tokens = len(tokens)
c_classes = ''.join(index2letter[i] for i in range(len(index2letter)))
cdict = letter2index
icdict = index2letter

### Borrowed from GANwriting ###
def label_padding(labels, num_tokens):
    new_label_len = []
    ll = [letter2index[i] for i in labels]
    new_label_len.append(len(ll) + 2)
    ll = np.array(ll) + num_tokens
    ll = list(ll)
    #ll = [tokens["GO_TOKEN"]] + ll + [tokens["END_TOKEN"]]
    num = OUTPUT_MAX_LEN - len(ll)
    if not num == 0:
        ll.extend([tokens["PAD_TOKEN"]] * num)  # replace PAD_TOKEN
    return ll


print('num_tokens', num_tokens)
print('num of character classes', char_classes)
vocab_size = char_classes + num_tokens



def setup_logging(args):
    #os.makedirs("models", exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'models'), exist_ok=True)
    if args.save_samples:
        os.makedirs(os.path.join(args.save_path, 'images'), exist_ok=True)

def save_images(images, path, args, **kwargs):
    #print('image', images.shape)
    grid = torchvision.utils.make_grid(images, padding=0, **kwargs)
    if args.latent == True:
        im = torchvision.transforms.ToPILImage()(grid)
        if args.color == False:
            im = im.convert('L')
        else:
            im = im.convert('RGB')
    else:
        ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
        im = Image.fromarray(ndarr)
    if path:
        im.save(path)
    return im

def crop_whitespace_width(img):
    #tensor image to PIL
    original_height = img.height
    img_gray = np.array(img)
    ret, thresholded = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresholded)
    x, y, w, h = cv2.boundingRect(coords)
    #rect = img.crop((x, 0, x + w, original_height))
    rect = img.crop((x, y, x + w, y + h))
    return np.array(rect)


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

class EMA:
    '''
    EMA is used to stabilize the training process of diffusion models by
    computing a moving average of the parameters, which can help to reduce
    the noise in the gradients and improve the performance of the model.
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


def _unwrap_model(model):
    return model.module if isinstance(model, DataParallel) else model


def build_ema_model(model, unet_kwargs, device_ids, device):
    ema_base = UNetModel(**unet_kwargs)
    if isinstance(model, DataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()
    ema_base.load_state_dict(state)
    if device_ids is not None:
        ema_base = DataParallel(ema_base, device_ids=device_ids)
    ema_base = ema_base.to(device)
    ema_base.eval()
    ema_base.requires_grad_(False)
    return ema_base



class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=(64, 256), args=None):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(args.device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.device = args.device


    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sampling_loader(self, model, test_loader, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=1.0, transform=None, character_classes=None, tokenizer=None, text_encoder=None):
        model.eval()
        tensor_list = []

        with torch.no_grad():
            pbar = tqdm(test_loader)
            style_feat = []
            for i, data in enumerate(pbar):
                images = data[0].to(args.device)
                transcr = data[1]
                s_id = data[2].to(args.device)
                style_images = data[3].to(args.device)
                cor_im = data[5].to(args.device)
                img_path = data[4]


                if args.model_name == 'wordstylist':
                    #print('transcr', transcr)
                    batch_word_embeddings = []
                    for trans in transcr:
                        word_embedding = label_padding(trans, num_tokens)
                        #print('word_embedding', word_embedding)
                        word_embedding = np.array(word_embedding, dtype="int64")
                        word_embedding = torch.from_numpy(word_embedding).long()
                        batch_word_embeddings.append(word_embedding)
                    text_features = torch.stack(batch_word_embeddings).to(args.device)
                else:
                    text_features = tokenizer(
                        transcr,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        max_length=args.text_max_len,
                    ).to(args.device)

                img_h, img_w = args.img_size
                reshaped_images = style_images.reshape(-1, 3, img_h, img_w)

                if style_extractor is not None:
                    style_features = style_extractor(reshaped_images).to(args.device)
                else:
                    style_features = None

                if args.latent == True:
                    x = torch.randn((images.size(0), 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)

                else:
                    x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)

                # Pre-compute null text features for CFG (empty string = unconditional)
                null_text_features = None
                if cfg_scale > 1.0 and tokenizer is not None:
                    null_texts = [""] * images.size(0)
                    null_text_features = tokenizer(
                        null_texts,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                        max_length=args.text_max_len,
                    ).to(args.device)

                #scheduler
                noise_scheduler.set_timesteps(50)
                for time in noise_scheduler.timesteps:

                    t_item = time.item()
                    t = (torch.ones(images.size(0)) * t_item).long().to(args.device)

                    with torch.no_grad():
                        noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                        if cfg_scale > 1.0 and null_text_features is not None:
                            noisy_residual_uncond = model(x, t, null_text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                            noisy_residual = noisy_residual_uncond + cfg_scale * (noisy_residual - noisy_residual_uncond)
                        prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                        x = prev_noisy_sample

        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = _unwrap_model(vae).decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()

            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x

    def sampling(self, model, vae, n, x_text, labels, args, style_extractor, noise_scheduler, mix_rate=None, cfg_scale=3, transform=None, character_classes=None, tokenizer=None, text_encoder=None, run_idx=None):
        model.eval()
        tensor_list = []

        with torch.no_grad():
            style_images = None
            text_features = x_text #[x_text]*n
            #print('text features', text_features.shape)
            text_features = tokenizer(
                text_features,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                max_length=args.text_max_len,
            ).to(args.device)
            if args.img_feat == True:
                #pick random image according to specific style
                with open('./writers_dict_train.json', 'r') as f:

                    wr_dict = json.load(f)
                reverse_wr_dict = {v: k for k, v in wr_dict.items()}

                #key = reverse_wr_dict[value]
                with open('./utils/splits_words/iam_train_val.txt', 'r') as f:
                #with open('./utils/splits_words/iam_test.txt', 'r') as f:
                    train_data = f.readlines()
                    train_data = [i.strip().split(',') for i in train_data]
                    style_featur = []
                    for label in labels:
                        #print('label', label)
                        label_index = label.item()

                        matching_lines = [line for line in train_data if line[1] == reverse_wr_dict[label_index] and len(line[2])>3]

                        #pick the first 5 from matching lines

                        if len(matching_lines) >= 5:
                            #five_styles = matching_lines[:5]
                            #pick first line and repeat
                            #five_styles = [matching_lines[0]]*5
                            five_styles = random.sample(matching_lines, 5)
                            #five_styles = matching_lines_style[:5]
                        else:
                            matching_lines = [line for line in train_data if line[1] == reverse_wr_dict[label_index]]
                            #print('matching lines', matching_lines)
                            five_styles = matching_lines_style[:5]
                            five_styles = [matching_lines[0]]*5
                            #five_styles = random.sample(matching_lines, 5)
                        print('five_styles', five_styles)
                        #five_styles = random.sample(matching_lines, 5)

                        cor_image_random = random.sample(matching_lines, 1)
                        #print('cor_image_random', cor_image_random)
                        #five_styles =[['a05/a05-084/a05-084-04-05.png', '000', 'which'], ['a03/a03-073/a03-073-04-04.png', '000', 'stage'], ['a01/a01-077u/a01-077u-02-02.png', '000', 'cables'], ['a05/a05-089/a05-089-00-05.png', '000', 'debate'], ['a05/a05-048/a05-048-00-00.png', '000', 'Long']] #class id 12
                        #five_styles = [['b06/b06-071/b06-071-08-06.png', '128', 'Labour'], ['b06/b06-019/b06-019-05-04.png', '128', 'West'], ['b06/b06-071/b06-071-05-03.png', '128', 'could'], ['c06/c06-027/c06-027-01-01.png', '128', 'advantage'], ['c06/c06-076/c06-076-01-05.png', '128', 'never']] #class id 1

                        interpol = False
                        if interpol == True:
                            label2 = random.randint(0, 339) #random label
                            matching_lines2 = [line for line in train_data if line[1] == reverse_wr_dict[label2] and len(line[2])>3]
                            five_styles = random.sample(matching_lines2, 5)
                        #print('five_styles', five_styles)
                        #cor_image
                        fheight, fwidth = 64, 256
                        root_path = './iam_data/words'
                        cor_im = False
                        if cor_im == True:
                            cor_image = Image.open(os.path.join(root_path, cor_image_random[0][0])).convert('RGB') #['a05/a05-089/a05-089-00-05.png', '000', 'debate']
                            (cor_image_width, cor_image_height) = cor_image.size
                            cor_image = cor_image.resize((int(cor_image_width * 64 / cor_image_height), 64))
                            (cor_image_width, cor_image_height) = cor_image.size

                            if cor_image_width < 256:
                                outImg = ImageOps.pad(cor_image, size=(256, 64), color= "white")#, centering=(0,0)) uncommment to pad right
                                cor_image = outImg

                            else:
                                #reduce image until width is smaller than 256
                                while cor_image_width > 256:
                                    cor_image = image_resize_PIL(cor_image, width=cor_image_width-20)
                                    (cor_image_width, cor_image_height) = cor_image.size
                                cor_image = centered_PIL(cor_image, (64, 256), border_value=255.0)

                            cor_im_tens = transform(cor_image).to(args.device)
                            #print('cor image', cor_im_tens.shape)
                            cor_im_tens = cor_im_tens.unsqueeze(0)
                            cor_images = _unwrap_model(vae).encode(cor_im_tens.to(torch.float32)).latent_dist.sample()
                            cor_images = cor_images * 0.18215

                        st_imgs = []
                        grid_imgs = []
                        for im_idx, random_f in enumerate(five_styles):
                            file_path = os.path.join(root_path, random_f[0])
                            #print('file_path', file_path)

                            try:
                                img_s = Image.open(file_path).convert('RGB')
                            except ValueError:
                                # Handle the exception (e.g., print an error message)
                                print(f"Error loading image from {file_path}")

                                # Find a replacement image that is not corrupted
                                replacement_idx = (im_idx + 1) % 5
                                replacement_f = five_styles[replacement_idx]
                                name = replacement_f[0] #.split(',')[1]
                                replacement_file_path = os.path.join(root_path, name)
                                img_s = Image.open(replacement_file_path).convert('RGB')

                            (img_width, img_height) = img_s.size
                            img_s = img_s.resize((int(img_width * 64 / img_height), 64))
                            (img_width, img_height) = img_s.size

                            if img_width < 256:
                                outImg = ImageOps.pad(img_s, size=(256, 64), color= "white")#, centering=(0,0)) uncommment to pad right
                                img_s = outImg

                            else:
                                #reduce image until width is smaller than 256
                                while img_width > 256:
                                    img_s = image_resize_PIL(img_s, width=img_width-20)
                                    (img_width, img_height) = img_s.size
                                img_s = centered_PIL(img_s, (64, 256), border_value=255.0)
                            #make grid of all 5 images
                            #img_s = img_s.convert('L')
                            transform_tensor = transforms.ToTensor()
                            grid_im = transform_tensor(img_s)
                            grid_imgs += [grid_im]

                            img_tens = transform(img_s).to(args.device)#.unsqueeze(0)
                            st_imgs += [img_tens]
                            #style_features = style_extractor(style_images).to(args.device)
                            #img_tensor = img_tensor.to(args.device)
                        s_imgs = torch.stack(st_imgs).to(args.device)
                        style_images = torch.cat((style_images, s_imgs)) if style_images is not None else s_imgs

                        grid_imgs = torch.stack(grid_imgs).to(args.device)


                        style_images = style_images.to(args.device)


                    #save style images
                    img_h, img_w = args.img_size
                    style_images = style_images.reshape(-1, 3, img_h, img_w)
                    style_features = style_extractor(style_images).to(args.device)
                    # style_features = torch.stack(style_featur, dim=0) #We get [320, 5, 2048]
                    #print('style features', style_features.shape)
                    #style_features = style_features.reshape(n, -1).to(args.device)
            else:
                style_images = None
                style_features = None
            if args.latent == True:
                x = torch.randn((n, 4, self.img_size[0] // 8, self.img_size[1] // 8)).to(args.device)
                if cor_im == True:
                    x_noise = torch.randn(cor_images.shape).to(args.device)

                    timesteps = torch.full((cor_images.shape[0],), 999, device=args.device, dtype=torch.long)

                    noisy_images = noise_scheduler.add_noise(
                        cor_images, x_noise, timesteps
                    )
                    x = noisy_images

            else:
                x = torch.randn((n, 3, self.img_size[0], self.img_size[1])).to(args.device)

            #scheduler
            noise_scheduler.set_timesteps(50)
            for time in noise_scheduler.timesteps:

                t_item = time.item()
                t = (torch.ones(n) * t_item).long().to(args.device)

                with torch.no_grad():
                    noisy_residual = model(x, t, text_features, labels, original_images=style_images, mix_rate=mix_rate, style_extractor=style_features)
                    prev_noisy_sample = noise_scheduler.step(noisy_residual, time, x).prev_sample
                    x = prev_noisy_sample


        model.train()
        if args.latent==True:
            latents = 1 / 0.18215 * x
            image = _unwrap_model(vae).decode(latents).sample

            image = (image / 2 + 0.5).clamp(0, 1)
            image = image.cpu().permute(0, 2, 3, 1).numpy()

            image = torch.from_numpy(image)
            x = image.permute(0, 3, 1, 2)

        else:
            x = (x.clamp(-1, 1) + 1) / 2
            x = (x * 255).type(torch.uint8)
        return x


def _select_short_text_batch(loader, max_len, max_items):
    for batch in loader:
        texts = batch[1]
        keep = [i for i, text in enumerate(texts) if len(text) <= max_len]
        if keep:
            keep = keep[:max_items]
            imgs = batch[0][keep]
            transcr = [texts[i] for i in keep]
            wids = batch[2][keep]
            style_imgs = batch[3][keep]
            paths = [batch[4][i] for i in keep]
            cor_im = batch[5][keep]
            return (imgs, transcr, wids, style_imgs, paths, cor_im)
    return None


def _maybe_upscale(images, scale):
    if scale is None or scale <= 1:
        return images
    return F.interpolate(images, scale_factor=scale, mode="bilinear", align_corners=False)


def train(diffusion, model, ema, ema_model, vae, optimizer, mse_loss, loader, test_loader, num_classes, style_extractor, vocab_size, noise_scheduler, transforms, args, tokenizer=None, text_encoder=None, lr_scheduler=None):
    model.train()
    loss_meter = AvgMeter()
    print('Training started....')

    for epoch in range(args.epochs):
        print('Epoch:', epoch)
        pbar = tqdm(loader)
        style_feat = []
        for i, data in enumerate(pbar):
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)


            if args.model_name == 'wordstylist':
                batch_word_embeddings = []
                for trans in transcr:
                    word_embedding = label_padding(trans, num_tokens)
                    word_embedding = np.array(word_embedding, dtype="int64")
                    word_embedding = torch.from_numpy(word_embedding).long()
                    batch_word_embeddings.append(word_embedding)
                text_features = torch.stack(batch_word_embeddings)
            else:
                text_features = tokenizer(
                    transcr,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=args.text_max_len,
                ).to(args.device)

            if style_extractor is not None:
                img_h, img_w = args.img_size
                reshaped_images = style_images.reshape(-1, 3, img_h, img_w)
                style_features = style_extractor(reshaped_images)

            else:
                style_features = None

            if args.latent == True:
                images = _unwrap_model(vae).encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215
                latents = images

            noise = torch.randn(images.shape).to(images.device)
            # Sample a random timestep for each image
            num_train_timesteps = diffusion.noise_steps

            timesteps = torch.randint(
                0, num_train_timesteps,
                (images.shape[0],), device=images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude
            # at each timestep (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(
                images, noise, timesteps
            )
            x_t = noisy_images
            t = timesteps

            # Style dropout (CFG training) — was setting unused `labels`, now correctly drops s_id
            s_id_in = None if np.random.random() < 0.1 else s_id

            # Text dropout (CFG training) — drop text conditioning with text_drop_prob
            if np.random.random() < args.text_drop_prob:
                null_texts = [""] * len(transcr)
                text_features_in = tokenizer(
                    null_texts,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=args.text_max_len,
                ).to(args.device)
            else:
                text_features_in = text_features

            predicted_noise = model(x_t, timesteps=t, context=text_features_in, y=s_id_in, style_extractor=style_features)

            loss = mse_loss(noise, predicted_noise)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            ema.step_ema(ema_model, model)

            count = images.size(0)
            loss_meter.update(loss.item(), count)
            pbar.set_postfix(MSE=loss_meter.avg)

            if lr_scheduler is not None and not isinstance(
                lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau
            ):
                lr_scheduler.step()

        val_loss = eval_epoch(
            diffusion,
            model,
            vae,
            mse_loss,
            test_loader,
            style_extractor,
            noise_scheduler,
            args,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )

        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)

        if args.tb_log and args.tb_writer is not None:
            args.tb_writer.add_scalar("train/mse", loss_meter.avg, epoch)
            args.tb_writer.add_scalar("val/mse", val_loss, epoch)
            args.tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        if args.wandb_log:
            wandb.log(
                {
                    "train/mse": loss_meter.avg,
                    "val/mse": val_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                },
                step=epoch,
            )

        do_sampling = (
            args.sample_every
            and (args.save_samples or args.wandb_media)
            and epoch % args.sample_every == 0
        )
        if do_sampling:
            labels = torch.arange(16).long().to(args.device)
            n = len(labels)
            sample_loader = test_loader
            if args.wandb_media and args.wandb_sample_max_len:
                short_batch = _select_short_text_batch(
                    test_loader,
                    args.wandb_sample_max_len,
                    args.wandb_sample_size,
                )
                if short_batch is not None:
                    sample_loader = [short_batch]
                    labels = short_batch[2].to(args.device)
                    n = labels.size(0)

            if args.sampling_word == True:
                #generates the word "text" in 16 different styles
                words = ['text']
                for x_text in words:
                    ema_sampled_images = diffusion.sample(ema_model, vae, n=n, x_text=x_text, labels=labels, args=args)

                    epoch_n = epoch
                    sampled_ema = None
                    if args.save_samples:
                        img_out = _maybe_upscale(ema_sampled_images, args.wandb_upscale)
                        sampled_ema = save_images(
                            img_out,
                            os.path.join(args.save_path, 'images', f"{x_text}_{epoch_n}_ema.jpg"),
                            args,
                        )
            else:
                #generates a batch of words
                ema_sampled_images = diffusion.sampling_loader(
                    ema_model,
                    sample_loader,
                    vae,
                    n=n,
                    x_text=None,
                    labels=labels,
                    args=args,
                    style_extractor=style_extractor,
                    noise_scheduler=noise_scheduler,
                    cfg_scale=args.cfg_scale,
                    transform=transforms,
                    character_classes=None,
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                )
                epoch_n = epoch
                sampled_ema = None
                if args.save_samples:
                    img_out = _maybe_upscale(ema_sampled_images, args.wandb_upscale)
                    sampled_ema = save_images(
                        img_out,
                        os.path.join(args.save_path, 'images', f"{epoch_n}_ema.jpg"),
                        args,
                    )

            if args.wandb_log==True and args.wandb_media:
                caption = f"{x_text}_{epoch}" if args.sampling_word else f"epoch_{epoch}"
                if sampled_ema is None:
                    img_out = _maybe_upscale(ema_sampled_images, args.wandb_upscale)
                    sampled_ema = save_images(img_out, None, args)
                wandb_sampled_ema= wandb.Image(sampled_ema, caption=caption)
                wandb.log({f"Sampled images": wandb_sampled_ema})
        do_save = args.save_every and epoch % args.save_every == 0
        if do_save or (args.save_last and epoch == args.epochs - 1):
            torch.save(model.state_dict(), os.path.join(args.save_path, "models", "ckpt.pt"))
            torch.save(ema_model.state_dict(), os.path.join(args.save_path, "models", "ema_ckpt.pt"))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, "models", "optim.pt"))


def eval_epoch(diffusion, model, vae, mse_loss, loader, style_extractor, noise_scheduler, args, tokenizer=None, text_encoder=None):
    model.eval()
    loss_meter = AvgMeter()
    with torch.no_grad():
        for data in loader:
            images = data[0].to(args.device)
            transcr = data[1]
            s_id = data[2].to(args.device)
            style_images = data[3].to(args.device)

            if args.model_name == 'wordstylist':
                batch_word_embeddings = []
                for trans in transcr:
                    word_embedding = label_padding(trans, num_tokens)
                    word_embedding = np.array(word_embedding, dtype="int64")
                    word_embedding = torch.from_numpy(word_embedding).long()
                    batch_word_embeddings.append(word_embedding)
                text_features = torch.stack(batch_word_embeddings)
            else:
                text_features = tokenizer(
                    transcr,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    max_length=args.text_max_len,
                ).to(args.device)

            if style_extractor is not None:
                img_h, img_w = args.img_size
                reshaped_images = style_images.reshape(-1, 3, img_h, img_w)
                style_features = style_extractor(reshaped_images)
            else:
                style_features = None

            if args.latent == True:
                images = _unwrap_model(vae).encode(images.to(torch.float32)).latent_dist.sample()
                images = images * 0.18215

            noise = torch.randn(images.shape).to(images.device)
            num_train_timesteps = diffusion.noise_steps
            timesteps = torch.randint(
                0, num_train_timesteps,
                (images.shape[0],), device=images.device
            ).long()
            noisy_images = noise_scheduler.add_noise(
                images, noise, timesteps
            )
            predicted_noise = model(
                noisy_images,
                timesteps=timesteps,
                context=text_features,
                y=s_id,
                style_extractor=style_features,
            )
            loss = mse_loss(noise, predicted_noise)
            loss_meter.update(loss.item(), images.size(0))
    model.train()
    return loss_meter.avg


def main():
    '''Main function'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=320)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_name', type=str, default='diffusionpen', help='diffusionpen or wordstylist (previous work)')
    parser.add_argument('--level', type=str, default='word', help='word, line')
    parser.add_argument('--img_size', type=int, default=(64, 256))
    parser.add_argument('--img_height', type=int, default=64)
    parser.add_argument('--img_width', type=int, default=256)
    parser.add_argument('--text_max_len', type=int, default=40)
    parser.add_argument('--dataset', type=str, default='iam', help='iam, gnhk')
    parser.add_argument('--dataset_root', type=str, default=None, help='Override dataset root (used for ukr).')
    parser.add_argument('--ukr_meta_file', type=str, default=None, help='Optional METAFILE.tsv path for ukr.')
    parser.add_argument('--val_size', type=int, default=None, help='Validation subset size (defaults to batch_size).')
    #UNET parameters
    parser.add_argument('--channels', type=int, default=4)
    parser.add_argument('--emb_dim', type=int, default=320)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_res_blocks', type=int, default=1)
    parser.add_argument('--save_path', type=str, default='./diffusionpen_iam_model_path')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--lr_schedule', type=str, default='cosine', help='none, cosine, plateau')
    parser.add_argument('--wandb_log', type=bool, default=False)
    parser.add_argument('--wandb_media', type=bool, default=False)
    parser.add_argument('--wandb_sample_max_len', type=int, default=20)
    parser.add_argument('--wandb_sample_size', type=int, default=8)
    parser.add_argument('--wandb_upscale', type=int, default=2)
    parser.add_argument('--wandb_project', type=str, default='DiffusionPen')
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--wandb_dir', type=str, default=None)
    parser.add_argument('--wandb_name', type=str, default=None)
    parser.add_argument('--tb_log', type=bool, default=True)
    parser.add_argument('--tb_logdir', type=str, default=None)
    parser.add_argument('--color', type=bool, default=True)
    parser.add_argument('--unet', type=str, default='unet_latent', help='unet_latent')
    parser.add_argument('--latent', type=bool, default=True)
    parser.add_argument('--img_feat', type=bool, default=True)
    parser.add_argument('--interpolation', type=bool, default=False)
    parser.add_argument('--dataparallel', type=bool, default=False)
    parser.add_argument('--load_check', type=bool, default=False)
    parser.add_argument('--sampling_word', type=bool, default=False)
    parser.add_argument('--sample_every', type=int, default=10)
    parser.add_argument('--save_samples', type=bool, default=True)
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--save_last', type=bool, default=True)
    parser.add_argument('--mix_rate', type=float, default=None)
    parser.add_argument('--text_drop_prob', type=float, default=0.1,
                        help='Probability of dropping text conditioning during training for CFG (default: 0.1)')
    parser.add_argument('--cfg_scale', type=float, default=1.0,
                        help='Classifier-free guidance scale at sampling (1.0 = off, 3-7 = typical range)')
    parser.add_argument('--style_path', type=str, default='./style_models/iam_style_diffusionpen.pth')
    parser.add_argument('--stable_dif_path', type=str, default='./stable-diffusion-v1-5')
    parser.add_argument('--train_mode', type=str, default='train', help='train, sampling')
    parser.add_argument('--sampling_mode', type=str, default='single_sampling', help='single_sampling (generate single image), paragraph (generate paragraph)')

    args = parser.parse_args()
    args.img_size = (args.img_height, args.img_width)

    print('torch version', torch.__version__)

    args.tb_writer = None
    if args.tb_log:
        log_root = args.tb_logdir or os.path.join(args.save_path, "tensorboard")
        run_name = f"{args.dataset}_{args.model_name}_{int(time.time())}"
        log_dir = os.path.join(log_root, run_name)
        os.makedirs(log_dir, exist_ok=True)
        args.tb_writer = SummaryWriter(log_dir=log_dir)
        print('TensorBoard log dir:', log_dir)

    if args.wandb_log==True:
        runs = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name or args.dataset,
            config=vars(args),
            dir=args.wandb_dir,
        )

        wandb.config.update(args)

    #create save directories
    setup_logging(args)

    ############################ DATASET ############################
    transform = transforms.Compose([
                        #transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=0.1, fill=255),
                        transforms.ToTensor(),
                        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                        ])

    if args.dataset == 'iam':
        print('loading IAM')
        iam_folder = './iam_data/words'
        myDataset = IAMDataset
        style_classes = 339
        if args.level == 'word':
            train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(args.img_height, args.img_width), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        else:
            train_data = myDataset(iam_folder, 'train', 'word', fixed_size=(args.img_height, args.img_width), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
            test_data = myDataset(iam_folder, 'test', 'word', fixed_size=(args.img_height, args.img_width), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=transform, args=args)
        print('train data', len(train_data))

        test_size = args.val_size or args.batch_size
        rest = len(train_data) - test_size
        test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))

    elif args.dataset == 'gnhk':
        print('loading GNHK')
        myDataset = GNHK_Dataset
        dataset_folder = 'path/to/GNHK'
        style_classes = 515
        train_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #transforms.Normalize((0.5,), (0.5,)),  #
                            ])
        train_data = myDataset(dataset_folder, 'train', 'word', fixed_size=(args.img_height, args.img_width), tokenizer=None, text_encoder=None, feat_extractor=None, transforms=train_transform, args=args)
        test_size = args.val_size or args.batch_size
        rest = len(train_data) - test_size
        test_data, _ = random_split(train_data, [test_size, rest], generator=torch.Generator().manual_seed(42))
    elif args.dataset == 'ukr':
        print('loading UKR')
        dataset_folder = args.dataset_root or './UkrHandwritten'
        if not os.path.isdir(dataset_folder):
            raise FileNotFoundError(
                f"UKR dataset root not found at '{dataset_folder}'. Use --dataset_root."
            )
        # Use UkrWordDataset if words/words/ exists, else UkrDataset (line-level)
        words_dir = os.path.join(dataset_folder, 'words', 'words')
        if os.path.isdir(words_dir):
            print('Using UkrWordDataset (word-level)')
            UkrClass = UkrWordDataset
        else:
            print('Using UkrDataset (line-level)')
            UkrClass = UkrDataset
        train_data = UkrClass(
            dataset_folder,
            'all',
            'word',
            fixed_size=(args.img_height, args.img_width),
            tokenizer=None,
            text_encoder=None,
            feat_extractor=None,
            transforms=transform,
            meta_file=args.ukr_meta_file,
        )
        style_classes = getattr(train_data, "wclasses", 0)
        test_size = args.val_size or args.batch_size
        rest = len(train_data) - test_size
        test_data, _ = random_split(
            train_data,
            [test_size, rest],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        raise ValueError(f"Dataset '{args.dataset}' is not supported.")

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    if args.dataset == 'ukr':
        character_classes = [WORD_INDEX2LETTER[i] for i in sorted(WORD_INDEX2LETTER)]
    else:
        character_classes = ['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ']

    ######################### MODEL #######################################
    if args.model_name == 'wordstylist':
        vocab_size = len(character_classes) + 2
        print('vocab size', vocab_size)
    else:
        vocab_size = len(character_classes)
    print('Vocab size: ', vocab_size)

    if args.dataparallel==True:
        device_ids = list(range(torch.cuda.device_count()))
        print('using dataparallel with device:', device_ids)
    else:
        idx = int(''.join(filter(str.isdigit, args.device)))
        device_ids = [idx]
    #unet = unet.to(args.device)

    if args.model_name == 'diffusionpen':
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = CanineModel.from_pretrained("google/canine-c")
        # Do NOT wrap text_encoder in DataParallel — it is stored inside UNetModel
        # (self.text_encoder) which is itself wrapped in outer DataParallel.
        # Nested DataParallel causes StopIteration on replica devices.
        text_encoder = text_encoder.to(args.device)

    else:
        tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        text_encoder = None

    if args.unet=='unet_latent':
        unet_kwargs = dict(
            image_size=args.img_size,
            in_channels=args.channels,
            model_channels=args.emb_dim,
            out_channels=args.channels,
            num_res_blocks=args.num_res_blocks,
            attention_resolutions=(1, 1),
            channel_mult=(1, 1),
            num_heads=args.num_heads,
            num_classes=style_classes,
            context_dim=args.emb_dim,
            vocab_size=vocab_size,
            text_encoder=text_encoder,
            args=args,
        )
        unet = UNetModel(**unet_kwargs)

    unet = DataParallel(unet, device_ids=device_ids)
    unet = unet.to(args.device)

    #print('unet parameters')
    #print('unet', sum(p.numel() for p in unet.parameters() if p.requires_grad))

    optimizer = optim.AdamW(unet.parameters(), lr=args.lr)
    lr_scheduler = None
    total_steps = args.epochs * max(1, len(train_loader))
    if args.lr_schedule == 'cosine':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=args.lr_min,
        )
    elif args.lr_schedule == 'plateau':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=5,
            factor=0.5,
            min_lr=args.lr_min,
        )

    mse_loss = nn.MSELoss()
    diffusion = Diffusion(img_size=args.img_size, args=args)

    ema = EMA(0.995)
    ema_model = build_ema_model(unet, unet_kwargs, device_ids, args.device)

    #load from last checkpoint

    if args.load_check==True:
        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', map_location=args.device))
        optimizer.load_state_dict(torch.load(f'{args.save_path}/models/optim.pt', map_location=args.device))
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt', map_location=args.device))
        print('Loaded models and optimizer')

    if args.latent==True:
        print('VAE is true')
        vae = AutoencoderKL.from_pretrained(args.stable_dif_path, subfolder="vae")
        vae = DataParallel(vae, device_ids=device_ids)
        vae = vae.to(args.device)
        # Freeze vae and text_encoder
        vae.requires_grad_(False)
    else:
        vae = None

    #add DDIM scheduler from huggingface
    ddim = DDIMScheduler.from_pretrained(args.stable_dif_path, subfolder="scheduler")

    #### STYLE ####
    feature_extractor = ImageEncoder(model_name='mobilenetv2_100', num_classes=0, pretrained=True, trainable=True)
    PATH = args.style_path

    state_dict = torch.load(PATH, map_location=args.device)
    model_dict = feature_extractor.state_dict()
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
    model_dict.update(state_dict)
    feature_extractor.load_state_dict(model_dict)
    feature_extractor = DataParallel(feature_extractor, device_ids=device_ids)
    feature_extractor = feature_extractor.to(args.device)
    feature_extractor.requires_grad_(False)
    feature_extractor.eval()

    if args.train_mode == 'train':
        train(diffusion, unet, ema, ema_model, vae, optimizer, mse_loss, train_loader, test_loader, style_classes, feature_extractor, vocab_size, ddim, transform, args, tokenizer=tokenizer, text_encoder=text_encoder, lr_scheduler=lr_scheduler)

    elif args.train_mode == 'sampling':

        print('Sampling started....')

        unet.load_state_dict(torch.load(f'{args.save_path}/models/ckpt.pt', map_location=args.device))
        print('unet loaded')
        unet.eval()

        ema = EMA(0.995)
        ema_model = build_ema_model(unet, unet_kwargs, device_ids, args.device)
        ema_model.load_state_dict(torch.load(f'{args.save_path}/models/ema_ckpt.pt', map_location=args.device))
        ema_model.eval()

        if args.sampling_mode == 'single_sampling':
            x_text = ['text', 'word']
            for x_text in x_text:
                print('Word:', x_text)
                s = random.randint(0, 339) #index for style class

                print('style', s)
                labels = torch.tensor([s]).long().to(args.device)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=x_text, labels=labels, args=args, style_extractor=feature_extractor, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, run_idx=None)
                save_single_images(ema_sampled_images, os.path.join(f'./image_samples/', f'{x_text}_style_{s}.png'), args)


        elif args.sampling_mode == 'paragraph':
            print('Sampling paragraph')
            #make the code to generate lines
            lines = 'In this work , we focus on style variation . We present a novel method to control the style of the text . Our method is able to mimic various writing styles .'
            fakes= []
            gap = np.ones((64, 16))
            max_line_width = 900
            total_char_count = 0
            avg_char_width = 0
            current_line_width = 0
            longest_word_length = max(len(word) for word in lines.strip().split(' '))
            #print('longest_word_length', longest_word_length)
            #s = random.randint(0, 339)#.long().to(args.device)
            #s = random.randint(0, 161)#.long().to(args.device)
            s = 12 #25 #129 #201
            for word in lines.strip().split(' '):
                print('Word:', word)
                print('Style:', s)
                labels = torch.tensor([s]).long().to(args.device)
                ema_sampled_images = diffusion.sampling(ema_model, vae, n=len(labels), x_text=word, labels=labels, args=args, style_extractor=feature_extractor, noise_scheduler=ddim, transform=transform, character_classes=None, tokenizer=tokenizer, text_encoder=text_encoder, clip_model=None, run_idx=None)
                #print('ema_sampled_images', ema_sampled_images.shape)
                image = ema_sampled_images.squeeze(0)

                im = torchvision.transforms.ToPILImage()(image)
                #reshape to height 32
                im = im.convert("L")
                #save im

                #if len(word) < 4:

                im = crop_whitespace_width(im)

                im = Image.fromarray(im)
                if len(word) == longest_word_length:
                    max_word_length_width = im.width
                    print('max_word_length_width', max_word_length_width)
                #im.save(f'./_REBUTTAL/{word}.png')
                # Calculate aspect ratio
                aspect_ratio = im.width / im.height

                im = np.array(im)
                #im = np.array(resized_img)

                fakes.append(im)

            # Calculate the scaling factor based on the longest word
            #find the average character width of the max length word

            avg_char_width = max_word_length_width / longest_word_length
            print('avg_char_width', avg_char_width)
            #scaling_factor = avg_char_width / (32 * aspect_ratio)  # Aspect ratio of an average character

            # Scale and pad each word
            scaled_padded_words = []
            max_height = 64  # Defined max height for all images

            for word, img in zip(lines.strip().split(' '), fakes):

                img_pil = Image.fromarray(img)
                as_ratio = img_pil.width / img_pil.height
                #scaled_width = int(scaling_factor * len(word))#) * as_ratio * max_height)
                scaled_width = int(avg_char_width * len(word))

                scaled_img = img_pil.resize((scaled_width, int(scaled_width / as_ratio)))
                print(f'Word {word} - scaled_img {scaled_img.size}')
                # Padding
                #if word is in punctuation:
                if word in punctuation:
                    #rescale to height 10
                    w_punc = scaled_img.width
                    h_punc = scaled_img.height
                    as_ratio_punct = w_punc / h_punc
                    if word == '.':
                        scaled_img = scaled_img.resize((int(5 * as_ratio_punct), 5))
                    else:
                        scaled_img = scaled_img.resize((int(13 * as_ratio_punct), 13))
                    #pad on top and leave the image in the bottom
                    padding_bottom = 10
                    padding_top = max_height - scaled_img.height - padding_bottom# All padding goes on top
                      # No padding at the bottom

                    # Apply padding
                    padded_img = np.pad(scaled_img, ((padding_top, padding_bottom), (0, 0)), mode='constant', constant_values=255)
                else:
                    if scaled_img.height < max_height:
                        padding = (max_height - scaled_img.height) // 2
                        #print(f'Word {word} - padding: {padding}')
                        padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)
                    else:
                        #resize to max height while maintaining aspect ratio
                        #ar = scaled_img.width / scaled_img.height

                        scaled_img = scaled_img.resize((int(max_height * as_ratio) - 4, max_height - 4))
                        padding = (max_height - scaled_img.height) // 2
                        #print(f'Word {word} - padding: {padding}')
                        padded_img = np.pad(scaled_img, ((padding, max_height - scaled_img.height - padding), (0, 0)), mode='constant', constant_values=255)

                    #padded_img = np.array(scaled_img)
                #print('padded_img', padded_img.shape)
                scaled_padded_words.append(padded_img)

            # Create a gap array (white space)
            height = 64  # Fixed height for all images
            gap = np.ones((height, 16), dtype=np.uint8) * 255  # White gap

            # Concatenate images with gaps
            sentence_img = gap  # Start with a gap
            lines = []
            line_img = gap
            # Concatenate images with gaps
            '''
            sentence_img = gap  # Start with a gap
            for img in scaled_padded_words:
                #print('img', img.shape)
                sentence_img = np.concatenate((sentence_img, img, gap), axis=1)
            '''

            for img in scaled_padded_words:
                img_width = img.shape[1] + gap.shape[1]

                if current_line_width + img_width < max_line_width:
                    # Add the image to the current line
                    if line_img.shape[0] == 0:
                        line_img = np.ones((height, 0), dtype=np.uint8) * 255  # Start a new line
                    line_img = np.concatenate((line_img, img, gap), axis=1)
                    current_line_width += img_width #+ gap.shape[1]
                    #print('current_line_width if', current_line_width)
                    # Check if adding this image exceeds the max line width
                else:
                    # Pad the current line with white space to max_line_width
                    remaining_width = max_line_width - current_line_width
                    line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
                    lines.append(line_img)

                    # Start a new line with the current word
                    line_img = np.concatenate((gap, img, gap), axis=1)
                    current_line_width = img_width #+ 2 * gap.shape[1]
                    #print('current_line_width else', current_line_width)
            # Add the last line to the lines list
            if current_line_width > 0:
                # Pad the last line to max_line_width
                remaining_width = max_line_width - current_line_width
                line_img = np.concatenate((line_img, np.ones((height, remaining_width), dtype=np.uint8) * 255), axis=1)
                lines.append(line_img)

            # # Concatenate all lines to form a paragraph, pad them if necessary
            # max_height = max([line.shape[0] for line in lines])
            # paragraph_img = np.ones((0, max_line_width), dtype=np.uint8) * 255
            # for line in lines:
            #     if line.shape[0] < max_height:
            #         padding = (max_height - line.shape[0]) // 2
            #         line = np.pad(line, ((padding, max_height - line.shape[0] - padding), (0, 0)), mode='constant', constant_values=255)

            #     #print the shapes
            #     print('line shape', line.shape)
            #print('paragraph shape', paragraph_img.shape)
            paragraph_img = np.concatenate((lines), axis=0)


            paragraph_image = Image.fromarray(paragraph_img)
            paragraph_image = paragraph_image.convert("L")

            paragraph_image.save(f'paragraph_style_{s}.png')

    if args.tb_writer is not None:
        args.tb_writer.close()

if __name__ == "__main__":
    main()
