import copy
import logging
import os
import pathlib
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils_SimCLR import save_config_file, accuracy, save_checkpoint

# Import utils
parent_dir = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(parent_dir))
import utils

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        try:
            self.writer.all_writers = {self.args.save_folder if k == self.writer.log_dir else k: v for k, v in self.writer.all_writers.items()}
            self.writer.log_dir = self.args.save_folder
        except AttributeError:
            pass
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.loss = ContrastiveLoss(self.args.batch_size, self.args.temperature,
                                    use_iipp=self.args.use_iipp, device=self.args.device)

    def info_nce_loss(self, features, meta_data=None):
        # features shape: [2b, nb_class]
        # meta_data shape: [b], containing the area id from which the sample was taken
        # Indicate which images are positive pairs
        if meta_data is None:
            labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        else:
            labels = torch.concat([meta_data, meta_data], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float() # shape: [2b, 2b]
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1) # shape: [2b, nb_class]

        similarity_matrix = torch.matmul(features, features.T) # shape: [2b, 2b]
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        # new matrix: same as the old, but with the self-self (diagonal) element removed
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device) # shape: [2b, 2b]
        labels = labels[~mask].view(labels.shape[0], -1) # shape: [2b, 2b-1]
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1) # shape: [2b, 2b-1]
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives (idx marked with a 1 in labels)
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1) # shape: [2b, 1]

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1) # shape: [2b, 2b-2]

        logits = torch.cat([positives, negatives], dim=1) # shape: [2b, 2b-1], 1st col containing positive similarity scores
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device) # shape: [2b]

        logits = logits / self.args.temperature # shape: [2b, 2b-1]
        return logits, labels

    @staticmethod
    def get_labels_from_metadata(meta_data):
        pass

    def train(self, train_loader):
        scaler = GradScaler(self.args.device, enabled=self.args.fp16_precision)
        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        if self.args.wandb.wandb_log:
            utils.wandb_init(self.args.wandb.project_name, hyperparams=vars(self.args))

        best_loss = 1e6
        best_top1 = 0
        best_epoch = 0
        for epoch_counter in range(self.args.epochs):
            print(f"================================\n"
                  f"Epoch {epoch_counter}")
            if (epoch_counter - best_epoch) >= self.args.patience:
                print(f'Loss has not improved for {self.args.patience} epochs. Training has stopped')
                print(f'Best loss was {best_loss} @ epoch {best_epoch}')
                break

            # Reset sampling weights
            if self.args.use_iipp:
                train_loader.dataset.reset_sampling_weights()
            avg_epoch_loss = []
            meta_data = None
            for images, _ in tqdm(train_loader):
                if self.args.use_iipp:
                    meta_data = images[1]
                    images = images[0] # shape: [b, c, h, w] x2, shape iipp: [b/num_same_area, num_same_area, h, w] x2 (transform pairs)
                    # DEBUG
                    """
                    # Create sample data: Value at [0,0] within an image indicates the original position.
                    # Ex: 0.3 means image 3 from area 0
                    a = torch.stack([j+torch.stack([torch.reshape(torch.arange(9), (3, 3)) + i*0.1 for i in range(self.args.num_same_area)], dim=0) for j in range(int(self.args.batch_size/self.args.num_same_area))], dim=0)
                    a = [a, -a] # Simulate transform
                    # Split grouped area images back to individual images
                    # [[[0.0, 1.0, 2.0, 3.0], -> image 0, area 0, 1, 2, 3
                    #    [0.1, 1.1, 2.1, 3.1], -> image 1, area 0, 1, 2, 3
                    #    [0.2, 1.2, 2.2, 3.2], -> image 2, area 0, 1, 2, 3
                    #    [0.3, 1.3, 2.3, 3,3]], -> image 3, area 0, 1, 2, 3
                    #  [transforms']]
                    b = [[ai[:, i * self.args.img_channel:(i + 1) * self.args.img_channel, :, :] for i in
                               range(self.args.num_same_area)] for ai in
                              a]  # shape: [[new_b, c, h, w] xnum_same_area] x2
                    # Contact per transform
                    # area.image
                    # [[0.0, 1.0, 2.0, 3.0, 0.1, 1.1, 2.1, 3.1, 0.2, ... 3.2, 0.3, ... 3.3], [transform']]
                    c = [torch.cat(bi, dim=0) for bi in b] # shape: [b, c, h, 2]
                    """
                    # Split grouped area images back to individual images
                    images = [[img[:, i * self.args.img_channel:(i + 1) * self.args.img_channel, :, :] for i in
                               range(self.args.num_same_area)] for img in
                              images]  # shape iipp: [[new_b, c, h, w] xnum_same_area] x2 (transform pairs)
                    images = [torch.cat(img, dim=0) for img in images] # shape: [b, c, h, w] x2 (transform pairs)
                    # Update meta_data to match image order
                    meta_data = meta_data.repeat(self.args.num_same_area) # meta data for one transform
                images = torch.cat(images, dim=0) # shape: [2b, c, h, w], shape iipp: [2b, c, h, w]
                images = images.to(self.args.device)


                with autocast(device_type=str(self.args.device), enabled=self.args.fp16_precision):
                    features = self.model(images) # shape:[2b, nb_class]
                    # Old loss, iipp not possible
                    # logits, labels = self.info_nce_loss(features, meta_data)
                    # loss = self.criterion(logits, labels)
                    # New loss, with iipp if needed
                    loss = self.loss(features[:self.args.batch_size, :], features[self.args.batch_size:, :], meta_data)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                if (n_iter % self.args.log_every_n_steps == 0) or self.args.wandb.wandb_log:
                    # top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    if self.args.wandb.wandb_log:
                        utils.wandb_log('batch',
                                        loss=loss,
                                        # acc_top1= top1[0],
                                        # acc_top5=top5[0],
                                        lr=self.scheduler.get_last_lr()[0])
                    if n_iter % self.args.log_every_n_steps == 0:
                        self.writer.add_scalar('loss', loss, global_step=n_iter)
                        # self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                        # self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                        self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1
                avg_epoch_loss.append(loss)

            avg_epoch_loss = float(torch.mean(torch.stack(avg_epoch_loss)).cpu().detach().numpy())
            # top1, top5 = accuracy(logits, labels, topk=(1, 5))
            print(f'Average loss: {avg_epoch_loss}')
            # print(f'Top 1: {top1}')
            # print(f'Top 5: {top5}')
            # if top1 > best_top1:
            if avg_epoch_loss < best_loss:
                best_epoch = epoch_counter
                best_loss = avg_epoch_loss
                # best_top1 = top1
                # best_model_weights = copy.deepcopy(self.model.state_dict())
                # Save model
                torch.save(self.model.state_dict(), f"{self.writer.log_dir}/checkpoint_best.pth")
                print(f'New best loss achieved at epoch {best_epoch}: {best_loss}')
            if self.args.wandb.wandb_log:
                utils.wandb_log('epoch',
                                loss=avg_epoch_loss,
                                # acc_top1=top1,
                                # acc_top5=top5,
                                lr=self.scheduler.get_last_lr()[0])

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            # logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")

        logging.info("Training has finished.")
        # Update best model name to include epoch
        best_weights_path = pathlib.Path(f"{self.writer.log_dir}/checkpoint_best.pth")
        best_weights_path.rename(best_weights_path.parent.joinpath(f'checkpoint_best_{best_epoch:04d}.pt'))
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")


class ContrastiveLoss(nn.Module):
   """
   Vanilla Contrastive loss, also called InfoNceLoss as in SimCLR paper
   Taken from https://theaisummer.com/simclr/#l2-normalization-and-cosine-similarity-matrix-calculation
   """
   def __init__(self, batch_size, temperature=0.5, use_iipp=False, device=None):
       super().__init__()
       self.batch_size = batch_size
       self.temperature = temperature
       self.use_iipp = use_iipp
       self.mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
       self.device = device
       if device is not None:
           self.mask.to(device)

   def calc_similarity_batch(self, a, b):
       representations = torch.cat([a, b], dim=0)
       return F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

   def forward(self, proj_1, proj_2, meta_data=None):
       """
       proj_1 and proj_2 are batched embeddings [batch, embedding_dim]
       where corresponding indices are pairs
       z_i, z_j in the SimCLR paper
       """
       batch_size = proj_1.shape[0]
       z_i = F.normalize(proj_1, p=2, dim=1) # shape: [b, nb_class]
       z_j = F.normalize(proj_2, p=2, dim=1) # shape: [b, nb_class]

       similarity_matrix = self.calc_similarity_batch(z_i, z_j) # shape: [2b, 2b]

        # Positive pairs, no iipp
       if meta_data is None:
           sim_ij = torch.diag(similarity_matrix, batch_size) # shape: [b]
           sim_ji = torch.diag(similarity_matrix, -batch_size) # shape: [b]
           positives = torch.cat([sim_ij, sim_ji], dim=0)  # shape: [2b]

           denominator = self.mask.to(similarity_matrix.device) * torch.exp(
               similarity_matrix / self.temperature)  # shape: [2b, 2b]
       else:
           positive_idxs = torch.concat([meta_data, meta_data], dim=0)
           positive_idxs = (positive_idxs.unsqueeze(0) == positive_idxs.unsqueeze(1)).float() # shape: [2b, 2b]
           positive_idxs = -1*torch.eye(positive_idxs.shape[0]) + positive_idxs # remove diagonal
           positives = torch.masked_select(similarity_matrix, positive_idxs.bool().to(similarity_matrix.device)) # shape: [x: nb_positive_pairs]
           # Find which rows are to be repeated how many times
           rep_rows = torch.sum(positive_idxs, dim=1).detach().tolist() # len: 2b
           # Repeat rows in similarity and mask matrix
           rep_sim = ContrastiveLoss.repeat_rows_iipp(similarity_matrix, rep_rows) # shape: [x, 2b]
           rep_mask = ContrastiveLoss.repeat_rows_iipp(self.mask, rep_rows) # shape: [x, 2b]
           denominator = rep_mask.to(rep_sim.device) * torch.exp(rep_sim / self.temperature)  # shape: [x, 2b]

       nominator = torch.exp(positives / self.temperature) # shape: [2b] or [x] for iipp

       # denominator = self.mask.to(similarity_matrix.device) * torch.exp(similarity_matrix / self.temperature) # shape: [2b, 2b]

       all_losses = -torch.log(nominator / torch.sum(denominator, dim=1)) # shape: [2b] or [x] for iipp
       loss = torch.sum(all_losses) / (2 * self.batch_size) # shape: 1
       return loss

   @staticmethod
   def repeat_rows_iipp(mat:torch.Tensor, rep_rows:list):
       rep_sim = [mat[:, i].repeat(int(r), 1) for i, r in enumerate(rep_rows)]
       mat = torch.cat(rep_sim, dim=0)
       return mat