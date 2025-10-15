import wandb
import torch.nn as nn
import torch
import argparse
import os
import time
import numpy as np

import torch.optim as optim
import torch.nn.functional as F

from datasets3D_studentTeacher_level_ps_global_local import get_loader
from config import POPAR_3D_Org
from monai.networks.nets import SwinUNETR_MIM, SwinUNETR

from utils import save_student_teacher_level_ps_model, AverageMeter
from timm.utils import NativeScaler
import math
import sys

parser = argparse.ArgumentParser('argument for training')
parser.add_argument('--batch_size', type=int, default=12,  help='batch_size')
parser.add_argument('--num_workers', type=int, default=4, help='num of workers to use')
parser.add_argument('--weight', dest='weight', default=None)
parser.add_argument("--num_steps", default=100000, type=int, help="number of training iterations")
parser.add_argument("--epochs", default=1000, type=int, help="number of training epochs")
parser.add_argument('--gpu', dest='gpu', default="0,1,2,3", type=str, help="gpu index")

parser.add_argument("--eval_num", default=100, type=int, help="evaluation frequency")
parser.add_argument("--image_size", default=128, type=int, help="evaluation frequency")
parser.add_argument("--patch_size", default=16, type=int, help="evaluation frequency")

parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--task', nargs='+', default=["level_ps", "global_consistency"])
parser.add_argument('--dataset', nargs='+')
parser.add_argument('--losses', nargs='+')

parser.add_argument("--opt", default="adamw", type=str, help="optimization algorithm")
parser.add_argument("--lrdecay", action="store_true", help="enable learning rate decay")
parser.add_argument("--decay", default=0.1, type=float, help="decay rate")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")

parser.add_argument("--grad_clip", action="store_true", help="gradient clip")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="maximum gradient norm")

parser.add_argument("--lr_schedule", default="warmup_cosine", type=str)
parser.add_argument("--warmup_steps", default=500, type=int, help="warmup steps")
parser.add_argument("--backbone", default="swinunetr", type=str, help="backbone")
parser.add_argument("--continue_pretrain", default=False, type=bool)

args = parser.parse_args()

assert set(args.task).issubset(["level_ps_global_local_consistency"])
assert set(args.dataset).issubset(["luna", "lidc", "tciacolon", "tciacovid", "hnscc", "oasis","amos22","btcv"])
assert set(args.losses).issubset(["order" ,"recovery"])
assert args.backbone in ["swinunetr"]

def step_decay(step, conf,warmup_epochs = 5):
    lr = conf.lr
    progress = (step - warmup_epochs) / float(conf.epochs - warmup_epochs)
    progress = np.clip(progress, 0.0, 1.0)
    #decay_type == 'cosine':
    lr = lr * 0.5 * (1. + np.cos(np.pi * progress))
    if warmup_epochs:
      lr = lr * np.minimum(1., step / warmup_epochs)
    return lr

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class POPAR3DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = SwinUNETR(
            in_channels=1,
            out_channels=1,
            img_size=config.image_size,
            feature_size=48,
            drop_rate = 0.0,
            attn_drop_rate = 0.0,
            dropout_path_rate = 0.1,
        )

        if config.patch_size == 16:
            self.cls_feature_index = 3
            self.cls_feature_dim = 384
        elif config.patch_size == 32:
            self.cls_feature_index = 4
            self.cls_feature_dim = 768

        self.head = nn.Sequential(
            nn.Linear(self.cls_feature_dim, self.cls_feature_dim),
            nn.ReLU(),
            nn.Linear(self.cls_feature_dim, self.cls_feature_dim),
            nn.ReLU(),
            nn.Linear(self.cls_feature_dim, 3),
        )
        self.mlp = self.MLP('8192-8192-8192', self.cls_feature_dim)
        self.mlp_local = self.MLP('512-512-512', self.cls_feature_dim)

    def forward(self, img_x, use_consistency):
        restor_out = self.model(img_x)
        cls_feature = self.model.hidden_states_out[self.cls_feature_index]
        B,C,D,H,W = cls_feature.shape
        cls_feature = cls_feature.permute(0, 2, 3, 4, 1)
        if use_consistency:
            avg_out = cls_feature.reshape(B, D * H * W, C).mean(dim=1)
            global_embd = self.mlp(avg_out)
            local_embd = self.mlp_local(cls_feature)
            local_embd = local_embd.view(B, D * H * W, -1)
            return global_embd, local_embd
        else:
            cls_feature = cls_feature.flatten(start_dim=0, end_dim=3)
            patch_order_predciton = self.head(cls_feature)
            avg_out = cls_feature.reshape(B, D * H * W, C).mean(dim=1)
            global_embd = self.mlp(avg_out)
            return patch_order_predciton, restor_out, global_embd

    def MLP(self, mlp, embed_dim): # 1024-8192-8192-8192
        mlp_spec = f"{embed_dim}-{mlp}"
        layers = []
        f = list(map(int, mlp_spec.split("-")))
        for i in range(len(f) - 2):
            layers.append(nn.Linear(f[i], f[i + 1]))
            layers.append(nn.LayerNorm(f[i + 1]))
            layers.append(nn.ReLU(True))
        layers.append(nn.Linear(f[-2], f[-1], bias=False))
        return nn.Sequential(*layers)


def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def build_model(config):

    if config.backbone == "swinunetr":
        model = POPAR3DModel(config=config)
    start_epoch = 1
    current_it = 1
    optimizer = optim.SGD(model.parameters(), lr=config.lr, weight_decay=0, momentum=0.9, nesterov=False)

    # optimizer = optim.AdamW(params=model.parameters(), lr=config.lr, weight_decay=0.1)

    loss_scaler = NativeScaler()

    if config.weight is not None:
        checkpoint = torch.load(config.weight, map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['teacher'].items()}
        message = model.load_state_dict(state_dict)

        if not config.continue_pretrain:
            start_epoch = checkpoint['epoch'] + 1
            current_it = checkpoint['current_it']
            if "loss_scaler" in checkpoint.keys():
                loss_scaler.load_state_dict(checkpoint['loss_scaler'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        # config.lr = optimizer.param_groups[0]['lr']
        print("Weight loaded from: {} ".format(config.weight), file=config.log_writter)
        print("Continue Pretraining on other dataset is: {}".format(config.continue_pretrain), file=config.log_writter)

        print(message, file=config.log_writter)

    if torch.cuda.is_available():
        if config.weight is not None:
            optimizer_to(optimizer, "cuda")
        model = torch.nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
        model = model.cuda()

    return model, optimizer,loss_scaler,start_epoch, current_it


def checkNan(gt_whole,aug_whole, batch, conf ):
    nan_flag = False
    if not math.isfinite(torch.min(gt_whole)):
        print("*********************** torch.min(gt_whole) is nan ******************************", gt_whole.shape,
              torch.min(gt_whole), torch.max(gt_whole), file=conf.log_writter)
        print("img path: ", batch['image_meta_dict']['filename_or_obj'], file=conf.log_writter)

        nan_flag = True
    if not math.isfinite(torch.max(gt_whole)):
        print("*********************** torch.max(gt_whole) is nan ******************************", gt_whole.shape,
              torch.min(gt_whole), torch.max(gt_whole), file=conf.log_writter)
        print("img path: ", batch['image_meta_dict']['filename_or_obj'], file=conf.log_writter)

        nan_flag = True
    if not math.isfinite(torch.min(aug_whole)):
        print("*********************** torch.min(aug_whole) is nan ******************************", aug_whole.shape,
              torch.min(aug_whole), torch.max(aug_whole), file=conf.log_writter)
        print("img path: ", batch['image_meta_dict']['filename_or_obj'], file=conf.log_writter)
        nan_flag = True
    if not math.isfinite(torch.max(aug_whole)):
        print("*********************** torch.max(aug_whole) is nan ******************************", aug_whole.shape,
              torch.min(aug_whole), torch.max(aug_whole), file=conf.log_writter)
        print("img path: ", batch['image_meta_dict']['filename_or_obj'], file=conf.log_writter)
        nan_flag = True
    return nan_flag



def save_image(aug_whole,gt_whole, fn_list, config):
    import nibabel as nib
    counter = 0
    for (aug, gt, fn) in zip(aug_whole, gt_whole,fn_list):
        aug =  aug.cpu().squeeze(0).numpy()*255
        gt = gt.cpu().squeeze(0).numpy()*255

        print("aug:", aug.shape, np.min(aug), np.max(aug))
        print("gt: ", gt.shape, np.min(gt), np.max(gt))
        img = nib.load(fn)

        ni_img = nib.Nifti1Image(aug, affine=img.affine)
        nib.save(ni_img, os.path.join(config.model_path, "{}_aug.nii.gz".format(counter)))

        ni_img = nib.Nifti1Image(gt, affine=img.affine)
        nib.save(ni_img, os.path.join(config.model_path, "{}_gt.nii.gz".format(counter)))

        counter +=1




def train_global_local_consistency(train_loader, student, teacher,momentum_schedule, optimizer, epoch,loss_scaler, conf, current_it):
    """one epoch training"""
    student.train(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    global_losses = AverageMeter()
    local_losses = AverageMeter()

    end = time.time()
    mse_loss =nn.MSELoss()
    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        patch1 = batch['patch_1'].cuda(non_blocking=True)
        patch2 = batch['patch_2'].cuda(non_blocking=True)
        index1 = torch.tensor(batch['overlap_mask_1']).cuda(non_blocking=True)
        index2 = torch.tensor(batch['overlap_mask_2']).cuda(non_blocking=True)
        shuffle = batch["shuffle"]

        bsz = patch1.shape[0]

        if checkNan(patch1,patch2,batch,conf):
            print("*********************** Skip ******************************", file=conf.log_writter)
            continue

        global_embd1_s, out1_s = student(patch1, use_consistency= True)  # pred_order1_s [36*196, 196]
        global_embd2_t, out2_t = teacher(patch2, use_consistency= True)
        global_embd2_s, out2_s = student(patch2, use_consistency= True)
        global_embd1_t, out1_t = teacher(patch1, use_consistency= True)

        global_embd1_s = F.normalize(global_embd1_s, p=2.0, dim=1, eps=1e-12, out=None)  # embedding1:[B,196,1024], global_embd1:[B,8192]
        global_embd2_t = F.normalize(global_embd2_t, p=2.0, dim=1, eps=1e-12, out=None)
        global_embd2_s = F.normalize(global_embd2_s, p=2.0, dim=1, eps=1e-12, out=None)
        global_embd1_t = F.normalize(global_embd1_t, p=2.0, dim=1, eps=1e-12, out=None)


        local_loss = torch.tensor([0.0]).cuda()
        if not shuffle.all():
            local_loss = torch.tensor([0.0]).cuda()
            not_shuffle = (1 - shuffle).bool()
            local_loss += mse_loss(out1_s[not_shuffle][index1[not_shuffle]], out2_t[not_shuffle][index2[not_shuffle]])
            local_loss += mse_loss(out1_t[not_shuffle][index1[not_shuffle]], out2_s[not_shuffle][index2[not_shuffle]])
            local_loss /= 2

        global_loss = mse_loss(global_embd1_s, global_embd2_t)+mse_loss(global_embd2_s, global_embd1_t)


        loss =  (global_loss + local_loss)/2

        global_losses.update(global_loss.item(), bsz)
        local_losses.update(local_loss.item(), bsz)
        losses.update(loss.item(), bsz)


        if not math.isfinite( loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=conf.log_writter)
            sys.exit(1)
        # update metric
        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=student.parameters(), create_graph=is_second_order)
        conf.log_writter.flush()

        with torch.no_grad():
            m = momentum_schedule[current_it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        current_it+=1

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'lr {lr}\t'
                  'Global loss {globalloss.val} ({globalloss.avg})\t'
                   'Local loss {localloss.val} ({localloss.avg})\t'
                  'Total loss {ttloss.val:.3f} ({ttloss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                lr=optimizer.param_groups[0]['lr'], globalloss=global_losses,
                localloss=local_losses, ttloss=losses),file=conf.log_writter)
            conf.log_writter.flush()
            if conf.debug_mode:
                break

    wandb.log({"global loss": global_losses.avg,
               "local loss": local_losses.avg})

    return losses.avg, current_it




def train_level_ps(train_loader, student, teacher, momentum_schedule, optimizer, epoch,loss_scaler, conf, current_it):
    """one epoch training"""
    student.train(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    order_losses = AverageMeter()
    restor_losses = AverageMeter()
    global_con_losses = AverageMeter()
    end = time.time()
    mse_loss =nn.MSELoss()

    for idx, batch in enumerate(train_loader):
        data_time.update(time.time() - end)
        aug_whole =  batch["aug_image"].cuda(non_blocking=True)
        gt_whole = batch["image"].cuda(non_blocking=True)
        randperm = batch["permutation"].float().cuda(non_blocking=True)

        if checkNan(gt_whole,aug_whole,batch,conf):
            print("*********************** Skip ******************************", file=conf.log_writter)
            continue



        B,N,C = randperm.shape
        randperm = randperm.reshape(B*N, C)
        
        

        order_predciton_s,restor_out_s,global_emb_s = student(aug_whole, use_consistency= False)
        order_predciton_t,restor_out_t,global_emb_t = teacher(aug_whole, use_consistency= False)
        
        global_embd_s = F.normalize(global_emb_s, p=2.0, dim=1, eps=1e-12, out=None)  # embedding1:[B,196,1024], global_embd1:[B,8192]
        global_embd_t = F.normalize(global_emb_t, p=2.0, dim=1, eps=1e-12, out=None)
        

        # print("randperm: ", randperm, file=conf.log_writter)
        # print("patch_order_predciton: ", patch_order_predciton, file=conf.log_writter)
        # print("aug_whole:", aug_whole.shape, torch.min(aug_whole),torch.max(aug_whole), file=conf.log_writter)
        # print("pred_restor:", restor_out.shape, torch.min(restor_out),torch.max(restor_out), file=conf.log_writter)
        # print("gt_whole: ", gt_whole.shape, torch.min(gt_whole),torch.max(gt_whole), file=conf.log_writter)
        # print("pred_order: ", order_predciton.shape, torch.min(order_predciton),torch.max(order_predciton), file=conf.log_writter)
        # print("randperm: ", randperm.shape, torch.min(randperm),torch.max(randperm), file=conf.log_writter)
        # print("img path: ", batch['image_meta_dict']['filename_or_obj'], file=conf.log_writter)


        # save_image(aug_whole, gt_whole,batch['image_meta_dict']['filename_or_obj'],conf)
        # exit(0)

        order_loss = mse_loss(order_predciton_s, randperm)
        restor_loss = mse_loss(restor_out_s,gt_whole)
        global_loss = mse_loss(global_embd_s, global_embd_t)

        loss = 0
        if "order" in conf.losses:
            loss +=order_loss
            #print("order_loss: ", order_loss.item(), file=conf.log_writter)
            order_losses.update(order_loss.item(), B)

        if "recovery" in conf.losses:
            loss +=restor_loss
            #print("restor_loss: ", restor_loss.item(), file=conf.log_writter)
            restor_losses.update(restor_loss.item(), B)
            loss +=global_loss
            #print("ps_global_consistency_loss: ", global_loss.item(), file=conf.log_writter)
            global_con_losses.update(global_loss.item(), B)
        

        

        # loss = (order_loss + restor_loss)

        conf.log_writter.flush()

        if not math.isfinite( loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), file=conf.log_writter)
            sys.exit(1)
        # update metric
        losses.update(loss.item(), B)

        optimizer.zero_grad()
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=None,
                    parameters=student.parameters(), create_graph=is_second_order)
        with torch.no_grad():
            m = momentum_schedule[current_it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        torch.cuda.synchronize()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        current_it+=1

        # print info
        if (idx + 1) % 10 == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lr {lr}\t'
                  'Order loss {orderloss.val:.3f} ({orderloss.avg:.3f})\t'
                  'Restor loss {restorloss.val:.3f} ({restorloss.avg:.3f})\t'                  
                  'PS Global Consistency loss {global_cons_loss.val:.3f} ({global_cons_loss.avg:.3f})\t'
                  'Total loss {ttloss.val:.3f} ({ttloss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, ttloss=losses, restorloss=restor_losses, orderloss=order_losses,global_cons_loss = global_con_losses,
                lr=optimizer.param_groups[0]['lr']), file=conf.log_writter)
            conf.log_writter.flush()
            if conf.debug_mode:
                break

    wandb.log({"order loss": order_losses.avg,
               "restor loss": restor_losses.avg,
               "ps global consistency loss": global_con_losses.avg})

    return losses.avg, current_it



def test(valid_loader, student, teacher, conf):
    student.eval()
    teacher.eval()
    mse_loss =nn.MSELoss()

    student_order_losses = AverageMeter()
    student_restor_losses = AverageMeter()
    student_global_con_losses = AverageMeter()

    with torch.no_grad():
        for idx,  batch in enumerate(valid_loader):
            aug_whole = batch["aug_image"].cuda(non_blocking=True)
            gt_whole = batch["image"].cuda(non_blocking=True)
            randperm = batch["permutation"].float().cuda(non_blocking=True)

            if checkNan(gt_whole, aug_whole, batch, conf):
                print("*********************** Skip in Validation ******************************", file=conf.log_writter)
                continue

            B, N, C = randperm.shape
            randperm = randperm.reshape(B * N, C)

            order_predciton_s, restor_out_s, global_emb_s = student(aug_whole, use_consistency=False)
            order_predciton_t, restor_out_t, global_emb_t = teacher(aug_whole, use_consistency=False)

            global_embd_s = F.normalize(global_emb_s, p=2.0, dim=1, eps=1e-12,out=None)  # embedding1:[B,196,1024], global_embd1:[B,8192]
            global_embd_t = F.normalize(global_emb_t, p=2.0, dim=1, eps=1e-12, out=None)


            student_restor_losses.update(mse_loss(restor_out_s, gt_whole).item(), B)
            student_order_losses.update(mse_loss(order_predciton_s, randperm).item(), B)
            student_global_con_losses.update(mse_loss(global_embd_s, global_embd_t).item(), B)
            if conf.debug_mode:
                break

    wandb.log({"student test restor loss": student_restor_losses.avg,
               "student test patch order loss": student_order_losses.avg,
               "student test ps global consistency loss": student_global_con_losses.avg,

               })

    return student_restor_losses.avg, student_order_losses.avg, student_global_con_losses.avg




def main(conf):

    wandb.login()
    with wandb.init(project=conf.method, config = conf):

        student, optimizer, loss_scaler, start_epoch, current_it = build_model(conf)
        teacher, _, _, _,_ = build_model(conf)
        for param_q, param_k in zip(student.module.parameters(), teacher.parameters()):
            param_k.data.mul_(0).add_(param_q.detach().data)
        print(student, file=conf.log_writter)
        # there is no back propagation through the teacher, so no need for gradients
        for p in teacher.parameters():
            p.requires_grad = False
        consistency_train_loader,level_ps_train_loader , valid_loader = get_loader(conf)

        momentum_schedule = cosine_scheduler(0.999, 1, conf.epochs, len(consistency_train_loader) + len(level_ps_train_loader))

        wandb.watch((teacher, student), criterion=None, log="all", log_freq=5, log_graph=True)

        for epoch in range(start_epoch, conf.epochs + 1):
            time1 = time.time()

            lr_ = step_decay(epoch,conf)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            loss, current_it = train_level_ps(level_ps_train_loader, student,teacher, momentum_schedule, optimizer , epoch,loss_scaler, conf, current_it)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1),file = conf.log_writter)
            # tensorboard logger
            print('order recovery loss: {}@Epoch: {}'.format(loss,epoch),file = conf.log_writter)
            print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'],epoch),file = conf.log_writter)
            time1 = time.time()

            loss, current_it = train_global_local_consistency(consistency_train_loader, student, teacher, momentum_schedule, optimizer, epoch, loss_scaler, conf, current_it)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1),file = conf.log_writter)
            print('global loss: {}@Epoch: {}'.format(loss,epoch),file = conf.log_writter)
            print('learning_rate: {},{}'.format(optimizer.param_groups[0]['lr'],epoch),file = conf.log_writter)
            conf.log_writter.flush()
            conf.log_writter.flush()
            if epoch % 5 == 0 or epoch == 1:

                save_file = os.path.join(conf.model_path, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                print('------validation-----', file=conf.log_writter)
                save_student_teacher_level_ps_model(student, teacher, optimizer, loss_scaler, epoch, current_it, save_file, conf.log_writter)
                student_restor_losses, student_order_losses, student_global_con_losses = test(valid_loader, student, teacher, conf)

                print('Student Patch Order Loss: {}. Student Restoration loss: {}. PS Global Consistency loss:{}'.format(student_order_losses, student_restor_losses, student_global_con_losses), file=conf.log_writter)

                conf.log_writter.flush()
                if conf.debug_mode:
                    break



        # save the last model
        save_file = os.path.join(conf.model_path, 'last.pth')
        save_student_teacher_level_ps_model(student, teacher, optimizer, loss_scaler, epoch,current_it, save_file,conf.log_writter)
        student_restor_losses, student_order_losses, student_global_con_losses = test(valid_loader, student, teacher,conf)

        print('Student Patch Order Loss: {}. Student Restoration loss: {}. PS Global Consistency loss:{}'.format(student_order_losses, student_restor_losses, student_global_con_losses), file=conf.log_writter)

        conf.log_writter.flush()


if __name__ == '__main__':
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = POPAR_3D_Org(args)
    torch.backends.cudnn.benchmark = True
    torch.autograd.set_detect_anomaly(True)
    config.display()
    main(config)