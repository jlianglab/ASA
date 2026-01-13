import torch
import torch.nn as nn



class PixelShuffle3d(nn.Module):
    '''
    This class is a 3d version of pixelshuffle.
    '''
    def __init__(self, scale):
        '''
        :param scale: upsample scale
        '''
        super().__init__()
        self.scale = scale

    def forward(self, input):
        batch_size, channels, in_depth, in_height, in_width = input.size()
        nOut = channels // self.scale ** 3

        out_depth = in_depth * self.scale
        out_height = in_height * self.scale
        out_width = in_width * self.scale

        input_view = input.contiguous().view(batch_size, nOut, self.scale, self.scale, self.scale, in_depth, in_height, in_width)

        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        return output.view(batch_size, nOut, out_depth, out_height, out_width)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        #print(target, pred, batch_size, correct, res)
        return res

def save_model(model, optimizer, conf, global_step, save_file):
    print('==> Saving...',file=conf.log_writter)
    state = {
        "global_step": global_step,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict()}
    torch.save(state, save_file)
    del state


def save_student_teacher_model(student, teacher, optimizer,loss_scaler, epoch, save_file, log_writter):
    print('==> Saving...', file=log_writter)
    state = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_scaler':loss_scaler.state_dict(),
        'epoch': epoch,
    }

    torch.save(state, save_file)
    del state

def save_student_teacher_level_ps_model(student, teacher, optimizer,loss_scaler, epoch, current_it, save_file, log_writter):
    print('==> Saving...', file=log_writter)
    state = {
        'student': student.state_dict(),
        'teacher': teacher.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_scaler':loss_scaler.state_dict(),
        'epoch': epoch,
        'current_it': current_it
    }

    torch.save(state, save_file)
    del state