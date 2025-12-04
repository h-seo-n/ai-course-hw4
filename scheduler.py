import torch

def MultiStepLR(param):
    milestones = param.get("milestones", [])
    gamma = param.get("gamma", [])

    milestones = sorted(milestones)

    def lr_lambda(epoch):
        count = 0
        for m in milestones:
            if epoch >= m:
                count += 1
        return gamma ** count

    return lr_lambda

# def YourLRScheduler(param):
#     def lr_lambda:
#         raise NotImplementedError
#     return lr_lambda

def LinearDecayLR(param):
    total_epochs = param.get("total_epochs", 100)

    def lr_lambda(epoch):
        return max(0.0, 1 - (epoch / total_epochs))

    return lr_lambda

def IncreaseDecayLR(param):
    warmup_epochs = param.get("warmup_epochs", 5)
    gamma = param.get("gamma", 0.95)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        else:
            return gamma ** (epoch - warmup_epochs)

    return lr_lambda


def CycleLR(param):
    cycle_size = param.get("cycle_size", 10)
    min_lr_factor = param.get("min_lr_factor", 0.1)

    def lr_lambda(epoch):
        cycle_pos = epoch % cycle_size
        half = cycle_size / 2

        if cycle_pos < half:
            # increase
            return min_lr_factor + (1 - min_lr_factor) * (cycle_pos / half)
        else:
            # decrease
            return min_lr_factor + (1 - min_lr_factor) * ((cycle_size - cycle_pos) / half)

    return lr_lambda

scheduler_func_dict = {
    "MultiStepLR": MultiStepLR,
    "LinearDecayLR": LinearDecayLR,
    "IncreaseDecayLR": IncreaseDecayLR,
    "CycleLR": CycleLR,
}

def get_scheduler(name, param, optimizer):
    lr_lambda = scheduler_func_dict[name](param)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler