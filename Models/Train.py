# # -*- coding: utf-8 -*-
# import copy
# import torch as th
# from tqdm import tqdm
# import torch
# import gc
#
#
# def run_epoch(models,
#               optimizer,
#               loss,
#               x,
#               device,
#               y_true):
#     optimizer.zero_grad()
#     x = x.to(device)
#     y_true = y_true.to(device)
#     with torch.set_grad_enabled(True):
#         y_pred = models.forward(x)
#         loss_value = loss(y_pred, y_true)
#         loss_value.backward()
#         optimizer.step()
#     return y_pred
#
#
# def train(model: torch.nn.Module,
#           loader,
#           loader_val,
#           epochs_count,
#           optimizer,
#           loss,
#           device,
#           score_func,
#           check_out=None,
#           wandb=None,
#           loader_test=None,
#           sh=False,
#           early_stopping_patience=50,
#           scheduler=None,
#           bar=False,
#           log=True):
#     if wandb is not None:
#         # wandb.define_metric("train_loss", summary="min")
#         # wandb.define_metric("val_loss", summary="min")
#         wandb.watch(model)
#     if scheduler is None:
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
#                                                                cooldown=10,
#                                                                patience=10,
#                                                                factor=0.95,
#                                                                verbose=True),
#     history = {"train": [],
#                "val": []}
#     # scheduler = StepLR(optimizer, step_size=500, gamma=0.75)
#
#     best_val_loss = float('inf')
#     best_epoch_i = 0
#     best_model = copy.deepcopy(model)
#     if not log and bar:
#         ebar = tqdm(range(epochs_count))
#     else:
#         ebar = range(epochs_count)
#
#     for epoch in ebar:
#         torch.cuda.empty_cache()
#         gc.collect()
#         model.train()
#         predict_true = []
#         predict_predict = []
#         if log and bar:
#             pbar = tqdm(loader, desc=f"{epoch + 1}/{epochs_count}")
#         else:
#             pbar = loader
#         for x, y_true in pbar:
#             # print(x.shape)
#             y_pred = run_epoch(model,
#                                optimizer,
#                                loss,
#                                x, device, y_true)
#             predict_true.append(y_true.detach())
#             predict_predict.append(y_pred.detach())
#
#         predict_true = th.cat(predict_true)
#         predict_predict = th.cat(predict_predict)
#         score_train = score_func(predict_predict, predict_true)
#         del predict_predict, predict_true
#         predict_true = []
#         predict_predict = []
#         model.eval()
#         with torch.no_grad():
#             for x, y_true in loader_val:
#                 y_true.to(device)
#                 optimizer.zero_grad()
#                 x = x.to(device).detach()
#                 y_pred = model.forward(x)
#                 predict_true.append(y_true.detach())
#                 predict_predict.append(y_pred.detach())
#
#         predict_true = th.cat(predict_true)
#         predict_predict = th.cat(predict_predict)
#         score_val = score_func(predict_predict, predict_true)
#         if wandb is not None:
#             wandb.log({"train_loss": score_train,
#                        "val_loss": score_val})
#         elif log:
#             print(f"\r epoch {epoch + 1} - Train:{score_train} Val:{score_val}")
#
#         val_loss = loss(predict_predict, predict_true)
#         if val_loss < best_val_loss:
#             best_epoch_i = epoch
#             best_val_loss = val_loss
#             best_model = copy.deepcopy(model)
#             if log:
#                 print('new best model')
#         if sh:
#             scheduler.step(val_loss)
#         if check_out(score_val) and check_out(score_train):
#             break
#         history["train"].append(score_train)
#         history["val"].append(score_val)
#
#     if wandb is not None:
#         wandb.log({"best_epoch": best_epoch_i})
#         # scheduler.step()
#
#     return history, best_epoch_i, copy.deepcopy(best_model)
