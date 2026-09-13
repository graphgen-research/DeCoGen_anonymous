import os
import time
import torch
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.utils import clip_grad_value_
from torch.utils.tensorboard import SummaryWriter

from utils import save_model, load_model, get_model_attribute
from graphgen.train import evaluate_loss as eval_loss_dfscode_rnn
from baselines.graph_rnn.train import evaluate_loss as eval_loss_graph_rnn
from baselines.dgmg.train import evaluate_loss as eval_loss_dgmg


def _hierarchy_penalty(args, node_level, t1, t2, lengths):
    """
    Compute the discovery-edge hierarchy penalty independently for each
    graph, then average the graph-level penalties across the batch.
    """
    B, T = node_level.shape
    device = node_level.device
    lengths_dev = lengths.to(device)

    t1c = t1.clamp(min=0, max=T - 1)
    t2c = t2.clamp(min=0, max=T - 1)

    step_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

    # lengths includes the EOS position, so exclude the final valid position.
    edge_lengths = (lengths_dev - 1).clamp_min(0)
    step_mask = step_idx < edge_lengths.unsqueeze(1)

    valid_idx = (
        (t1c < lengths_dev.unsqueeze(1))
        & (t2c < lengths_dev.unsqueeze(1))
    )
    forward_ok = t2c > t1c
    candidate = step_mask & valid_idx & forward_ok

    BIG = 10**9
    min_pos = torch.full(
        (B, T),
        BIG,
        device=device,
        dtype=torch.long,
    )

    for s in range(T):
        mask_s = candidate[:, s]

        if not mask_s.any():
            continue

        k = t2c[:, s]
        cur = min_pos.gather(1, k.view(B, 1)).squeeze(1)
        upd = torch.minimum(cur, torch.full_like(cur, s))

        min_pos.scatter_(
            1,
            k.view(B, 1),
            torch.where(mask_s, upd, cur).view(B, 1),
        )

    discovery_mask = candidate & (
        step_idx == min_pos.gather(1, t2c)
    )
    full_mask = discovery_mask.float()

    discovery_step_v = min_pos.gather(1, t2c).clamp(max=T - 1)
    discovery_step_u = min_pos.gather(1, t1c).clamp(max=T - 1)

    lvl_v = node_level.gather(1, discovery_step_v)
    lvl_u = node_level.gather(1, discovery_step_u)

    # The root is never discovered as a target, so assign its defined depth 0.
    is_root_parent = t1c == 0
    lvl_u = torch.where(
        is_root_parent,
        torch.zeros_like(lvl_u),
        lvl_u,
    )

    delta = lvl_v - lvl_u  # positive means going deeper

    alpha = float(getattr(args, "alpha", 0.25))
    margin_pen = torch.clamp(alpha - delta, min=0.0)

    # Exclude all non-discovery-edge positions.
    margin_pen = margin_pen * full_mask

    # Calculate L_depth(G) independently for each graph.
    penalty_per_graph = margin_pen.sum(dim=1)
    num_discoveries_per_graph = full_mask.sum(dim=1)

    depth_loss_per_graph = torch.where(
        num_discoveries_per_graph > 0,
        penalty_per_graph
        / num_discoveries_per_graph.clamp_min(1.0),
        torch.zeros_like(penalty_per_graph),
    )

    # Average graph-level losses across the batch.
    hierarchy_penalty = depth_loss_per_graph.mean()

    return hierarchy_penalty


def evaluate_loss(args, model, data, feature_map):
    """
    Unified loss interface returning (total, recon, hier).
    For GraphRNN/DGMG, hier=0.
    For DFScodeRNN, graphgen/train.py runs the single RNN forward pass and
    returns recon_loss + node_level output needed for the hierarchy penalty,
    avoiding a second forward pass.

    args.position_control     -> use raw generation-step control (Exp 1e)
    (default)                 -> DeCoGen's original soft margin loss
    """
    if args.note == 'GraphRNN':
        total = eval_loss_graph_rnn(args, model, data, feature_map)
        zero  = torch.tensor(0.0, device=args.device)
        return total, total, zero

    elif args.note == 'DFScodeRNN':
        recon_loss, node_level, t1, t2, lengths = eval_loss_dfscode_rnn(
            args, model, data, feature_map)

        hier_loss = torch.tensor(0.0, device=args.device)
        if getattr(args, 'lambda_hier', 0.0) > 0:
            penalty = _hierarchy_penalty(args, node_level, t1, t2, lengths)
            hier_loss = args.lambda_hier * penalty 

        total_loss = recon_loss + hier_loss
        return total_loss, recon_loss, hier_loss

    elif args.note == 'DGMG':
        total = eval_loss_dgmg(model, data)
        zero  = torch.tensor(0.0, device=args.device)
        return total, total, zero

    else:
        raise ValueError(f"Unknown model note: {args.note}")


def train_epoch(epoch, args, model, dataloader_train, optimizer,
                scheduler, feature_map, summary_writer=None):
    for _, net in model.items():
        net.train()

    batch_count = len(dataloader_train)
    total_sum = recon_sum = hier_sum = 0.0

    for batch_id, data in enumerate(dataloader_train):
        for _, net in model.items():
            net.zero_grad()

        total_loss, recon_loss, hier_loss = evaluate_loss(args, model, data, feature_map)
        total_loss.backward()

        total_sum += float(total_loss.item())
        recon_sum += float(recon_loss.item()) if torch.is_tensor(recon_loss) else float(recon_loss)
        hier_sum  += float(hier_loss.item())  if torch.is_tensor(hier_loss)  else float(hier_loss)

        if getattr(args, "gradient_clipping", False):
            for _, net in model.items():
                params_with_grads = [p for p in net.parameters() if p.grad is not None]
                if params_with_grads:
                    clip_grad_value_(params_with_grads, 1.0)

        for _, opt in optimizer.items():
            opt.step()

        for _, sched in scheduler.items():
            sched.step()

        if getattr(args, "log_tensorboard", False) and summary_writer:
            global_step = batch_id + batch_count * epoch
            summary_writer.add_scalar(
                f'{args.note} {args.graph_type} Loss/train_batch_total',
                total_loss.item(), global_step)
            summary_writer.add_scalar(
                f'{args.note} {args.graph_type} Loss/train_batch_recon',
                float(recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss),
                global_step)
            summary_writer.add_scalar(
                f'{args.note} {args.graph_type} Loss/train_batch_hier',
                float(hier_loss.item() if torch.is_tensor(hier_loss) else hier_loss),
                global_step)

    denom = max(batch_count, 1)
    return total_sum / denom, recon_sum / denom, hier_sum / denom


def test_data(args, model, dataloader, feature_map):
    for _, net in model.items():
        net.eval()

    batch_count = len(dataloader)
    total_sum = recon_sum = hier_sum = 0.0

    with torch.no_grad():
        for _, data in enumerate(dataloader):
            total_loss, recon_loss, hier_loss = evaluate_loss(args, model, data, feature_map)
            total_sum += float(total_loss.item())
            recon_sum += float(recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss)
            hier_sum  += float(hier_loss.item()  if torch.is_tensor(hier_loss)  else hier_loss)

    denom = max(batch_count, 1)
    return total_sum / denom, recon_sum / denom, hier_sum / denom


def train(args, dataloader_train, model, feature_map, dataloader_validate=None):
    best_val_loss  = float('inf')
    best_val_epoch = None
    bad_epochs     = 0

    # Per-epoch wall-clock timing (training-only, excludes validation) -
    # used to report the practical training overhead of the hierarchy
    # penalty relative to a plain reconstruction-only run (e.g.
    # lambda_hier=0), as requested in review.
    epoch_times = []

    optimizer = {}
    for name, net in model.items():
        optimizer['optimizer_' + name] = optim.Adam(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=args.lr,
            weight_decay=5e-5
        )

    scheduler = {}
    for name, net in model.items():
        scheduler['scheduler_' + name] = MultiStepLR(
            optimizer['optimizer_' + name],
            milestones=args.milestones,
            gamma=args.gamma
        )

    if getattr(args, "load_model", False):
        load_model(args.load_model_path, args.device, model, optimizer, scheduler)
        print('Model loaded from', args.load_model_path)
        epoch = get_model_attribute('epoch', args.load_model_path, args.device)
    else:
        epoch = 0

    if getattr(args, "log_tensorboard", False):
        writer = SummaryWriter(
            log_dir=os.path.join(args.tensorboard_path, f"{args.fname}_{args.time}"),
            flush_secs=5
        )
    else:
        writer = None

    while epoch < args.epochs:
        epoch_start = time.time()
        train_tot, train_rec, train_hier = train_epoch(
            epoch, args, model, dataloader_train, optimizer, scheduler, feature_map, writer
        )
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)

        epoch += 1

        running_avg = sum(epoch_times) / len(epoch_times)

        if getattr(args, "log_tensorboard", False) and writer:
            writer.add_scalar(f'{args.note} {args.graph_type} Loss/train_total',  train_tot,  epoch)
            writer.add_scalar(f'{args.note} {args.graph_type} Loss/train_recon',  train_rec,  epoch)
            writer.add_scalar(f'{args.note} {args.graph_type} Loss/train_hier',   train_hier, epoch)
            writer.add_scalar(f'{args.note} {args.graph_type} Time/epoch_seconds', epoch_elapsed, epoch)
            writer.add_scalar(f'{args.note} {args.graph_type} Time/epoch_seconds_running_avg', running_avg, epoch)
        else:
            print(f'Epoch: {epoch}/{args.epochs}, '
                  f'train total: {train_tot:.6f} '
                  f'(recon: {train_rec:.6f}, hier: {train_hier:.6f}), '
                  f'epoch time: {epoch_elapsed:.2f}s (running avg: {running_avg:.2f}s)')

        if getattr(args, "save_model", False) and epoch != 0 and epoch % args.epochs_save == 0:
            save_model(epoch, args, model, optimizer, scheduler, feature_map=feature_map)
            print(f'Model Saved - Epoch: {epoch}/{args.epochs}, '
                  f'train total: {train_tot:.6f} '
                  f'(recon: {train_rec:.6f}, hier: {train_hier:.6f}), '
                  f'epoch time: {epoch_elapsed:.2f}s (running avg: {running_avg:.2f}s)')

        if dataloader_validate is not None:
            should_validate = (
                getattr(args, "early_stop", False) or
                epoch % getattr(args, "epochs_validate", 1) == 0
            )

            if should_validate:
                val_tot, val_rec, val_hier = test_data(
                    args, model, dataloader_validate, feature_map)

                if getattr(args, "log_tensorboard", False) and writer:
                    writer.add_scalar(f'{args.note} {args.graph_type} Loss/validate_total', val_tot,  epoch)
                    writer.add_scalar(f'{args.note} {args.graph_type} Loss/validate_recon', val_rec,  epoch)
                    writer.add_scalar(f'{args.note} {args.graph_type} Loss/validate_hier',  val_hier, epoch)
                else:
                    print(f'Epoch: {epoch}/{args.epochs}, '
                          f'validation total: {val_tot:.6f} '
                          f'(recon: {val_rec:.6f}, hier: {val_hier:.6f})')

                if getattr(args, "early_stop", False):
                    if val_tot < best_val_loss:
                        best_val_loss  = val_tot
                        best_val_epoch = epoch
                        bad_epochs     = 0
                        print(f"New best validation total: {best_val_loss:.6f} at epoch {best_val_epoch}")
                    else:
                        bad_epochs += 1
                        print(f"No improvement for {bad_epochs}/{getattr(args, 'patience', 10)} epochs")

                        if bad_epochs >= getattr(args, 'patience', 10):
                            print("Early stopping triggered.")
                            break

    save_model(epoch, args, model, optimizer, scheduler, feature_map=feature_map)

    mean_epoch_time = sum(epoch_times) / len(epoch_times) if epoch_times else 0.0
    total_train_time = sum(epoch_times)

    print(f'Training complete. Final epoch: {epoch}/{args.epochs}, '
          f'last train total: {train_tot:.6f} '
          f'(recon: {train_rec:.6f}, hier: {train_hier:.6f})')
    print(f'Timing summary: {len(epoch_times)} epochs, '
          f'mean epoch time: {mean_epoch_time:.2f}s, '
          f'total training time: {total_train_time:.2f}s '
          f'({total_train_time / 3600:.2f}h)')

    if getattr(args, "early_stop", False) and dataloader_validate is not None and best_val_epoch is not None:
        print(f"Best validation total: {best_val_loss:.6f} at epoch {best_val_epoch}")
    elif dataloader_validate is None:
        print("No validation data provided.")

    if writer is not None:
        writer.close()

    return epoch_times
