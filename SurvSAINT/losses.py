import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor

class CoxPHLoss(nn.Module):
    """
    Cox Proportional Hazards Loss Function.

    This loss is based on the partial likelihood of the Cox model.
    It assumes the input log_h (log hazard) is sorted by descending event times.

    Arguments:
        eps (float): A small value to prevent division by zero in log operations.
    """

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, log_h: Tensor, y_gts: Tensor) -> Tensor:
        """
        Compute the negative log partial likelihood for CoxPH.

        Args:
            log_h (Tensor): Log hazard predictions from the model. Shape [batch_size, 1].
            y_gts (Tensor): A tensor containing durations and event indicators.
                            Expected shape: [batch_size, 2], where
                            y_gts[:, 0] = durations, y_gts[:, 1] = events.

        Returns:
            Tensor: CoxPH loss.
        """
        durations, events = y_gts[:, 0], y_gts[:, 1]  # Extract durations and events

        # Sort by decreasing survival time
        idx = durations.argsort(descending=True)
        log_h = log_h[idx].view(-1)
        events = events[idx].view(-1)

        # Compute cumulative sum of exp(log_hazard)
        risk_set_sum = torch.logcumsumexp(log_h, dim=0)

        # Compute log likelihood
        loss = - (log_h - risk_set_sum) * events
        return loss.sum() / events.sum()  # Normalize by the number of events

def pair_rank_mat(durations: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise rank matrix for DeepHit's rank loss.

    Arguments:
        durations (torch.Tensor): [B] durations for a batch
        events (torch.Tensor): [B] event indicators (1 for event, 0 for censored)

    Returns:
        rank_mat (torch.Tensor): [B, B] matrix where rank_mat[i, j] = 1
                                  if T_i < T_j and E_i == 1, else 0
    """
    durations = durations.view(-1)
    events = events.view(-1)
    n = durations.size(0)

    # Create a matrix: T_i < T_j
    diff_matrix = durations.unsqueeze(1) < durations.unsqueeze(0)  # shape [B, B]
    # Only use if i experienced the event
    event_mask = events.unsqueeze(1).bool()                        # shape [B, 1]

    # Combine to create rank_mat
    rank_mat = (diff_matrix & event_mask).float()  # final shape [B, B]
    return rank_mat

class DeepHitSingleLoss(nn.Module):
    def __init__(self, alpha=0.5, sigma=0.1, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.sigma = sigma
        self.reduction = reduction

    def forward(self, phi, idx_durations, events, rank_mat):
        nll = self.nll_pmf(phi, idx_durations, events)
        rank = self.rank_loss(phi, idx_durations, events, rank_mat)
        return self.alpha * nll + (1. - self.alpha) * rank

    def pad_col(self, x):
        return F.pad(x, (1, 0))

    def nll_pmf(self, phi, idx_durations, events, epsilon=1e-7):
        idx_durations = idx_durations.view(-1, 1)
        phi = self.pad_col(phi)
        gamma = phi.max(1)[0]
        cumsum = phi.sub(gamma.view(-1, 1)).exp().cumsum(1)
        sum_ = cumsum[:, -1]
        part1 = phi.gather(1, idx_durations).view(-1).sub(gamma).mul(events)
        part2 = - sum_.relu().add(epsilon).log()
        part3 = sum_.sub(cumsum.gather(1, idx_durations).view(-1)).relu().add(epsilon).log().mul(1. - events)
        return - part1.add(part2).add(part3).mean()

    def rank_loss(self, phi, idx_durations, events, rank_mat):
        pmf = self.pad_col(phi).softmax(1)
        y = torch.zeros_like(pmf).scatter(1, idx_durations.view(-1, 1), 1.)
        return self._rank_loss(pmf, y, rank_mat)

    def _rank_loss(self, pmf, y, rank_mat):
        rank = (pmf.unsqueeze(1) - pmf.unsqueeze(0)) * rank_mat.unsqueeze(-1)
        return torch.sum(rank.pow(2)) / rank_mat.sum().clamp(min=1).float()

class CombinedLossMonotonic(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        """
        Combined loss for monotonic hazard outputs.
        Args:
            alpha: weight for ranking loss
            beta: weight for Brier score
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, raw_output, durations, events, eval_times):
        # Monotonic hazard NLL
        nll_loss, survival, hazard_increments = monotonic_hazard_loss(raw_output, durations, events, eval_times)

        # Pairwise ranking loss
        rank_loss = pairwise_ranking_loss_from_survival(survival, durations, events)

        # Brier score loss
        brier_loss = brier_score_loss_from_survival(survival, durations, events, eval_times)

        return nll_loss + self.alpha * rank_loss + self.beta * brier_loss, survival, hazard_increments

def monotonic_hazard_loss(raw_output, durations, events, eval_times):
    """
    Negative log-likelihood for monotonic hazard parameterization.
    Predicts hazard increments via softplus and integrates to survival.

    Args:
        raw_output: (B, T) raw logits from model
        durations: (B,) event/censoring times
        events: (B,) binary event indicators
        eval_times: (T,) discrete time grid
    Returns:
        nll: scalar loss
        survival: (B, T) survival probabilities
        hazard_increments: (B, T) hazard increments
    """
    eps = 1e-8
    B, T = raw_output.shape
    device = raw_output.device

    # Ensure eval_times tensor
    if not isinstance(eval_times, torch.Tensor):
        eval_times = torch.tensor(eval_times, dtype=torch.float32, device=device)
    else:
        eval_times = eval_times.to(device)

    # Convert to hazard increments (≥ 0)
    hazard_increments = F.softplus(raw_output) + eps

    # Cumulative hazard & survival
    cumulative_hazard = torch.cumsum(hazard_increments, dim=1)  # (B, T)
    survival = torch.exp(-cumulative_hazard)  # (B, T), monotonic decreasing

    # Event index
    idx = torch.searchsorted(eval_times, durations, right=True).clamp(0, T - 1)

    # Hazard at event time & cumulative hazard
    event_hazard = hazard_increments.gather(1, idx.unsqueeze(1)).squeeze(1)
    cum_hazard_at_event = cumulative_hazard.gather(1, idx.unsqueeze(1)).squeeze(1)

    # NLL: for event: -log(hazard) + cum_hazard, for censor: cum_hazard only
    nll = -events * torch.log(event_hazard + eps) + cum_hazard_at_event

    return nll.mean(), survival, hazard_increments


def brier_score_loss_from_survival(survival, durations, events, eval_times):
    """
    Time-dependent Brier score for survival predictions.
    """
    B, T = survival.shape
    device = survival.device
    if not isinstance(eval_times, torch.Tensor):
        eval_times = torch.tensor(eval_times, dtype=torch.float32, device=device)
    else:
        eval_times = eval_times.to(device)

    # Index of event/censoring time
    idx = torch.searchsorted(eval_times, durations, right=True).clamp(0, T - 1)

    # Ground truth survival curve (1 if alive at t, 0 if event at idx)
    gt = torch.ones_like(survival)
    gt[events.bool(), idx[events.bool()]] = 0.0

    return ((survival - gt) ** 2).mean()


def pairwise_ranking_loss_from_survival(survival, durations, events, reduction="mean"):
    """
    Pairwise ranking loss using final risk score (1 - survival at last time).
    """
    risk_score = 1 - survival[:, -1]  # higher risk = lower survival at last time

    dur_i = durations.view(-1, 1)
    dur_j = durations.view(1, -1)
    evt_i = events.view(-1, 1)

    valid_pairs = (dur_i < dur_j) & (evt_i == 1)
    margin_matrix = risk_score.view(1, -1) - risk_score.view(-1, 1)
    loss_matrix = F.softplus(margin_matrix)

    total_loss = loss_matrix[valid_pairs].sum()
    count = valid_pairs.sum()

    if count == 0:
        return torch.tensor(0.0, device=survival.device)
    return total_loss / count if reduction == "mean" else total_loss
