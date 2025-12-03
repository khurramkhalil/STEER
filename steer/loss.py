import torch
import torch.nn as nn
import torch.nn.functional as F
from steer.signals import SudokuSignals
from steer.stl import STEERProperties

class STEERLoss(nn.Module):
    def __init__(self, epsilon_viol=0.1, epsilon_stab=0.01):
        super().__init__()
        self.signals = SudokuSignals()
        self.properties = STEERProperties(epsilon_viol=epsilon_viol, epsilon_stab=epsilon_stab)
        
    def forward(self, trajectory: torch.Tensor):
        """
        Args:
            trajectory: (batch, steps, seq_len, vocab)
        
        Returns:
            loss: scalar
            metrics: dict
        """
        batch_size, steps, seq_len, vocab = trajectory.shape
        
        # Ensure signals are on the correct device
        self.signals.to(trajectory.device)
        
        # Compute probabilities for signals
        # Use softmax to get differentiable probabilities
        probs = F.softmax(trajectory, dim=-1) # (batch, steps, seq_len, vocab)
        
        # Flatten batch and steps for efficient signal computation
        flat_probs = probs.view(-1, seq_len, vocab)
        
        # Compute signals
        # Violations: (batch * steps,)
        violations = self.signals.compute_violation_score(flat_probs)
        violations = violations.view(batch_size, steps)
        
        # Progress: (batch * steps,)
        progress = self.signals.compute_progress_score(flat_probs)
        progress = progress.view(batch_size, steps)
        
        # Stability: (batch * steps,)
        # Stability at t is diff between t and t-1.
        # stability[0] is 0.
        stability = torch.zeros(batch_size, steps, device=trajectory.device)
        if steps > 1:
            # Compute diff between t and t-1 for t=1..steps-1
            curr_probs = probs[:, 1:, :, :]
            prev_probs = probs[:, :-1, :, :]
            
            # Reuse signal function but need to reshape
            flat_curr = curr_probs.reshape(-1, seq_len, vocab)
            flat_prev = prev_probs.reshape(-1, seq_len, vocab)
            
            stab_scores = self.signals.compute_stability_score(flat_curr, flat_prev)
            stability[:, 1:] = stab_scores.view(batch_size, steps - 1)
            
        # Compute STL Robustness
        robustness = self.properties.compute_robustness(violations, progress, stability)
        
        # Compute Loss
        # We want to maximize robustness, so minimize ReLU(-rho)
        # Loss = sum(ReLU(-rho))
        
        loss_improve = F.relu(-robustness["rho_improve"]).mean()
        loss_valid = F.relu(-robustness["rho_valid"]).mean()
        loss_converge = F.relu(-robustness["rho_converge"]).mean()
        
        total_loss = loss_improve + loss_valid + loss_converge
        
        metrics = {
            "steer/loss": total_loss.detach(),
            "steer/rho_improve": robustness["rho_improve"].mean().detach(),
            "steer/rho_valid": robustness["rho_valid"].mean().detach(),
            "steer/rho_converge": robustness["rho_converge"].mean().detach(),
            "steer/viol_final": violations[:, -1].mean().detach(),
            "steer/prog_final": progress[:, -1].mean().detach(),
        }
        
        return total_loss, metrics
