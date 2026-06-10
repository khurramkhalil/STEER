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
        
        # --- Publication Metrics (No Grad) ---
        with torch.no_grad():
            # 1. Trajectory Analysis (Step-wise)
            # violations: (batch, steps)
            for t in range(steps):
                metrics[f"steer/viol_step_{t}"] = violations[:, t].mean()
                metrics[f"steer/prog_step_{t}"] = progress[:, t].mean()
                if t > 0:
                    metrics[f"steer/stab_step_{t}"] = stability[:, t].mean()

            # 2. Validity Rate (Final Step)
            # Check if violations are effectively zero (< 0.01)
            is_valid = (violations[:, -1] < 0.01).float()
            metrics["steer/validity_rate"] = is_valid.mean()

            # 3. Convergence Step
            # Find first step where stability remains low for all subsequent steps
            # This is complex to define perfectly, simplified: first step where stab < epsilon
            # stability: (batch, steps)
            # mask: (batch, steps) where stab < epsilon
            stab_mask = (stability < self.properties.epsilon_stab)
            # We want the first index where it becomes stable and STAYS stable? 
            # For now, just first step < epsilon is a good proxy for "settling"
            # Note: stability[:, 0] is 0 by definition, so ignore t=0
            if steps > 1:
                # (batch, steps-1)
                valid_stab = stab_mask[:, 1:]
                # argmax gives first index of True, but if all False it gives 0.
                # We need to handle "never converged" case.
                # Add a column of True at the end to ensure we find an index
                sentinel = torch.ones(batch_size, 1, device=trajectory.device, dtype=torch.bool)
                valid_stab_ext = torch.cat([valid_stab, sentinel], dim=1)
                converged_idx = torch.argmax(valid_stab_ext.int(), dim=1) + 1 # +1 because we skipped t=0
                
                # If index is steps-1 (the sentinel), it implies it never converged before the end
                # We can filter those out or just report mean
                metrics["steer/convergence_step"] = converged_idx.float().mean()
                metrics["steer/compute_savings"] = 1.0 - (converged_idx.float().mean() / steps)

            metrics.update({
                "steer/loss": total_loss.detach(),
                "steer/rho_improve": robustness["rho_improve"].mean().detach(),
                "steer/rho_valid": robustness["rho_valid"].mean().detach(),
                "steer/rho_converge": robustness["rho_converge"].mean().detach(),
                "steer/viol_final": violations[:, -1].mean().detach(),
                "steer/prog_final": progress[:, -1].mean().detach(),
            })
        
        return total_loss, metrics
