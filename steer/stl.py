import torch

class STEERProperties:
    def __init__(self, epsilon_viol=0.1, epsilon_stab=0.01):
        self.epsilon_viol = epsilon_viol
        self.epsilon_stab = epsilon_stab

    def compute_robustness(self, violations, progress, stability):
        """
        Computes robustness scores for the three STEER properties.
        
        Args:
            violations: (batch, T+1) - Violation scores at each step
            progress: (batch, T+1) - Progress scores at each step
            stability: (batch, T+1) - Stability scores at each step (stability[0] is 0)
            
        Returns:
            robustness: Dict of tensors (batch,)
        """
        T = violations.shape[1] - 1
        
        # 1. Monotonic Progress (Phi_improve)
        # G (prog(t) <= prog(t-1))  <=>  prog(t-1) - prog(t) >= 0
        # Robustness(t) = prog(t-1) - prog(t)
        # Global Robustness = min_{t=1..T} (prog(t-1) - prog(t))
        
        prog_diff = progress[:, :-1] - progress[:, 1:] # (batch, T)
        rho_improve = prog_diff.min(dim=1).values
        
        # 2. Path Validity (Phi_valid)
        # G (viol(t) < eps) <=> eps - viol(t) > 0
        # Robustness(t) = eps - viol(t)
        # Global Robustness = min_{t=0..T} (eps - viol(t))
        
        rho_valid = (self.epsilon_viol - violations).min(dim=1).values
        
        # 3. Eventual Convergence (Phi_converge)
        # F (stab(t) < eps) <=> max_{t=0..T} (eps - stab(t)) > 0
        # Robustness = max_{t=0..T} (eps - stab(t))
        
        rho_converge = (self.epsilon_stab - stability).max(dim=1).values
        
        return {
            "rho_improve": rho_improve,
            "rho_valid": rho_valid,
            "rho_converge": rho_converge
        }
