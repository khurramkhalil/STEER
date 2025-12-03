import torch
import torch.nn.functional as F

class SudokuSignals:
    def __init__(self):
        # Pre-compute indices for rows, cols, boxes
        self.rows = self._get_row_indices()
        self.cols = self._get_col_indices()
        self.boxes = self._get_box_indices()
        self.all_units = self.rows + self.cols + self.boxes
        
        # Indices as a single tensor for efficient gathering: [27, 9]
        # 27 units (9 rows + 9 cols + 9 boxes), each has 9 cells
        self.unit_indices = torch.tensor(self.all_units, dtype=torch.long)

    def _get_row_indices(self):
        return [[r * 9 + c for c in range(9)] for r in range(9)]

    def _get_col_indices(self):
        return [[r * 9 + c for r in range(9)] for c in range(9)]

    def _get_box_indices(self):
        indices = []
        for br in range(3):
            for bc in range(3):
                box = []
                for r in range(3):
                    for c in range(3):
                        box.append((br * 3 + r) * 9 + (bc * 3 + c))
                indices.append(box)
        return indices

    def to(self, device):
        self.unit_indices = self.unit_indices.to(device)
        return self

    def compute_violation_score(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the violation score for a batch of Sudoku grids.
        
        Args:
            probs: Tensor of shape (batch_size, 81, 11). 
                   11 classes: 0=PAD, 1=Blank, 2..10=Digits 1..9.
                   
        Returns:
            violation_score: Tensor of shape (batch_size,)
        """
        batch_size = probs.shape[0]
        
        # We only care about digits 1-9 (indices 2-10 in vocab)
        # Shape: (batch_size, 81, 9)
        digit_probs = probs[:, :, 2:] 
        
        # Gather probabilities for each unit
        # unit_indices shape: (27, 9)
        # gathered_probs shape: (batch_size, 27, 9, 9) 
        # (batch, unit, cell_in_unit, digit)
        
        # Expand indices for batch gathering
        flat_indices = self.unit_indices.view(-1) # (243,)
        gathered_probs = digit_probs[:, flat_indices, :].view(batch_size, 27, 9, 9)
        
        # Sum probabilities for each digit within each unit
        # Shape: (batch_size, 27, 9) - sum over cell_in_unit dim
        digit_counts = gathered_probs.sum(dim=2)
        
        # A valid unit has exactly one of each digit. 
        # Violation is excess probability mass > 1.
        # We want to penalize if sum(prob(digit)) > 1.
        # Using ReLU(count - 1) captures "more than one instance"
        
        violations = F.relu(digit_counts - 1.0)
        
        # Sum over all units and all digits
        total_violations = violations.sum(dim=(1, 2))
        
        return total_violations

    def compute_progress_score(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the progress score (number of blanks).
        
        Args:
            probs: Tensor of shape (batch_size, 81, 11).
            
        Returns:
            progress_score: Tensor of shape (batch_size,)
        """
        # Index 1 is the "Blank" token
        blank_probs = probs[:, :, 1]
        
        # Sum of blank probabilities across the grid
        return blank_probs.sum(dim=1)

    def compute_stability_score(self, current_probs: torch.Tensor, prev_probs: torch.Tensor) -> torch.Tensor:
        """
        Computes stability score (difference from previous step).
        Using MSE for smoothness.
        
        Args:
            current_probs: (batch_size, 81, 11)
            prev_probs: (batch_size, 81, 11)
            
        Returns:
            stability_score: (batch_size,)
        """
        # Mean Squared Error summed over grid and vocab
        diff = (current_probs - prev_probs) ** 2
        return diff.sum(dim=(1, 2))
