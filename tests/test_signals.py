import torch
import sys
import os

# Add repo root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steer.signals import SudokuSignals

def test_signals():
    signals = SudokuSignals()
    
    # Create a dummy batch: 2 examples
    # Example 0: Valid Sudoku (all filled, no violations)
    # Example 1: Invalid Sudoku (duplicates in row 0)
    
    batch_size = 2
    vocab_size = 11 # 0=PAD, 1=Blank, 2-10=Digits
    
    # Initialize with one-hot encoding for "Blank" (index 1)
    probs = torch.zeros(batch_size, 81, vocab_size)
    probs[:, :, 1] = 1.0 
    
    # --- Setup Example 0: Valid Sudoku ---
    # Just filling row 0 with 1..9 for simplicity of testing "row" logic
    # To test full validity we'd need a full valid grid, but let's test local logic first.
    # Actually, let's just test that "1..9 in a row" gives 0 violation for that row.
    
    # Row 0: 1, 2, 3, 4, 5, 6, 7, 8, 9
    for c in range(9):
        digit = c + 1
        digit_idx = digit + 1 # 2..10
        probs[0, c, :] = 0
        probs[0, c, digit_idx] = 1.0
        
    # --- Setup Example 1: Invalid Row ---
    # Row 0: 1, 1, 3, 4, 5, 6, 7, 8, 9 (Duplicate 1s)
    for c in range(9):
        digit = c + 1
        if c == 1: digit = 1 # Duplicate 1 at pos 1
        digit_idx = digit + 1
        probs[1, c, :] = 0
        probs[1, c, digit_idx] = 1.0
        
    # Compute Violation Score
    # For Ex 0: Row 0 is valid. Other rows are all blanks (no digit violations).
    # For Ex 1: Row 0 has two 1s. Count(1) = 2. Violation = ReLU(2-1) = 1.
    # Note: Columns and Boxes will also see these digits.
    # Ex 0: Col 0 has '1', others blank. Valid.
    # Ex 1: Col 0 has '1', Col 1 has '1'. Valid (locally).
    
    # Let's just check the values.
    violations = signals.compute_violation_score(probs)
    print(f"Violations: {violations}")
    
    # Ex 0 should be 0 (if we ignore the fact that 0s are blanks and not violations)
    # Wait, blanks don't cause violations in my logic (only digits 1-9).
    # So Ex 0 should have 0 violations.
    # Ex 1 should have > 0 violations (at least from Row 0).
    
    assert violations[0] == 0, f"Expected 0 violations for valid row, got {violations[0]}"
    assert violations[1] > 0, f"Expected >0 violations for invalid row, got {violations[1]}"
    
    # --- Test Progress Score ---
    # Ex 0: 9 cells filled, 72 blanks. Score should be 72.
    # Ex 1: 9 cells filled, 72 blanks. Score should be 72.
    progress = signals.compute_progress_score(probs)
    print(f"Progress: {progress}")
    assert torch.isclose(progress[0], torch.tensor(72.0)), f"Expected 72 blanks, got {progress[0]}"
    
    # --- Test Stability Score ---
    # Create a "next step" where Ex 0 changes one cell
    probs_next = probs.clone()
    # Change Ex 0, Cell 0 from '1' to '2'
    probs_next[0, 0, :] = 0
    probs_next[0, 0, 3] = 1.0 # Digit 2
    
    stability = signals.compute_stability_score(probs_next, probs)
    print(f"Stability: {stability}")
    
    # Ex 0 changed. MSE: (1-0)^2 + (0-1)^2 = 2.
    assert torch.isclose(stability[0], torch.tensor(2.0)), f"Expected stability 2.0, got {stability[0]}"
    # Ex 1 didn't change.
    assert stability[1] == 0, "Expected stability 0 for unchanged"

    print("All tests passed!")

if __name__ == "__main__":
    test_signals()
