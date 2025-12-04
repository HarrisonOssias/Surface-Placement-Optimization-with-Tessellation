"""
Comparison Script
=================

Runs side-by-side comparison of Traditional CV vs YOLO detection.
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from evaluation.evaluate_system import SystemEvaluator


def main():
    """Run comparison between methods."""
    print("\n" + "="*70)
    print("BOX DETECTION METHOD COMPARISON")
    print("Traditional Computer Vision vs YOLOv8")
    print("="*70 + "\n")
    
    evaluator = SystemEvaluator()
    
    print("Running evaluation on all test scenarios...")
    evaluations = evaluator.evaluate_all()
    
    if not evaluations:
        print("\n⚠ No test scenarios found")
        print("\nTo create test scenarios:")
        print("  1. Run: python run_placement_system.py")
        print("  2. Capture several box+desk scenarios")
        print("  3. Save screenshots with 'S' key")
        print("  4. Re-run this comparison script")
        return
    
    print(f"\n✓ Evaluated {len(evaluations)} scenarios")
    
    # Generate and save report
    report_file = evaluator.save_report(evaluations)
    
    print(f"\n{'='*70}")
    print("COMPARISON COMPLETE")
    print('='*70)
    print(f"\nFull report: {report_file}")
    print("\nKey findings:")
    
    # Calculate summary stats
    import numpy as np
    
    trad_maes = [e.traditional_detection.mean_absolute_error_cm 
                 for e in evaluations 
                 if e.traditional_detection.mean_absolute_error_cm]
    
    yolo_maes = [e.yolo_detection.mean_absolute_error_cm 
                 for e in evaluations 
                 if e.yolo_detection and e.yolo_detection.mean_absolute_error_cm]
    
    if trad_maes:
        print(f"  • Traditional CV: {np.mean(trad_maes):.2f}cm average error")
    
    if yolo_maes:
        print(f"  • YOLO: {np.mean(yolo_maes):.2f}cm average error")
        improvement = (np.mean(trad_maes) - np.mean(yolo_maes)) / np.mean(trad_maes) * 100
        print(f"  • Improvement: {improvement:.1f}% more accurate with YOLO")
    
    correct = sum(1 for e in evaluations if e.placement_metrics.correct_decision)
    accuracy = correct / len(evaluations) * 100
    print(f"  • Placement decisions: {accuracy:.0f}% accurate")


if __name__ == "__main__":
    main()

