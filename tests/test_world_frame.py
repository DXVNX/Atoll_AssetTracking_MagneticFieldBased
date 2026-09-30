"""
Sanity-check tests for world_frame.py

Core invariant: rotation is orthogonal, so |mb| must equal |mn| for every sample.
Any violation > ~0.01 uT indicates a sign/axis convention bug rather than sensor noise.

Acceptance criteria:
  |mb| == |mn| within ~1% for at least 95% of test samples.
"""

import sys, os
import numpy as np
import pandas as pd
import glob

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.world_frame import get_rotation_matrix, body_to_world, compute_horizontal_vertical, compute_inclination, add_world_frame


def test_rotation_matrix_orthogonality():
    """R should be orthogonal: R^T R = I, det(R) = +1."""
    print("=" * 60)
    print("TEST 1: Rotation matrix orthogonality")
    print("=" * 60)

    test_cases = [
        (0, 0, 0),
        (90, 0, 0),
        (0, 45, 0),
        (0, 0, 30),
        (251.5, -2.2, 1.9),      # Real value from P1
        (180, -89, 45),           # Extreme angles
        (359.9, 0.1, -0.1),      # Near-boundary
    ]

    all_pass = True
    for az, pitch, roll in test_cases:
        R = get_rotation_matrix(az, pitch, roll)
        I_check = R.T @ R
        identity_err = np.max(np.abs(I_check - np.eye(3)))
        det = np.linalg.det(R)

        ok = identity_err < 1e-12 and abs(det - 1.0) < 1e-12
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  ({az:7.1f}, {pitch:6.1f}, {roll:5.1f})  "
              f"max|R^T R - I| = {identity_err:.2e}  det = {det:.10f}  [{status}]")

    print(f"\n  Result: {'ALL PASS' if all_pass else 'SOME FAILED'}\n")
    return all_pass


def test_magnitude_preservation_synthetic():
    """body_to_world should preserve vector magnitude exactly (synthetic data)."""
    print("=" * 60)
    print("TEST 2: Magnitude preservation (synthetic)")
    print("=" * 60)

    rng = np.random.default_rng(42)
    n_tests = 1000
    max_err = 0.0

    for _ in range(n_tests):
        mb = rng.standard_normal(3) * 50  # random body-frame vector ~50 uT
        az, pitch, roll = rng.uniform(0, 360), rng.uniform(-90, 90), rng.uniform(-180, 180)
        R = get_rotation_matrix(az, pitch, roll)
        mn = body_to_world(mb, R)
        err = abs(np.linalg.norm(mb) - np.linalg.norm(mn))
        max_err = max(max_err, err)

    print(f"  {n_tests} random vectors, max |mb| - |mn| error: {max_err:.2e} uT")
    ok = max_err < 1e-10
    print(f"  Result: {'PASS' if ok else 'FAIL'}\n")
    return ok


def test_compute_horizontal_vertical():
    """Check mh and mv decomposition."""
    print("=" * 60)
    print("TEST 3: Horizontal/Vertical decomposition")
    print("=" * 60)

    # Test vector: [3, 4, -5] → mh = sqrt(9+16) = 5, mv = -5
    mn = np.array([3.0, 4.0, -5.0])
    mh, mv = compute_horizontal_vertical(mn)
    assert abs(mh - 5.0) < 1e-10, f"mh expected 5.0, got {mh}"
    assert abs(mv - (-5.0)) < 1e-10, f"mv expected -5.0, got {mv}"
    print(f"  mn=[3,4,-5] → mh={mh:.1f}, mv={mv:.1f}  [PASS]")

    # Verify |B| = sqrt(mh² + mv²)
    mag = np.linalg.norm(mn)
    recon = np.sqrt(mh**2 + mv**2)
    assert abs(mag - recon) < 1e-10
    print(f"  |mn|={mag:.4f}, sqrt(mh²+mv²)={recon:.4f}  [PASS]")

    print(f"  Result: PASS\n")
    return True


def test_compute_inclination():
    """Check inclination calculation."""
    print("=" * 60)
    print("TEST 4: Inclination calculation")
    print("=" * 60)

    # Horizontal field only → inclination = 0°
    assert abs(compute_inclination(10.0, 0.0)) < 1e-10
    print(f"  mh=10, mv=0   → incl={compute_inclination(10.0, 0.0):.1f}°  [PASS]")

    # Pure downward field → inclination = 90°
    assert abs(compute_inclination(0.0, 10.0) - 90.0) < 1e-10
    print(f"  mh=0,  mv=10  → incl={compute_inclination(0.0, 10.0):.1f}°  [PASS]")

    # 45° dip
    incl = compute_inclination(10.0, 10.0)
    assert abs(incl - 45.0) < 1e-10
    print(f"  mh=10, mv=10  → incl={incl:.1f}°  [PASS]")

    # Negative dip
    incl_neg = compute_inclination(10.0, -10.0)
    assert abs(incl_neg - (-45.0)) < 1e-10
    print(f"  mh=10, mv=-10 → incl={incl_neg:.1f}°  [PASS]")

    print(f"  Result: PASS\n")
    return True


def test_magnitude_preservation_real_data():
    """
    Core acceptance test on real sensor data.
    For every cleaned CSV, run add_world_frame and verify |mb| ≈ |mn|.
    Acceptance: |mb| == |mn| within 1% for at least 95% of samples.
    """
    print("=" * 60)
    print("TEST 5: Magnitude preservation on REAL DATA")
    print("=" * 60)

    processed_dir = os.path.join(PROJECT_ROOT, 'data', 'processed')
    cleaned_files = sorted(glob.glob(os.path.join(processed_dir, 'static_*_cleaned.csv')))

    if not cleaned_files:
        print("  WARNING: No cleaned CSV files found. Skipping real-data test.")
        return True

    total_samples = 0
    pass_samples = 0
    fail_samples = 0
    worst_file = None
    worst_err = 0.0
    worst_pct = 0.0

    for path in cleaned_files:
        fname = os.path.basename(path)
        df = pd.read_csv(path)

        # Check required columns exist
        required = ['Bx_cal', 'By_cal', 'Bz_cal', 'Azimuth', 'Pitch', 'Roll']
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"  SKIP {fname}: missing columns {missing}")
            continue

        df_wf = add_world_frame(df)

        mag_body  = np.sqrt(df_wf['Bx_cal']**2 + df_wf['By_cal']**2 + df_wf['Bz_cal']**2)
        mag_world = np.sqrt(df_wf['Bx_w']**2  + df_wf['By_w']**2  + df_wf['Bz_w']**2)

        abs_err = (mag_body - mag_world).abs()
        rel_err = abs_err / mag_body  # relative error

        n = len(df_wf)
        n_pass = int((rel_err < 0.01).sum())  # within 1%
        n_fail = n - n_pass
        file_max_err = abs_err.max()
        file_pass_rate = n_pass / n * 100

        total_samples += n
        pass_samples += n_pass
        fail_samples += n_fail

        if file_max_err > worst_err:
            worst_err = file_max_err
            worst_file = fname
            worst_pct = file_pass_rate

        print(f"  {fname:<50s}  samples={n:4d}  "
              f"pass={file_pass_rate:6.2f}%  max_err={file_max_err:.6f} uT")

    overall_rate = pass_samples / total_samples * 100 if total_samples > 0 else 0
    print(f"\n  SUMMARY:")
    print(f"    Total samples tested : {total_samples}")
    print(f"    Passing (< 1% error) : {pass_samples} ({overall_rate:.2f}%)")
    print(f"    Failing              : {fail_samples}")
    print(f"    Worst file           : {worst_file} (max abs err = {worst_err:.6f} uT)")

    ok = overall_rate >= 95.0
    print(f"\n  Acceptance (>=95% pass): {'PASS' if ok else 'FAIL'} ({overall_rate:.2f}%)\n")
    return ok


if __name__ == '__main__':
    results = []
    results.append(("Rotation matrix orthogonality", test_rotation_matrix_orthogonality()))
    results.append(("Magnitude preservation (synthetic)", test_magnitude_preservation_synthetic()))
    results.append(("Horizontal/Vertical decomposition", test_compute_horizontal_vertical()))
    results.append(("Inclination calculation", test_compute_inclination()))
    results.append(("Magnitude preservation (real data)", test_magnitude_preservation_real_data()))

    print("=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, ok in results:
        status = "PASS" if ok else "FAIL"
        if not ok:
            all_pass = False
        print(f"  [{status}]  {name}")

    print()
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")

    sys.exit(0 if all_pass else 1)
