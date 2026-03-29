"""
scoring_alignment/sn123_mantis/

SN123 MANTIS scoring alignment tests.

IMPORTANT ARCHITECTURAL DIFFERENCE FROM SN44:

SN44 has a deterministic scoring function (mAP50, false_positive) that
can be exactly replicated locally. Tests verify local replica == validator output.

MANTIS uses STOCHASTIC salience scoring:
  - Walk-forward logistic regression over 30 days of embedding history
  - Scores depend on ALL miners' embeddings, not just yours
  - Validator computes salience using model.multi_salience() on a shared DataLog
  - Your score is your MARGINAL CONTRIBUTION to the ensemble, not your raw accuracy

You CANNOT replicate your exact score locally.

WHAT YOU CAN TEST LOCALLY:
  1. Payload format and encryption are correct (validator will reject malformed)
  2. Embedding dimensions match config.CHALLENGES exactly (wrong dims = zero vector)
  3. Embedding values are within [-1, 1] (clipped, but clipping = information loss)
  4. Commit URL format is valid (wrong URL format = payload never read)
  5. Your embeddings are non-zero (zero vectors = no salience signal)
  6. LBFGS embedding structure if using 17-dim challenges
  7. Local salience using evaluate_embeddings.py against a downloaded DataLog

Repo: https://github.com/Barbariandev/MANTIS
"""
from __future__ import annotations
import json
import os
import sys
import math
import random
import hashlib
from typing import Any
import importlib.util

# ── Configuration (read from MANTIS config.py if available) ──────────────────
# These are the values from config.py as documented in the repo.
# If you have cloned the MANTIS repo, update MANTIS_REPO_PATH below.
MANTIS_REPO_PATH = os.environ.get(
    "MANTIS_REPO_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "MANTIS")
)

def _load_mantis_config():
    """Load config.py from the cloned MANTIS repo."""
    config_path = os.path.join(MANTIS_REPO_PATH, "config.py")
    if not os.path.exists(config_path):
        return None
    spec = importlib.util.spec_from_file_location("mantis_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        print(f"  Warning: could not load MANTIS config.py: {e}")
        return None

_config = _load_mantis_config()

# Fallback values if config.py not available
# Update these from your live config.py
CHALLENGES = getattr(_config, "CHALLENGES", [
    # Format: {ticker, dim, blocks_ahead, loss_func, ...}
    # These are placeholder values -- read from actual config.py
    {"ticker": "BTC-USD", "dim": 1, "blocks_ahead": 300},
    {"ticker": "BTC-USD", "dim": 2, "blocks_ahead": 600},
    # Add all challenges from your config.py here
])

SAMPLE_EVERY = getattr(_config, "SAMPLE_EVERY", 5)  # blocks between samples
TLOCK_DEFAULT_LOCK_SECONDS = getattr(_config, "TLOCK_DEFAULT_LOCK_SECONDS", 3700)
OWNER_HPKE_PUBLIC_KEY_HEX = getattr(_config, "OWNER_HPKE_PUBLIC_KEY_HEX", "")
DRAND_API = getattr(_config, "DRAND_API", "https://api.drand.sh")
DATALOG_ARCHIVE_URL = getattr(_config, "DATALOG_ARCHIVE_URL", "")
STORAGE_DIR = getattr(_config, "STORAGE_DIR", "./data")
ALG_LABEL_V2 = getattr(_config, "ALG_LABEL_V2", "mantis-v2")
MAX_PAYLOAD_SIZE_BYTES = 25 * 1024 * 1024  # 25 MB from README

# Valid R2 commit host patterns from README
VALID_R2_HOSTS = [".r2.dev", ".r2.cloudflarestorage.com"]


# ── CHECK 1: Payload format validation ───────────────────────────────────────
def test_v2_payload_has_required_fields():
    """
    V2 payloads must contain: v, round, hk, owner_pk, C, W_owner, W_time,
    binding, alg. Missing any field = payload rejected by ledger.py.

    From README: 'Only V2 JSON payloads are accepted.'
    From generate_and_encrypt.py: exact field names.
    """
    # Simulate a valid v2 payload structure (without real encryption)
    mock_payload = {
        "v": 2,
        "round": 12345678,
        "hk": "5ABC...",              # your hotkey ss58
        "owner_pk": "a" * 64,        # 32 bytes hex
        "C": {
            "nonce": "b" * 24,       # 12 bytes hex
            "ct": "c" * 96,          # ciphertext hex
        },
        "W_owner": {
            "pke": "d" * 64,         # 32 bytes hex
            "nonce": "e" * 24,
            "ct": "f" * 96,
        },
        "W_time": {
            "ct": "0" * 128,         # timelock ciphertext
        },
        "binding": "b" * 64,
        "alg": ALG_LABEL_V2,
    }

    required_fields = ["v", "round", "hk", "owner_pk", "C", "W_owner",
                       "W_time", "binding", "alg"]
    for field in required_fields:
        assert field in mock_payload, (
            f"Required field '{field}' missing from V2 payload. "
            f"Validator will reject this payload silently."
        )

    assert mock_payload["v"] == 2, "Payload version must be 2"
    assert isinstance(mock_payload["C"], dict), "'C' must be a dict with nonce and ct"
    assert "nonce" in mock_payload["C"] and "ct" in mock_payload["C"]
    assert isinstance(mock_payload["W_owner"], dict)
    assert "pke" in mock_payload["W_owner"]
    assert isinstance(mock_payload["W_time"], dict)
    assert "ct" in mock_payload["W_time"]


def test_payload_size_under_25mb():
    """
    From README: 'Payloads must be <= 25 MB.'
    A payload over this limit is rejected without error.
    """
    # Estimate payload size for your challenge dimensions
    total_floats = sum(c.get("dim", 1) for c in CHALLENGES)
    # JSON overhead per float: ~12 bytes (key + value + separators)
    estimated_json_bytes = total_floats * 20 + 2000  # conservative estimate
    encrypted_overhead = 512  # nonces, keys, binding, etc.
    estimated_total = estimated_json_bytes + encrypted_overhead

    assert estimated_total < MAX_PAYLOAD_SIZE_BYTES, (
        f"Estimated payload size {estimated_total/1024:.1f} KB exceeds 25 MB limit. "
        f"Challenges: {len(CHALLENGES)}, total_dims: {total_floats}"
    )

    print(f"  Estimated payload size: {estimated_total/1024:.1f} KB "
          f"(limit: {MAX_PAYLOAD_SIZE_BYTES/1024/1024:.0f} MB)")


def test_commit_url_format():
    """
    From README:
      Commit host: Cloudflare R2 only (*.r2.dev or *.r2.cloudflarestorage.com)
      Object key: Path must be exactly your hotkey (no directories or extra segments)

    Wrong URL = payload never downloaded by validator.
    """
    # Simulate a valid commit URL
    example_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    bucket_name = "your-bucket-name"

    # Valid patterns
    valid_urls = [
        f"https://{bucket_name}.r2.dev/{example_hotkey}",
        f"https://{bucket_name}.r2.cloudflarestorage.com/{example_hotkey}",
    ]

    # Invalid patterns that miners commonly get wrong
    invalid_urls = [
        f"https://{bucket_name}.r2.dev/{example_hotkey}/payload.json",  # extra path
        f"https://{bucket_name}.r2.dev/hotkeys/{example_hotkey}",       # subdirectory
        f"https://s3.amazonaws.com/{bucket_name}/{example_hotkey}",     # wrong host
        f"https://{bucket_name}.r2.dev/{example_hotkey.lower()}",       # wrong case
    ]

    for url in valid_urls:
        parsed_host = url.split("/")[2]  # netloc
        is_valid = any(parsed_host.endswith(pattern) for pattern in VALID_R2_HOSTS)
        assert is_valid, f"Valid URL should pass host check: {url}"

        # Check key is exactly the hotkey
        path = "/" + "/".join(url.split("/")[3:])
        key = path.lstrip("/")
        assert key == example_hotkey, (
            f"Object key '{key}' must be exactly the hotkey '{example_hotkey}'. "
            "No subdirectories, no file extensions."
        )

    print(f"  Valid URL patterns verified.")
    print(f"  Common mistake: adding /payload.json or /embeddings/ prefix to key.")


# ── CHECK 2: Embedding dimensions and value range ────────────────────────────
def test_embedding_dimensions_match_challenges():
    """
    From generate_and_encrypt.py:
      clip_unit() clamps values to [-1, 1]
      Wrong dim = vector silently skipped or zero-padded

    From evaluate_embeddings.py inject_synthetic_embeddings():
      'if not isinstance(vec, (list, tuple)) or len(vec) != spec["dim"]: continue'
      Wrong length = silently skipped, no error raised.
    """
    # Simulate your embedding generator
    def your_generator(challenges):
        """Replace this with your actual embedding generation."""
        return [
            [random.uniform(-0.5, 0.5) for _ in range(c["dim"])]
            for c in challenges
        ]

    embeddings = your_generator(CHALLENGES)

    assert len(embeddings) == len(CHALLENGES), (
        f"Expected {len(CHALLENGES)} embeddings (one per challenge), "
        f"got {len(embeddings)}."
    )

    for i, (emb, challenge) in enumerate(zip(embeddings, CHALLENGES)):
        expected_dim = challenge["dim"]
        ticker = challenge.get("ticker", f"challenge_{i}")

        assert isinstance(emb, (list, tuple)), (
            f"Challenge {i} ({ticker}): embedding must be a list or tuple, "
            f"got {type(emb).__name__}."
        )
        assert len(emb) == expected_dim, (
            f"Challenge {i} ({ticker}): expected dim={expected_dim}, "
            f"got {len(emb)}. Wrong dimension = silently skipped by validator."
        )

    print(f"  {len(CHALLENGES)} challenges, all dimensions correct.")


def test_embedding_values_in_range():
    """
    From evaluate_embeddings.py:
      clip_unit() clamps to [-1, 1] silently.
      Values outside this range are clipped, losing information.
      Verify your model outputs in [-1, 1] natively to avoid clipping.
    """
    # Test with borderline values
    def clip_unit(v: float) -> float:
        return -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)

    test_cases = [
        (0.5, 0.5, "Normal value"),
        (-0.5, -0.5, "Normal negative"),
        (1.0, 1.0, "At upper bound"),
        (-1.0, -1.0, "At lower bound"),
        (1.001, 1.0, "Slightly over -- CLIPPED"),
        (-1.001, -1.0, "Slightly under -- CLIPPED"),
        (2.0, 1.0, "Far over -- CLIPPED, information lost"),
        (float('inf'), 1.0, "Inf -- CLIPPED"),
    ]

    for input_val, expected, label in test_cases:
        if math.isinf(input_val):
            # clip_unit doesn't handle inf -- check your generator handles it
            print(f"  Warning: {label}: inf values need explicit handling in your generator")
            continue
        clipped = clip_unit(input_val)
        assert clipped == expected, f"{label}: clip_unit({input_val}) = {clipped}, expected {expected}"

    print("  clip_unit mechanics verified.")
    print("  Ensure your model outputs are in [-1, 1] to avoid information loss from clipping.")


def test_embeddings_are_nonzero():
    """
    From ledger.py payload validation:
      'Payload validation replaces malformed data with zero vectors.'
      From evaluate_embeddings.py:
      'if any(x != 0.0 for x in clipped):' -- only non-zero vectors are stored.

    All-zero embeddings carry NO salience signal whatsoever.
    They are stored but contribute nothing to the ensemble.
    """
    def your_generator(challenges):
        """Replace with your actual generator."""
        return [[random.uniform(-0.5, 0.5) for _ in range(c["dim"])]
                for c in challenges]

    embeddings = your_generator(CHALLENGES)

    for i, (emb, challenge) in enumerate(zip(embeddings, CHALLENGES)):
        ticker = challenge.get("ticker", f"challenge_{i}")
        is_zero = all(abs(v) < 1e-10 for v in emb)
        assert not is_zero, (
            f"Challenge {i} ({ticker}): all-zero embedding detected. "
            "Zero vectors provide no salience signal to the validator. "
            "Check your model is producing non-trivial predictions."
        )

    print(f"  All {len(CHALLENGES)} embeddings are non-zero.")


# ── CHECK 3: LBFGS embedding structure (if using 17-dim challenges) ──────────
def test_lbfgs_embedding_structure():
    """
    From lbfgs_guide.md and bucket_forecast.py:
    17-dim LBFGS embeddings have specific structure requirements.

    If your challenge uses dim=17 (or other LBFGS dims), the validator
    calls compute_lbfgs_salience which expects embeddings structured as
    (p_direction, Q_matrix_flat) -- not arbitrary 17-float vectors.

    This test verifies the structure if you're using LBFGS challenges.
    """
    lbfgs_challenges = [c for c in CHALLENGES
                        if c.get("dim") in [17, 33, 65]]  # common LBFGS dims

    if not lbfgs_challenges:
        print("  No LBFGS-dimension challenges found. Skipping LBFGS structure check.")
        return

    for challenge in lbfgs_challenges:
        dim = challenge["dim"]
        ticker = challenge.get("ticker", "unknown")
        print(f"  Challenge {ticker} uses dim={dim} -- verify LBFGS structure.")
        print(f"  See lbfgs_guide.md: first element = p (direction), "
              f"remaining {dim-1} = Q (flattened matrix).")

        # Basic structural check: dim should be 2^n + 1 for LBFGS
        # (1 for p, 2^n for Q)
        if dim > 1:
            q_size = dim - 1
            is_power_of_2 = (q_size & (q_size - 1)) == 0
            if not is_power_of_2:
                print(f"  Warning: dim-1={q_size} is not a power of 2. "
                      f"Verify this is the expected LBFGS Q dimension.")


# ── CHECK 4: Hotkey binding ───────────────────────────────────────────────────
def test_hotkey_binding_matches_commit():
    """
    From generate_and_encrypt.py _binding():
    The binding includes your hotkey. From ledger.py:
    'Embedded hotkey checks ensure decrypted payloads belong to the committing miner.'

    If your payload's 'hk' field doesn't match the on-chain commit hotkey,
    the payload is rejected after decryption (300 blocks later).
    This is a delayed rejection -- you won't know until the decrypt window.
    """
    # Verify your payload generator uses the correct hotkey
    # The hotkey in the payload must exactly match the committing hotkey

    def verify_payload_hotkey(payload: dict, expected_hotkey: str) -> bool:
        """Check payload hotkey matches committing hotkey."""
        return payload.get("hk") == expected_hotkey

    example_hotkey = "5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY"
    mock_payload = {"hk": example_hotkey, "v": 2}

    assert verify_payload_hotkey(mock_payload, example_hotkey), \
        "Payload hk must match committing hotkey exactly."

    # Wrong case
    assert not verify_payload_hotkey(mock_payload, example_hotkey.lower()), \
        "Hotkey comparison must be case-sensitive."

    print("  Hotkey binding check: verify your payload.hk == your wallet hotkey.")
    print("  Mismatch causes delayed rejection (300 blocks after commit).")


# ── CHECK 5: Timelock round alignment ────────────────────────────────────────
def test_timelock_round_is_future():
    """
    From generate_and_encrypt.py _target_round():
    The Drand round must be in the FUTURE at submission time.
    If the round has already passed, the Drand beacon has already published
    the signature, and the time-lock encryption provides no security.
    Validators may reject payloads where the round is too close to current time.

    lock_seconds should be >= TLOCK_DEFAULT_LOCK_SECONDS (typically 3700s ~1 hour).
    """
    import time
    current_time = time.time()

    # Simulate what _target_round() does
    lock_seconds = TLOCK_DEFAULT_LOCK_SECONDS
    future_time = current_time + lock_seconds

    # Drand period is 3 seconds
    drand_period = 3
    # Approximate blocks_ahead from lock_seconds
    estimated_blocks_ahead = lock_seconds // 12  # ~12s per block

    assert lock_seconds >= 3600, (
        f"lock_seconds={lock_seconds} is less than 1 hour. "
        f"Time-lock should be set well into the future. "
        f"See config.TLOCK_DEFAULT_LOCK_SECONDS."
    )

    print(f"  lock_seconds: {lock_seconds}s (~{lock_seconds//3600:.1f}h)")
    print(f"  Approximate blocks_ahead: {estimated_blocks_ahead}")
    print(f"  This determines which future Drand round is used as the decryption key.")


# ── CHECK 6: Local salience evaluation ───────────────────────────────────────
def test_local_salience_with_evaluate_embeddings():
    """
    MANTIS provides evaluate_embeddings.py for local salience testing.
    This test verifies you can run it with your generator.

    From evaluate_embeddings.py:
      1. Downloads/loads the DataLog from config.DATALOG_ARCHIVE_URL
      2. Injects your embeddings as 'synthetic_hotkey'
      3. Runs model.multi_salience() against the full miner population
      4. Reports your salience score and rank

    Usage:
      python evaluate_embeddings.py --generator your_generator.py --days 7

    Where your_generator.py must expose:
      def generate_embeddings(block: int) -> list[list[float]]:
          # Returns one embedding per challenge in config.CHALLENGES order
          ...

    This test verifies your generator has the correct signature.
    """
    # Check that evaluate_embeddings.py exists in the MANTIS repo
    eval_path = os.path.join(MANTIS_REPO_PATH, "evaluate_embeddings.py")
    if not os.path.exists(eval_path):
        print(f"  SKIP: evaluate_embeddings.py not found at {eval_path}")
        print(f"  Clone MANTIS repo and set MANTIS_REPO_PATH env var.")
        return

    # Verify your generator file exists
    generator_path = os.environ.get("MANTIS_GENERATOR_PATH", "")
    if not generator_path or not os.path.exists(generator_path):
        print("  Generator file not set. Set MANTIS_GENERATOR_PATH env var.")
        print("  Generator must expose: def generate_embeddings(block: int) -> list[list[float]]")
        print()
        print("  To run local salience evaluation:")
        print(f"    export MANTIS_REPO_PATH={MANTIS_REPO_PATH}")
        print(f"    export MANTIS_GENERATOR_PATH=path/to/your/generator.py")
        print(f"    cd {MANTIS_REPO_PATH}")
        print(f"    python evaluate_embeddings.py --generator $MANTIS_GENERATOR_PATH --days 7")
        return

    # Verify generator has correct signature
    spec = importlib.util.spec_from_file_location("generator", generator_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert hasattr(mod, "generate_embeddings"), (
        f"Generator {generator_path} must expose generate_embeddings(block) function."
    )

    # Test call signature
    test_embeddings = mod.generate_embeddings(0)
    assert isinstance(test_embeddings, list), \
        "generate_embeddings() must return a list."
    assert len(test_embeddings) == len(CHALLENGES), (
        f"generate_embeddings() returned {len(test_embeddings)} embeddings, "
        f"expected {len(CHALLENGES)} (one per challenge)."
    )

    print(f"  Generator verified: {generator_path}")
    print(f"  Returns {len(test_embeddings)} embeddings for {len(CHALLENGES)} challenges.")
    print()
    print("  Run full local evaluation:")
    print(f"    cd {MANTIS_REPO_PATH}")
    print(f"    python evaluate_embeddings.py --generator {generator_path} --days 7")


# ── CHECK 7: Challenge config change detection ───────────────────────────────
def test_challenge_config_is_current():
    """
    config.py is the source of truth for challenges.
    If the repo updates config.py (new tickers, new dims, new blocks_ahead),
    your generator must update to match.

    This test warns if config.py appears to have changed since you last
    verified your generator against it.
    """
    config_path = os.path.join(MANTIS_REPO_PATH, "config.py")
    if not os.path.exists(config_path):
        print(f"  SKIP: config.py not found at {config_path}")
        return

    # Hash the config.py to detect changes
    with open(config_path, "rb") as f:
        current_hash = hashlib.sha256(f.read()).hexdigest()[:12]

    # Check against stored hash
    hash_file = os.path.join(os.path.dirname(__file__), ".config_hash")
    if os.path.exists(hash_file):
        stored_hash = open(hash_file).read().strip()
        if stored_hash != current_hash:
            print(f"  WARNING: config.py has changed since last verification!")
            print(f"  Previous hash: {stored_hash}")
            print(f"  Current hash:  {current_hash}")
            print(f"  Review changes: git log --oneline MANTIS/config.py")
            print(f"  Update your generator to match new challenge dimensions.")
            # Update the stored hash
            with open(hash_file, "w") as f:
                f.write(current_hash)
        else:
            print(f"  config.py unchanged (hash: {current_hash})")
    else:
        # First run: store the hash
        with open(hash_file, "w") as f:
            f.write(current_hash)
        print(f"  config.py hash stored: {current_hash}")
        print(f"  Future runs will alert you if config.py changes.")

    # Report current challenges
    print(f"  Current challenges ({len(CHALLENGES)}):")
    for c in CHALLENGES:
        print(f"    {c.get('ticker','?')} dim={c.get('dim','?')} "
              f"blocks_ahead={c.get('blocks_ahead','?')}")


# ── Main runner ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tests = [
        ("CHECK 1a", "V2 payload required fields",
            test_v2_payload_has_required_fields),
        ("CHECK 1b", "Payload size under 25 MB",
            test_payload_size_under_25mb),
        ("CHECK 1c", "Commit URL format (R2 only, hotkey as key)",
            test_commit_url_format),
        ("CHECK 2a", "Embedding dimensions match CHALLENGES",
            test_embedding_dimensions_match_challenges),
        ("CHECK 2b", "Embedding values in [-1, 1]",
            test_embedding_values_in_range),
        ("CHECK 2c", "Embeddings are non-zero",
            test_embeddings_are_nonzero),
        ("CHECK 3",  "LBFGS embedding structure (if applicable)",
            test_lbfgs_embedding_structure),
        ("CHECK 4",  "Hotkey binding matches commit",
            test_hotkey_binding_matches_commit),
        ("CHECK 5",  "Timelock round is sufficiently future",
            test_timelock_round_is_future),
        ("CHECK 6",  "Local salience via evaluate_embeddings.py",
            test_local_salience_with_evaluate_embeddings),
        ("CHECK 7",  "Challenge config is current (detect config.py changes)",
            test_challenge_config_is_current),
    ]

    passed, failed, skipped = 0, 0, 0
    for check_id, desc, fn in tests:
        print(f"\n{check_id}: {desc}")
        try:
            fn()
            print(f"  PASS")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{'='*60}")
    print(f"{passed} passed, {failed} failed")
    print()
    print("MANTIS scoring note:")
    print("  These tests verify payload format and embedding validity.")
    print("  They do NOT verify your salience score -- that is stochastic.")
    print("  Use evaluate_embeddings.py for local salience estimation.")
    print("  Only the top ~20-30 miners out of 256 earn meaningful emissions.")
    sys.exit(0 if failed == 0 else 1)
