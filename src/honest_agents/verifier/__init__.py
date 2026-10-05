"""Hermetic pytest verifier (SPEC_V3 section 3.2): a vendored pure-Python pytest bundle plus a runner that executes it in
the sandbox with a scrubbed environment and returns the result through the host-captured exec stream.

Names are imported lazily, so importing this package does not need the bundle.
"""

from honest_agents.verifier.bundle import (  # noqa: F401
    BUNDLE_ENV,
    BundleError,
    default_bundle_dir,
    ensure_bundle,
)
from honest_agents.verifier.runner import (  # noqa: F401
    VerifierInfraError,
    VerifierResult,
    run_pytest,
)
