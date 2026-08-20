"""One process-wide worker thread for every model call the API makes.

All GPU work — analysis, robustness sweeps, adversarial attacks — runs on this single
executor, so no two model passes are ever in flight at the same time.

That is a correctness requirement, not just tidiness. ``lxt`` patches transformer layers
by ``setattr`` on the CLASS, which makes the patched/un-patched state process-global.
As long as patching was monotone (patch once, never undo) overlapping requests were
harmless: everyone wanted the same state. :func:`src.utils.attnlrp.lxt_patches_disabled`
breaks that assumption — it un-patches for the duration of a Chefer pass. With one
executor per router (the previous layout) a Chefer request on the analyze thread could
un-patch while a robustness sweep sat mid-``explain()`` on its own thread, silently
degrading that sweep's AttnLRP to plain Input×Gradient: no exception, no log line, just
unfaithful heatmaps. Serialising the work removes the race by construction, with no
locking in the relevance paths.

The single worker also matches the hardware: the three routers previously contended for
one GPU anyway, so queueing makes latency predictable rather than slower on average.

Trade-off, deliberately accepted: a robustness sweep and an analysis request no longer
overlap. A long sweep therefore delays a queued analysis until it finishes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

# max_workers=1 is the invariant this module exists to guarantee — raising it would
# reintroduce the patch race described above.
inference_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="inference")
