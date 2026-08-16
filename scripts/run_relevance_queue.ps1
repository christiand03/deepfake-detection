<#
.SYNOPSIS
    Chain the relevance-regularization runs: wait for the control, then start Run 1.

.DESCRIPTION
    Run 0 (loc_lambda = 0) and Run 1 (loc_lambda > 0) must not overlap: the 8 GB GPU
    fits exactly one of them (Gate G2 measured 7.57 GB peak at batch 1), so launching
    both would spill to shared memory and slow each by roughly an order of magnitude.
    This waits for the running control to exit before starting the regularized run.

    Run 1 is started even if Run 0 exits non-zero. The two are independent training
    runs -- Run 0 is the comparison baseline, not a dependency -- and an unattended
    queue that silently skips the main experiment because the control hiccupped is
    worse than one that runs both and lets the results be compared later. The exit code
    of Run 0 is recorded in the log either way.

.PARAMETER WaitPid
    PID of the already-running Run 0. Omit to start Run 0 as well.

.EXAMPLE
    powershell -File scripts/run_relevance_queue.ps1 -WaitPid 30280
    powershell -File scripts/run_relevance_queue.ps1
#>
param(
    [int]$WaitPid = 0,
    [string]$LogDir = "temp"
)

$ErrorActionPreference = "Continue"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Write-Stamp($Message) {
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Output $line
    Add-Content -Path "$LogDir/relevance_queue.log" -Value $line -Encoding utf8
}

Write-Stamp "queue started"

if ($WaitPid -gt 0) {
    $proc = Get-Process -Id $WaitPid -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Stamp "waiting for Run 0 (PID $WaitPid, started $($proc.StartTime.ToString('HH:mm:ss')))"
        Wait-Process -Id $WaitPid -ErrorAction SilentlyContinue
        Write-Stamp "Run 0 process exited"
    }
    else {
        Write-Stamp "PID $WaitPid not running - proceeding straight to Run 1"
    }
}
else {
    Write-Stamp "starting Run 0 (control, loc_lambda=0)"
    & python -u src/train.py experiment=train_video_relevance_reg_lambda0 logger=csv extras.enforce_tags=false 2>&1 |
        Tee-Object -FilePath "$LogDir/run0_control.log"
    Write-Stamp "Run 0 finished with exit code $LASTEXITCODE"
}

# The GPU allocator needs a moment to release the context before the next run claims it.
Write-Stamp "GPU before Run 1: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>$null)"

Write-Stamp "starting Run 1 (regularized, loc_lambda>0)"
& python -u src/train.py experiment=train_video_relevance_reg logger=csv extras.enforce_tags=false 2>&1 |
    Tee-Object -FilePath "$LogDir/run1_regularized.log"
Write-Stamp "Run 1 finished with exit code $LASTEXITCODE"

Write-Stamp "queue complete - next: scripts/eval_localization.py on both checkpoints"
