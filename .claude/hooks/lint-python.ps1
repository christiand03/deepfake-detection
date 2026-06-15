$stdin = [Console]::In.ReadToEnd()
try {
    $payload = $stdin | ConvertFrom-Json
} catch {
    exit 0
}

$filePath = $payload.tool_input.file_path
if (-not $filePath) { exit 0 }

$normalized = $filePath -replace '\\', '/'
if ($normalized -notmatch '\.py$') { exit 0 }
if ($normalized -notmatch '/(src|tests)/') { exit 0 }
if (-not (Test-Path -LiteralPath $filePath)) { exit 0 }

# Run ruff from the project root with a project-relative path. tool.ruff
# per-file-ignores globs (e.g. "src/models/*") only match a path relative to the
# project root; an absolute path bypasses them and would wrongly auto-strip
# quoted single-token jaxtyping axes ("batch" -> batch via UP037), breaking the
# file at runtime.
$projDir = $env:CLAUDE_PROJECT_DIR
$ruffPath = $filePath
if ($projDir) {
    $projNorm = ($projDir -replace '\\', '/').TrimEnd('/')
    if ($normalized.StartsWith("$projNorm/", [System.StringComparison]::OrdinalIgnoreCase)) {
        $ruffPath = $normalized.Substring($projNorm.Length + 1)
    }
}

$runDir = if ($projDir) { $projDir } else { Split-Path -LiteralPath $filePath }
Push-Location -LiteralPath $runDir
try {
    $checkOutput = (& ruff check $ruffPath --fix 2>&1 | Out-String).Trim()
    & ruff format $ruffPath 2>&1 | Out-Null
} finally {
    Pop-Location
}

if ($checkOutput -and $checkOutput -notmatch '^All checks passed!?$') {
    $result = @{
        hookSpecificOutput = @{
            hookEventName     = "PostToolUse"
            additionalContext = "ruff check found remaining issues in ${filePath}:`n${checkOutput}"
        }
    }
    $result | ConvertTo-Json -Compress -Depth 5
}

exit 0
