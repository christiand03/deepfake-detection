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

$checkOutput = (& ruff check $filePath --fix 2>&1 | Out-String).Trim()
& ruff format $filePath 2>&1 | Out-Null

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
