param(
    [string]$Message = "Auto-commit: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    [switch]$Push = $true
)

$ErrorActionPreference = "Stop"

Write-Host "Staging all changes..." -ForegroundColor Cyan
git add -A

$status = git status --porcelain
if ($status) {
    Write-Host "Changes detected. Committing..." -ForegroundColor Green
    git commit -m $Message
    
    if ($Push) {
        Write-Host "Pushing to remote..." -ForegroundColor Green
        git push
        Write-Host "Successfully pushed to GitHub!" -ForegroundColor Green
    }
} else {
    Write-Host "No changes to commit." -ForegroundColor Yellow
}

git status --short