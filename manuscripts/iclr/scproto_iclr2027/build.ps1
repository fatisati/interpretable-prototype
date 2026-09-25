# Build scProto's ICLR 2027 PDF.
#   usage:  .\build.ps1          run from this folder
# All aux/log clutter lands in build\ ; the finished PDF is copied back here
# as scproto_iclr2027.pdf.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$jobname = 'scproto_iclr2027'
New-Item -ItemType Directory -Force build | Out-Null

# pdflatex must find style\ and figures\ relative to this folder, and bibtex
# must find the .bib here while its .aux sits in build\ -- hence these three
# search paths (trailing ';' = "then also look in the normal places").
$env:TEXINPUTS = "$PSScriptRoot;$PSScriptRoot\style;"
$env:BIBINPUTS = "$PSScriptRoot;"
$env:BSTINPUTS = "$PSScriptRoot;$PSScriptRoot\style;"

# NOTE: the flag is -job-name (with the hyphen). -jobname is silently ignored,
# and the args go in an array so PowerShell expands $jobname before pdflatex
# sees it.
$texArgs = @(
    '-interaction=nonstopmode',
    '-halt-on-error',
    '-output-directory=build',
    "-job-name=$jobname",
    'main.tex'
)

function Invoke-PdfLatex {
    pdflatex @texArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`npdflatex FAILED -- errors from the log:" -ForegroundColor Red
        if (Test-Path "build\$jobname.log") {
            Select-String -Path "build\$jobname.log" -Pattern '^!' -Context 0,4 |
                ForEach-Object { $_.Line; $_.Context.PostContext }
        }
        exit 1
    }
}

Write-Host '[1/4] pdflatex' -ForegroundColor Cyan ; Invoke-PdfLatex
Write-Host '[2/4] bibtex'   -ForegroundColor Cyan ; bibtex "build\$jobname"
Write-Host '[3/4] pdflatex' -ForegroundColor Cyan ; Invoke-PdfLatex
Write-Host '[4/4] pdflatex' -ForegroundColor Cyan ; Invoke-PdfLatex

Copy-Item "build\$jobname.pdf" ".\$jobname.pdf" -Force
# MiKTeX hard-wraps log lines, so "Output written on ... (18 pages" can be split
# across two lines -- join the log before matching.
$logText = (Get-Content "build\$jobname.log" -Raw) -replace "`r?`n", ' '
$pages = if ($logText -match '\((\d+)\s+pages?,') { $Matches[1] } else { '?' }
Write-Host ""
Write-Host "OK -> $jobname.pdf  ($pages pages)" -ForegroundColor Green

# Surface the things that are easy to miss in a multi-pass build.
$undef = Select-String -Path "build\$jobname.log" -Pattern 'Citation .* undefined|Reference .* undefined'
if ($undef) {
    Write-Host "`nUNDEFINED refs/citations ($($undef.Count)):" -ForegroundColor Yellow
    $undef | ForEach-Object { "  " + $_.Line.Trim() }
} else {
    Write-Host 'All references and citations resolved.' -ForegroundColor Green
}
$missing = Select-String -Path 'sections\*.tex','appendix\*.tex' -Pattern 'missingfig'
if ($missing) {
    Write-Host "`nPLACEHOLDER figures still in the paper:" -ForegroundColor Yellow
    $missing | ForEach-Object { "  $($_.Filename):$($_.LineNumber)  $($_.Line.Trim())" }
}
