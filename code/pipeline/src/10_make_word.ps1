param(
    [string]$InputPdf = "E:\Research\iv_surface\research_pipeline\paper\main.pdf",
    [string]$OutputDocx = "E:\Research\iv_surface\research_pipeline\output\word\iv_surface_revised.docx"
)

$ErrorActionPreference = "Stop"
$inputPath = (Resolve-Path -LiteralPath $InputPdf).Path
$outputDirectory = Split-Path -Parent $OutputDocx
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null

$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($inputPath, $false, $true, $false)
    $document.SaveAs2($OutputDocx, 16)
}
finally {
    if ($null -ne $document) {
        $document.Close($false)
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

$item = Get-Item -LiteralPath $OutputDocx
Write-Output ("Created {0} ({1:N0} bytes)" -f $item.FullName, $item.Length)
