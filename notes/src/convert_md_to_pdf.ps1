# Author: Lucas Ford
# Purpose: Convert markdown to pdf files using pandoc
# Date: 08/03/2021

$md_file=$args[0]

if ($null -eq $md_file) {
    Write-Host "Must provide path to md file."
    Write-Host "e.g. .\convert_md_to_pdf.ps1 ..\md\example.md"
} else {
    $pdf_file=$md_file -replace "md", "pdf"

    Write-Host "Converting" $md_file, "to", $pdf_file
    pandoc $md_file -o $pdf_file --from markdown --template eisvogel `
        --listing --lua-filter=currentdate.lua
    return
}