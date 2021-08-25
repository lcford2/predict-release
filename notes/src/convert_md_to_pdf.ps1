# Author: Lucas Ford
# Purpose: Convert markdown to pdf files using pandoc
# Date: 08/03/2021

# get user provided arg
$md_file=$args[0]

# must provide an argument, else exit gracefully and help
if ($null -eq $md_file) {
    Write-Host "Must provide path to md file."
    Write-Host "e.g. .\convert_md_to_pdf.ps1 ..\md\example.md"
} else {
    # replace all 'md' with 'pdf'
    # would be safer to only replace '.md' with '.pdf'
    # but I know that the diretory structure I am using 
    # has markdown files in an 'md' folder and pdf files
    # in a 'pdf' folder so this is two birds
    # with one stone
    $pdf_file=$md_file -replace "md", "pdf"

    # update user on whats going on
    Write-Host "Converting" $md_file, "to", $pdf_file
    # luafilter here parses the md file and replaces 
    # {{}} fields with proper text.
    $lua_file="$PSScriptRoot/currentdate.lua"

    pandoc $md_file -o $pdf_file --from markdown --template eisvogel `
        --listing --lua-filter=$lua_file -V papersize:letter
    return
}