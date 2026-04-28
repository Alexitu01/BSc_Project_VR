# Run-SHARP-Batch.ps1

# Folder containing the images
$inputFolder = "C:\Uni stuff\BSc_Project_VR\Projection\faces"

# Output folder
$outputFolder = "C:\Uni stuff\BSc_Project_VR\Projection\ml-sharp\gaussians"

# Model path
$modelPath = "C:\Uni stuff\BSc_Project_VR\Projection\ml-sharp\sharp.pt"

# Get all files in the folder
Get-ChildItem -Path $inputFolder -File | ForEach-Object {

    $inputImage = $_.FullName

    Write-Host "Processing $inputImage ..."

    sharp predict `
        -i "$inputImage" `
        -o "$outputFolder" `
        -c "$modelPath"
}