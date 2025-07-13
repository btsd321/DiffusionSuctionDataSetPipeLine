# PowerShell script to activate DiffusionSuctionDataSetPipeLine environment
Write-Host "Activating DiffusionSuctionDataSetPipeLine environment..." -ForegroundColor Green

# Add Anaconda to PATH
$env:PATH = "D:\SoftWare\anaconda3\Scripts;D:\SoftWare\anaconda3;D:\SoftWare\anaconda3\Library\bin;" + $env:PATH

# Activate the conda environment
& conda activate D:\Project\DiffusionSuctionDataSetPipeLine\diffusion_suction_conda

Write-Host "Environment activated successfully!" -ForegroundColor Green
Write-Host "You can now use bpy, pybullet, opencv, and other packages." -ForegroundColor Yellow
