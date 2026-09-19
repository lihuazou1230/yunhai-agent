<#
  云海工作台 · 打包「Agent 后端部署包」（服务器形态：IIS 同源子应用）
  ================================================================
  本机执行，产出一个 zip：拖进 RDP 会话解压、右键 install.ps1 运行即完成部署。

  用法（本机 PowerShell，无需管理员）：
      powershell -NoProfile -ExecutionPolicy Bypass -File deploy\build-agent-package.ps1

  产物：
      agent-deploy-package.zip        ← 交付物
      build/agent-deploy-package/     ← 中间暂存目录（已 gitignore，方便肉眼核对内容）

  参数：
      -InstallDir 'C:\yunhai-agent'   包里 install.ps1 的默认安装目录（写进说明与预检提示）
      -SkipVerify                     跳过预检（不建议）

  注意：这个包**不含 torch / sentence-transformers**（服务器形态向量走 API），
        也不含 data\（向量库与本机 bge 的维度不兼容，服务器上重新上传文档即可）。
#>
[CmdletBinding()]
param(
    [string]$InstallDir = 'C:\yunhai-agent',
    [switch]$SkipVerify
)

$ErrorActionPreference = 'Stop'

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Fail($msg) { throw $msg }

$repoRoot = Split-Path -Parent $PSScriptRoot
$deploySrc = Join-Path $repoRoot 'deploy'
$staging = Join-Path $repoRoot 'build\agent-deploy-package'
$zipPath = Join-Path $repoRoot 'agent-deploy-package.zip'

Write-Host ''
Write-Host '========================================' -ForegroundColor White
Write-Host ' 云海工作台 · 打包 Agent 部署包' -ForegroundColor White
Write-Host '========================================' -ForegroundColor White

Write-Step '1/4 校验源文件'
$required = @(
    'app\main.py',
    'requirements-server.txt',
    '.env.example'
)
foreach ($rel in $required) {
    if (-not (Test-Path (Join-Path $repoRoot $rel))) { Fail "缺少 $rel" }
}
foreach ($name in @('install.ps1', 'web.config')) {
    if (-not (Test-Path (Join-Path $deploySrc $name))) { Fail "缺少 deploy\$name" }
}
Write-Ok "源文件齐全（app\ + requirements-server.txt + deploy\install.ps1 + deploy\web.config）"

Write-Step '2/4 组装暂存目录'
if (Test-Path $staging) { Remove-Item $staging -Recurse -Force }
New-Item -ItemType Directory -Path $staging -Force | Out-Null

# app\ 里的 __pycache__ / *.pyc 不进包（服务器上会自己生成）；data\ 也不进包
$appStage = Join-Path $staging 'app'
New-Item -ItemType Directory -Path $appStage -Force | Out-Null
$appFiles = Get-ChildItem -Path (Join-Path $repoRoot 'app') -Recurse -File |
    Where-Object { $_.FullName -notmatch '\\__pycache__\\' -and $_.Extension -ne '.pyc' }
foreach ($file in $appFiles) {
    $rel = $file.FullName.Substring((Join-Path $repoRoot 'app').Length).TrimStart('\')
    $dest = Join-Path $appStage $rel
    $destDir = Split-Path -Parent $dest
    if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir -Force | Out-Null }
    Copy-Item -Path $file.FullName -Destination $dest -Force
}
Write-Ok "app\ 已收集（$($appFiles.Count) 个 .py，已排除 __pycache__）"

foreach ($name in @('requirements-server.txt', '.env.example', 'README.md')) {
    $src = Join-Path $repoRoot $name
    if (Test-Path $src) { Copy-Item -Path $src -Destination (Join-Path $staging $name) -Force }
}
foreach ($name in @('install.ps1', 'web.config', '部署说明.md')) {
    $src = Join-Path $deploySrc $name
    if (Test-Path $src) { Copy-Item -Path $src -Destination (Join-Path $staging $name) -Force }
}

$staged = @(Get-ChildItem -Path $staging -Recurse -File)
$sizeKb = [Math]::Round(($staged | Measure-Object -Property Length -Sum).Sum / 1KB, 1)
Write-Ok "共 $($staged.Count) 个文件，$sizeKb KB"

Write-Step '3/4 预检'
# 服务器是 Windows Server 2012 R2 + Windows PowerShell 4.0：无 BOM 的 UTF-8 会被按 GBK 读，
# 中文提示全成乱码。先补 BOM，再做后面的解析/校验
$utf8Bom = New-Object System.Text.UTF8Encoding($true)
foreach ($name in @('install.ps1', '部署说明.md')) {
    $path = Join-Path $staging $name
    if (-not (Test-Path $path)) { continue }
    $bytes = [System.IO.File]::ReadAllBytes($path)
    $hasBom = $bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF
    if (-not $hasBom) {
        $text = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
        [System.IO.File]::WriteAllText($path, $text, $utf8Bom)
        Write-Ok "$name 已补 UTF-8 BOM"
    } else {
        Write-Ok "$name 已带 UTF-8 BOM"
    }
}

if (-not $SkipVerify) {
    # install.ps1 要在服务器上跑，本机先做语法解析，别把打不开的脚本发出去
    $installPath = Join-Path $staging 'install.ps1'
    $installText = [System.IO.File]::ReadAllText($installPath, [System.Text.Encoding]::UTF8)
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseInput($installText, $installPath, [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors -and $errors.Count -gt 0) {
        Fail "install.ps1 语法错误：$($errors[0].Message)（行 $($errors[0].Extent.StartLineNumber)）"
    }
    Write-Ok 'install.ps1 语法解析通过'

    # web.config 必须是合法 XML —— IIS 遇到不合法配置是 500.19，现场排起来很费劲
    $configPath = Join-Path $staging 'web.config'
    try {
        [xml]$config = Get-Content $configPath -Raw -Encoding UTF8
    } catch {
        Fail "web.config 不是合法 XML：$($_.Exception.Message)"
    }
    if (-not $config.configuration.'system.webServer'.rewrite) {
        Fail 'web.config 里没有 rewrite 节点，IIS 反代配不起来'
    }
    Write-Ok 'web.config XML 合法且带 rewrite 节点'

    # 占位符必须还在（install.ps1 靠 replace 渲染；被谁手改成写死端口就会装错）
    $configText = Get-Content $configPath -Raw -Encoding UTF8
    if ($configText -notmatch [regex]::Escape('__AGENT_PORT__')) {
        Fail 'web.config 里找不到占位符 __AGENT_PORT__ —— 它必须由 install.ps1 渲染'
    }
    Write-Ok 'web.config 占位符完整（__AGENT_PORT__）'
}

Write-Step '4/4 生成 zip'
if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
Compress-Archive -Path (Join-Path $staging '*') -DestinationPath $zipPath -CompressionLevel Optimal
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($zipPath)
try {
    $entries = $zip.Entries
    $zipKb = [Math]::Round((Get-Item $zipPath).Length / 1KB, 1)
    # Windows PowerShell 的 Compress-Archive 用反斜杠做条目分隔符，比较前统一成正斜杠
    $names = @($entries | ForEach-Object { $_.FullName -replace '\\', '/' })
    Write-Ok "$zipPath（$zipKb KB，$($entries.Count) 个条目）"
    foreach ($need in @('install.ps1', 'web.config', 'requirements-server.txt', '部署说明.md', 'app/main.py', 'app/url_prefix.py')) {
        if (-not ($names | Where-Object { $_ -eq $need })) { Fail "zip 里缺少 $need" }
    }
    # torch 不该出现：它代表误用了本机那份 requirements.txt
    if (($names | Where-Object { $_ -eq 'requirements.txt' })) {
        Fail 'zip 里混进了 requirements.txt（含 torch，服务器上装不上）—— 只该带 requirements-server.txt'
    }
    Write-Ok 'zip 结构校验通过'
} finally {
    $zip.Dispose()
}

Write-Host ''
Write-Host '========================================' -ForegroundColor White
Write-Host ' 打包完成' -ForegroundColor Green
Write-Host '========================================' -ForegroundColor White
Write-Host " 交付物 : $zipPath"
Write-Host " 下一步 : RDP 登录 124.220.159.58 → 把 zip 粘进远程会话 → 解压到任意目录（如桌面）"
Write-Host "          → 右键 install.ps1「使用 PowerShell 运行」（必须管理员）"
Write-Host "          → 默认装到 $InstallDir，IIS 子应用挂在 80 端口的 /yhai"
Write-Host ''
Write-Host ' 之后   : 部署包构建时已把前端默认基地址写成 http://124.220.159.58/yhai；' -ForegroundColor Cyan
Write-Host '          重新打前端包（pnpm deploy:package）并重新部署 /workspace/ 即可生效。' -ForegroundColor Cyan
Write-Host ''
