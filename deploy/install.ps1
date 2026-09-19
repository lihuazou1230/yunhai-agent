<#
  云海工作台 · Agent 后端一键部署（IIS 同源子应用 /yhai，不开新端口）
  ================================================================
  阶段：第十阶段之后（把 agent 从"本机跑"升级成"服务器上跑"）

  为什么需要这个脚本
  ------------------
  部署到 http://124.220.159.58/workspace/ 的页面里，AI 助手页原本默认连
  http://127.0.0.1:8000 —— 那是**访问者自己的电脑**。而且浏览器从公网页面
  访问回环地址还要过 Local Network Access 权限（HTTP 页面连申请资格都没有），
  所以"本机 agent + 部署页"这条路在当前 Chrome 上根本走不通。
  正解：agent 跑在这台服务器上，并且**复用 80 端口**由 IIS 暴露成同源子路径
  /yhai —— 不动腾讯云安全组、不需要 CORS、桌面壳的 CSP 也天然满足。

  为什么用 ARR 反代，而不是 HttpPlatformHandler
  --------------------------------------------
  HPH v1.2 自带 **8KB 输出缓冲且无法关闭**（微软从未发布修好的 v1.3），
  用它扛 SSE = 回答憋到最后一次性吐。在 handler 上加 responseBufferLimit="0"
  只能关掉 IIS 那层 4MB 缓冲，关不掉模块内部那 8KB。
  ARR 的响应缓冲是可配的（机器级 minResponseBuffer=0），所以走 ARR。
  代价：ARR 不管进程，所以第 7 步额外注册一个"开机自启 + 失败重启"的计划任务。

  它做了什么
  ----------
  0) 环境侦察（OS / PowerShell / IIS / 已装 Python）
  1) 准备 Python 3.10~3.12（缺则下载安装；包内自带 exe 时优先用包内的）
  2) 部署应用文件 + 建 venv + 装 requirements-server.txt（**不含 torch**）
  3) 写 .env（双 Key + EMBEDDER=api）
  4) 离开 IIS 先自检：直连 127.0.0.1 起一次，验依赖与 Key，再停掉
  5) 装 URL Rewrite + ARR，并用 appcmd 开机器级代理（minResponseBuffer=0 等）
  6) 建 IIS 子应用 /yhai + 独立应用池 + 渲染 web.config
  7) 注册常驻计划任务并拉起 uvicorn（127.0.0.1:8000）
  8) 经 IIS 探活 /yhai/api/health

  目录约定（刻意分开，**应用不放网站目录里**）
  ------------------------------------------
    C:\yunhai-agent\            ← 应用 + venv + data + .env + logs（不在网站根下）
    C:\inetpub\wwwroot\yhai\    ← IIS 子应用的物理目录，**只放一个 web.config**

  在服务器上的运行方式（**管理员**身份，任选其一）
    1. 右键 install.ps1 →「使用 PowerShell 运行」
    2. 管理员 PowerShell： powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1

  参数（都可省；省了会交互式询问 / 用默认值）
    -AppName yhai                 子应用名（= URL 里的路径段，改了要同步前端基地址）
    -SiteName "Default Web Site"  主站点名（默认自动探测 80 端口已启动的站点）
    -Port 80                      按端口探测主站点
    -AgentPort 8000               agent 本机监听端口（只监听 127.0.0.1，不对外）
    -InstallDir C:\yunhai-agent   agent 应用目录（站点外）
    -PythonExe C:\...\python.exe   指定已装好的 Python（给了就不下载）
    -PythonVersion 3.12.10        要下载安装的 Python 版本（3.10~3.12；3.13+ 不支持 2012 R2）
    -PythonTargetDir C:\Python312 安装到的目录（仅下载安装时用）
    -PipIndexUrl <url>            pip 源。**默认清华镜像**（国内服务器访问 pypi.org 常慢到像卡死）；
                                  想回到官方源就传 -PipIndexUrl ''
    -LlmApiKey / -LlmBaseUrl / -LlmModel              生成侧（默认 DeepSeek）
    -EmbedApiKey / -EmbedApiBaseUrl / -EmbedApiModel  向量侧（默认 SiliconFlow bge-m3）
      提示：Key 这几个参数**能不用就不用** —— 省掉时脚本会交互式提示你粘贴，那才是推荐路径。
      用参数传的话，整条命令行会留在 PowerShell 的 ConsoleHost_history.txt 里（明文）。
      真传过就事后 Clear-History 并删掉那个历史文件（见「部署说明.md」第六节）。
    -SkipArrInstall               跳过 URL Rewrite / ARR 的检测与安装
    -SkipSelfCheck                跳过第 4 步的离线自检（不建议）
    -Force                        重建 venv、重写 .env
    -NoPause                      跑完不等待按键
#>
[CmdletBinding()]
param(
    [string]$AppName = 'yhai',
    [string]$SiteName,
    [int]$Port = 80,
    [int]$AgentPort = 8000,
    [string]$InstallDir = 'C:\yunhai-agent',
    [string]$PythonExe,
    [string]$PythonVersion = '3.12.10',
    [string]$PythonTargetDir = 'C:\Python312',
    [string]$PipIndexUrl = 'https://pypi.tuna.tsinghua.edu.cn/simple',
    [string]$LlmApiKey,
    [string]$LlmBaseUrl = 'https://api.deepseek.com/v1',
    [string]$LlmModel = 'deepseek-chat',
    [string]$EmbedApiKey,
    [string]$EmbedApiBaseUrl = 'https://api.siliconflow.cn/v1',
    [string]$EmbedApiModel = 'BAAI/bge-m3',
    [switch]$SkipArrInstall,
    [switch]$SkipSelfCheck,
    [switch]$Force,
    [switch]$NoPause
)

$ErrorActionPreference = 'Stop'
# Invoke-WebRequest 的进度条在 RDP 会话里极慢，能把几 MB 的下载拖成几分钟
$ProgressPreference = 'SilentlyContinue'

# 官方 MSI 直链（2026-09 实测 200，并已核对 MSI 内的 ProductName/ProductVersion）。
# 网传的 ARRv3_0_setup_amd64_en-US.msi / 大写 HttpPlatformHandler_amd64.msi 实测 404，别用。
$UrlRewriteUrls = @(
    'https://download.microsoft.com/download/1/2/8/128E2E22-C1B9-44A4-BE2A-5859ED1D4592/rewrite_amd64_en-US.msi'
)
$UrlRewriteMsiNames = @('rewrite_amd64_en-US.msi', 'rewrite_2.0_rtw_x64.msi')
$ArrUrls = @(
    'https://download.microsoft.com/download/E/9/8/E9849D6A-020E-47E4-9FD0-A023E99B54EB/requestRouter_amd64.msi'
)
$ArrMsiNames = @('requestRouter_amd64.msi', 'ARRv3_0_setup_amd64_en-US.msi')

$TaskName = 'yunhai-agent'
$PoolName = 'yunhai-agent'

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg) { Write-Host "    [OK]   $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "    [警告] $msg" -ForegroundColor Yellow }
function Write-Info($msg) { Write-Host "    $msg" -ForegroundColor Gray }

function Assert-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw '当前不是管理员会话。请右键脚本 →「以管理员身份运行」，或在管理员 PowerShell 中执行。'
    }
}

function Enable-Tls12 {
    # 2012 R2 默认只启用到 TLS 1.0，download.microsoft.com / python.org 会直接掐断连接
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch {
        Write-Warn "无法启用 TLS 1.2（$($_.Exception.Message)），下载可能失败，可改用包内离线文件"
    }
}

function Write-TextFileNoBom {
    <#
      写 UTF-8 **不带 BOM** 的文本文件。

      坑：Windows PowerShell 5.1 的 Out-File -Encoding utf8 会写 BOM，2012 R2 的
      PowerShell 4.0 更是连 utf8NoBOM 都没有。.env 带 BOM 会让第一行的键名变成
      "\ufeffLLM_API_KEY"，pydantic 读不到 —— 现象是"Key 明明填了却报未配置"。
    #>
    param([string]$Path, [string]$Content)
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

function Get-PythonVersionOf {
    param([string]$Exe, [string[]]$ExtraArgs = @())
    try {
        $raw = & $Exe @ExtraArgs -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -ne 0) { return '' }
        return ([string]$raw).Trim()
    } catch {
        return ''
    }
}

function Find-UsablePython {
    <#
      找一个 3.10~3.12 的 Python。3.13+ 起官方最低要求 Windows 10，
      这台 2012 R2 装了也跑不起来，所以明确排除并提示，而不是装完才失败。
      （3.12 是**最后**支持 Server 2012 R2 的版本，别升级。）
    #>
    $candidates = New-Object System.Collections.Generic.List[hashtable]
    if ($PythonExe) { $candidates.Add(@{ Exe = $PythonExe; Args = @() }) }
    foreach ($guess in @(
            "$env:ProgramFiles\Python312\python.exe",
            "$env:ProgramFiles\Python311\python.exe",
            "$env:ProgramFiles\Python310\python.exe",
            'C:\Python312\python.exe', 'C:\Python311\python.exe', 'C:\Python310\python.exe')) {
        if (Test-Path $guess) { $candidates.Add(@{ Exe = $guess; Args = @() }) }
    }
    foreach ($name in @('python', 'python3')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $candidates.Add(@{ Exe = $cmd.Source; Args = @() }) }
    }
    $py = Get-Command 'py' -ErrorAction SilentlyContinue
    if ($py) {
        foreach ($ver in @('3.12', '3.11', '3.10')) { $candidates.Add(@{ Exe = $py.Source; Args = @("-$ver") }) }
    }

    foreach ($candidate in $candidates) {
        $version = Get-PythonVersionOf -Exe $candidate.Exe -ExtraArgs $candidate.Args
        if (-not $version) { continue }
        if ($version -match '^3\.(10|11|12)$') { return @{ Exe = $candidate.Exe; Args = $candidate.Args; Version = $version } }
        Write-Warn "跳过 Python $version（$($candidate.Exe)）：本服务在 2012 R2 上只支持 3.10~3.12"
    }
    return $null
}

function Install-Python {
    param([string]$Version, [string]$TargetDir)
    $exeName = "python-$Version-amd64.exe"

    # 1) 包内离线安装包优先（服务器出不了外网时全靠它）
    $local = $null
    foreach ($name in @($exeName, 'python-amd64.exe')) {
        $candidate = Join-Path $PSScriptRoot $name
        if (Test-Path $candidate) { $local = $candidate; Write-Ok "使用包内离线安装包：$name"; break }
    }

    # 2) 下载官方安装包
    if (-not $local) {
        Enable-Tls12
        $url = "https://www.python.org/ftp/python/$Version/$exeName"
        $target = Join-Path $env:TEMP $exeName
        try {
            Write-Info "下载 $url"
            Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing -TimeoutSec 900
            $local = $target
        } catch {
            throw @"
下载 Python $Version 失败：$($_.Exception.Message)
  三条出路（任选）：
   1) 在能上网的机器上下载 https://www.python.org/ftp/python/$Version/$exeName ，
      和本脚本放在同一目录后重跑；
   2) 在服务器上手动装好 Python 3.10~3.12（**不要装 3.13+**），
      然后用 -PythonExe "C:\Python312\python.exe" 重跑；
   3) 检查服务器能否访问外网（很多云主机默认只放行 80）。
"@
        }
    }

    Write-Info "静默安装 $local（约 1~3 分钟）"
    $logPath = Join-Path $env:TEMP 'python-install.log'
    # InstallAllUsers=1：装成全局，常驻任务/服务身份才访问得到
    # TargetDir：固定目录，脚本后面好定位；CompileAll 顺带把标准库编译一遍
    $args = @(
        '/quiet', 'InstallAllUsers=1', 'PrependPath=1', 'Include_test=0',
        'Include_launcher=1', 'InstallLauncherAllUsers=1', 'Include_pip=1', 'CompileAll=1',
        "TargetDir=$TargetDir", "/log", $logPath
    )
    $proc = Start-Process -FilePath $local -ArgumentList $args -Wait -PassThru
    # 0 = 成功；3010 = 成功但待重启
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        throw @"
Python 安装失败，退出码 $($proc.ExitCode)（日志：$logPath）。
  2012 R2 上最常见的原因是缺 **Universal C Runtime（UCRT）**：装 VC++ 2015-2022 Redist
  （x64）或打 KB2999226 / KB3118401，然后重跑本脚本。
"@
    }
    Write-Ok "Python $Version 安装完成（$TargetDir）"

    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
    if (Test-Path "$TargetDir\python.exe") {
        return @{ Exe = "$TargetDir\python.exe"; Args = @(); Version = (Get-PythonVersionOf -Exe "$TargetDir\python.exe") }
    }
    return $null
}

function Install-Venv {
    param([string]$Python, [string[]]$PythonArgs, [string]$Dir, [string]$IndexUrl, [switch]$Recreate)

    $venvDir = Join-Path $Dir '.venv'
    $venvPython = Join-Path $venvDir 'Scripts\python.exe'

    if ($Recreate -and (Test-Path $venvDir)) {
        Write-Warn '指定了 -Force，删除旧 venv 重建'
        Remove-Item $venvDir -Recurse -Force
    }

    if (Test-Path $venvPython) {
        Write-Ok "复用已有 venv（$venvDir）"
    } else {
        Write-Info "创建 venv：$venvDir"
        & $Python @PythonArgs -m venv $venvDir
        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) { throw "创建 venv 失败（退出码 $LASTEXITCODE）" }
        Write-Ok 'venv 创建完成'
    }

    $req = Join-Path $Dir 'requirements-server.txt'
    if (-not (Test-Path $req)) { throw "找不到 $req（包不完整）" }

    # pip 公共参数。为什么要放宽超时/重试：国内云主机访问 pypi.org 经常慢到"看着像卡死"，
    # 默认 15 秒超时会让它反复重试、进度长时间不动。
    $common = New-Object System.Collections.Generic.List[string]
    $common.Add('--disable-pip-version-check')
    $common.Add('--timeout'); $common.Add('60')
    $common.Add('--retries'); $common.Add('5')
    if ($IndexUrl) {
        $common.Add('-i'); $common.Add($IndexUrl)
        Write-Info "pip 源：$IndexUrl"
    } else {
        Write-Info 'pip 源：官方 pypi.org（-PipIndexUrl 被显式留空）'
    }
    $pipArgs = $common.ToArray()

    # 刻意不用 --quiet：这里一等就是几分钟，没有输出就没法判断是"在下载"还是"卡死了"
    Write-Info '升级 pip（约 10~60 秒）...'
    & $venvPython -m pip install --upgrade pip @pipArgs
    if ($LASTEXITCODE -ne 0) { throw "升级 pip 失败（退出码 $LASTEXITCODE）" }

    Write-Info '安装依赖（不含 torch，约 2~10 分钟；下面会逐条打印下载进度）...'
    & $venvPython -m pip install -r $req @pipArgs
    if ($LASTEXITCODE -ne 0) { throw "pip install 失败（退出码 $LASTEXITCODE）" }
    Write-Ok '依赖安装完成'
    return $venvPython
}

function New-EnvFile {
    param([string]$Dir, [hashtable]$Values, [switch]$Recreate)

    $envPath = Join-Path $Dir '.env'
    if ((Test-Path $envPath) -and -not $Recreate) {
        Write-Ok '.env 已存在 → 保留（要重写请加 -Force）'
        return $envPath
    }

    $lines = New-Object System.Collections.Generic.List[string]
    $lines.Add('# 云海工作台 · Agent 后端（服务器形态）—— 由 deploy\install.ps1 生成')
    $lines.Add('# 本文件含 API Key：别拷进 Git、别放进网站目录。')
    $lines.Add('')
    $lines.Add('# ---------- 生成侧（DeepSeek；任何 OpenAI 兼容接口都行）----------')
    $lines.Add("LLM_API_KEY=$($Values.LlmApiKey)")
    $lines.Add("LLM_BASE_URL=$($Values.LlmBaseUrl)")
    $lines.Add("LLM_MODEL=$($Values.LlmModel)")
    $lines.Add('')
    $lines.Add('# ---------- 向量化：走 API，服务器上不装 torch ----------')
    $lines.Add('# bge-m3 是 1024 维，和本机 bge-small-zh（512 维）不兼容 —— data\ 不能从本机拷过来')
    $lines.Add('EMBEDDER=api')
    $lines.Add("EMBED_API_KEY=$($Values.EmbedApiKey)")
    $lines.Add("EMBED_API_BASE_URL=$($Values.EmbedApiBaseUrl)")
    $lines.Add("EMBED_API_MODEL=$($Values.EmbedApiModel)")
    $lines.Add('')
    $lines.Add('# ---------- 子路径 ----------')
    $lines.Add('# ARR 反代会剥掉 /yhai 前缀（到这里的路径已是 /api/...），所以这项是**双保险**：')
    $lines.Add('# 万一以后换成不剥前缀的托管方式（HttpPlatformHandler / HTTP Bridge），照样能用。')
    $lines.Add("AGENT_URL_PREFIX=/$($Values.AppName)")
    $lines.Add('')
    $lines.Add('# ---------- 检索与兜底（与本地一致）----------')
    $lines.Add('FALLBACK_MODE=refuse')
    $lines.Add('')
    $lines.Add('# ---------- 跨域 ----------')
    $lines.Add('# 同源反代下压根不走 CORS；留着是为了一旦改成"另开端口直连"也能用。')
    $lines.Add('CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://tauri.localhost,tauri://localhost,http://124.220.159.58')
    $lines.Add('')

    Write-TextFileNoBom -Path $envPath -Content ($lines -join "`r`n")
    Write-Ok '.env 已生成（含 AGENT_URL_PREFIX）'
    return $envPath
}

function Write-AgentLauncher {
    <#
      生成启动包装脚本 run-agent.cmd。

      为什么要包一层：计划任务不会捕获 stdout/stderr，直接跑 python 的话
      后端的启动报错和 traceback 就彻底丢了。用 cmd 重定向到 logs\agent.log，
      排障时至少有东西可看（说明.md 里也把"看日志"写成了第一步）。
    #>
    param([string]$Dir, [string]$VenvPython, [int]$AgentPort)
    $logDir = Join-Path $Dir 'logs'
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
    $logFile = Join-Path $logDir 'agent.log'
    $cmdPath = Join-Path $Dir 'run-agent.cmd'
    $content = @(
        '@echo off',
        'rem 由 deploy\install.ps1 生成：常驻计划任务用这个脚本拉起 agent',
        'chcp 65001 >nul',
        "cd /d `"$Dir`"",
        "echo [%date% %time%] starting yunhai-agent >> `"$logFile`"",
        "`"$VenvPython`" -m uvicorn app.main:app --host 127.0.0.1 --port $AgentPort >> `"$logFile`" 2>&1",
        "echo [%date% %time%] yunhai-agent exited with %errorlevel% >> `"$logFile`""
    )
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    # .cmd 必须是 CRLF，且中文只在 echo 里 —— 注意本文件里没有中文输出，避免代码页问题
    [System.IO.File]::WriteAllText($cmdPath, ($content -join "`r`n") + "`r`n", $utf8NoBom)
    Write-Ok "启动脚本已生成：$cmdPath"
    return @{ Cmd = $cmdPath; Log = $logFile }
}

function Test-AgentDirectly {
    <#
      上 IIS 之前先把服务直接跑起来探活：依赖缺没缺、Key 对不对（真编码一次），
      都能在这里暴露，省得挂上 IIS 之后对着 502 猜。
    #>
    param([string]$Dir, [string]$VenvPython, [string]$Prefix, [int]$AgentPort)
    $logOut = Join-Path $env:TEMP 'yunhai-agent-selfcheck.log'
    $logErr = Join-Path $env:TEMP 'yunhai-agent-selfcheck.err.log'

    Write-Info "临时起一次 127.0.0.1:$AgentPort 探活..."
    $proc = Start-Process -FilePath $VenvPython `
        -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '127.0.0.1', '--port', "$AgentPort") `
        -WorkingDirectory $Dir -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $logOut -RedirectStandardError $logErr

    try {
        $health = $null
        for ($i = 1; $i -le 30; $i++) {
            Start-Sleep -Seconds 2
            if ($proc.HasExited) { break }
            try {
                $health = Invoke-RestMethod -Uri "http://127.0.0.1:$AgentPort$Prefix/api/health" -TimeoutSec 5
                break
            } catch { }
        }
        if (-not $health) {
            $tail = ''
            if (Test-Path $logErr) { $tail = (Get-Content $logErr -Tail 20 -Encoding UTF8) -join "`n" }
            throw "服务没起来。最后几行日志：`n$tail"
        }
        Write-Ok "health：status=$($health.status) embedder=$($health.embedder)/$($health.embedder_model) url_prefix=$($health.url_prefix)"
        if (-not $health.llm_configured) { Write-Warn 'LLM Key 未配置：问答会报「后端未配置 LLM Key」' }

        # 真编码一次：验证 EMBED_API_KEY 真的可用（health 本身不调外部 API）
        try {
            $warm = Invoke-RestMethod -Uri "http://127.0.0.1:$AgentPort$Prefix/api/health/embedder" -TimeoutSec 60
            Write-Ok "向量自检通过：$($warm.embedder) dim=$($warm.dim)"
        } catch {
            Write-Warn "向量自检失败：$($_.Exception.Message)（检查 EMBED_API_KEY 是否有效、服务器能否访问 $EmbedApiBaseUrl）"
        }
    } finally {
        if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
    }
}

# ---------------- 安装 IIS 模块（URL Rewrite + ARR）----------------

function Test-UrlRewriteInstalled {
    if (Test-Path 'HKLM:\SOFTWARE\Microsoft\IIS Extensions\URL Rewrite') { return $true }
    if (Test-Path (Join-Path $env:windir 'system32\inetsrv\rewrite.dll')) { return $true }
    return $false
}

function Test-ArrInstalled {
    if (Test-Path 'HKLM:\SOFTWARE\Microsoft\IIS Extensions\Application Request Routing') { return $true }
    if (Test-Path (Join-Path $env:windir 'system32\inetsrv\requestRouter.dll')) { return $true }
    return $false
}

function Install-MsiModule {
    param(
        [string[]]$Urls,
        [string[]]$LocalNames,
        [string]$Label,
        [string]$OfflineHint
    )
    $msi = $null
    foreach ($name in $LocalNames) {
        $local = Join-Path $PSScriptRoot $name
        if (Test-Path $local) { $msi = $local; Write-Ok "使用包内离线安装包：$name"; break }
    }
    if (-not $msi) {
        Enable-Tls12
        $target = Join-Path $env:TEMP $LocalNames[0]
        foreach ($url in $Urls) {
            try {
                Write-Info "下载 $url"
                Invoke-WebRequest -Uri $url -OutFile $target -UseBasicParsing -TimeoutSec 300
                $msi = $target
                break
            } catch {
                Write-Warn "下载失败：$($_.Exception.Message)"
            }
        }
    }
    if (-not $msi) { throw "无法获取 $Label 安装包（服务器可能访问不了外网）。$OfflineHint" }

    Write-Info "静默安装 $msi"
    $proc = Start-Process -FilePath 'msiexec.exe' -ArgumentList @('/i', "`"$msi`"", '/quiet', '/norestart') -Wait -PassThru
    if ($proc.ExitCode -ne 0 -and $proc.ExitCode -ne 3010) {
        throw "$Label 安装失败，msiexec 退出码 $($proc.ExitCode)"
    }
    Write-Ok "$Label 安装完成"
}

function Set-ArrMachineConfig {
    <#
      机器级设置只能写 applicationHost.config —— system.webServer/proxy 是机器范围的节，
      写进 web.config 会直接 500.19 / HRESULT 0x80070021（配置节被锁定）。

      属性名说明：enabled 与 minResponseBuffer 有官方出处；timeout 与 preserveHostHeader
      的确切 schema 名未取得公开原文，所以这四条都是**尽力而为**：设不上只警告不中断，
      并在最后把 IIS 管理器里的手工路径打出来。
    #>
    $appcmd = Join-Path $env:windir 'system32\inetsrv\appcmd.exe'
    if (-not (Test-Path $appcmd)) { Write-Warn '找不到 appcmd.exe'; return }

    $settings = @(
        @{ Arg = '/enabled:"True"'; Desc = '开代理（默认是关的！）' },
        @{ Arg = '/minResponseBuffer:"0"'; Desc = '关掉响应缓冲阈值（SSE 关键）' },
        @{ Arg = '/timeout:"00:30:00"'; Desc = '抬高代理超时（默认 120s 会截断长回答）' },
        @{ Arg = '/preserveHostHeader:"True"'; Desc = '保留原始 Host 头' }
    )
    foreach ($item in $settings) {
        $cmdArgs = @('set', 'config', '-section:system.webServer/proxy', $item.Arg, '/commit:apphost')
        $output = & $appcmd @cmdArgs 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "proxy $($item.Arg)（$($item.Desc)）"
        } else {
            Write-Warn "proxy $($item.Arg) 设置失败（$($item.Desc)）：$($output -join ' ')"
        }
    }
    Write-Info '若上面有失败项，可手工设置：IIS 管理器 → 服务器节点 → Application Request Routing Cache → Server Proxy Settings'
    Write-Info '  → 勾 Enable proxy；Response buffer threshold (KB) 填 0；Time-out (seconds) 填 1800 → Apply'
}

# ---------------- 常驻计划任务 ----------------

function Register-AgentTask {
    param([string]$LauncherCmd, [string]$TaskLabel, [string]$Dir)
    <#
      开机自启 + 失败重启的计划任务（不依赖任何第三方服务包装器）。

      用 ScheduledTasks 模块而不是 schtasks.exe：后者拼引号极易出错（路径带空格就炸）。
      -ExecutionTimeLimit 0 = 不限时（长回答不会被任务计划掐断）；
      -RestartCount/-RestartInterval = 进程崩了自动重拉。
    #>
    $action = New-ScheduledTaskAction -Execute $LauncherCmd
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    try {
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -ExecutionTimeLimit ([TimeSpan]::Zero) -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
            -MultipleInstances IgnoreNew
    } catch {
        Write-Warn "部分任务设置不受支持（$($_.Exception.Message)），退回最小设置"
        $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit ([TimeSpan]::Zero)
    }
    Register-ScheduledTask -TaskName $TaskLabel -Action $action -Trigger $trigger `
        -Principal $principal -Settings $settings -Force `
        -Description "云海工作台 Agent 后端（uvicorn，仅监听 127.0.0.1，由 IIS 子应用 /yhai 反代）" | Out-Null
    Write-Ok "计划任务 $TaskLabel 已注册（开机自启、SYSTEM、失败重启）"
}

function Restart-AgentTaskProcess {
    <#
      停任务 + 兜底清掉可能残留的 uvicorn 进程。

      为什么需要兜底：任务动作是 cmd 包装脚本，个别环境下结束任务只结束 cmd，
      python 还占着端口 —— 下次启动就会 "address already in use"。
      只杀命令行里带本应用目录的 python，不碰机器上别的 Python 进程。
    #>
    param([string]$TaskLabel, [string]$Dir)
    Stop-ScheduledTask -TaskName $TaskLabel -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Get-WmiObject Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -and $_.CommandLine -like "*$Dir*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

function Wait-AgentPort {
    param([int]$AgentPort, [string]$Prefix, [int]$TimeoutSeconds = 90)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:$AgentPort$Prefix/api/health" -TimeoutSec 5
            return $resp
        } catch { }
    }
    return $null
}

# ---------------- IIS 站点/应用 ----------------

function Get-SiteBindingInfo {
    <#
      取站点绑定串（形如 *:80:）。坑：Get-Website 返回的是 IIS 配置对象，
      $site.Bindings 直接 -join 只会得到类型名；绑定元素要经 .Collection 取，
      属性名 bindingInformation 大小写也可能不同，两种都试。
    #>
    param($Site)
    $result = New-Object System.Collections.Generic.List[string]
    try {
        foreach ($b in @(Get-WebBinding -Name $Site.Name -ErrorAction Stop)) {
            if ($b.bindingInformation) { $result.Add([string]$b.bindingInformation) }
        }
    } catch { }
    if ($result.Count -eq 0 -and $Site.Bindings) {
        $items = @()
        if ($Site.Bindings.PSObject.Properties['Collection']) { $items = @($Site.Bindings.Collection) } else { $items = @($Site.Bindings) }
        foreach ($item in $items) {
            if ($item -is [string]) { $result.Add($item); continue }
            foreach ($prop in @('bindingInformation', 'BindingInformation')) {
                if ($item.PSObject.Properties[$prop]) { $result.Add([string]$item.$prop); break }
            }
        }
    }
    return $result
}

function Test-SiteBindsPort {
    param($Site, [int]$PortNumber)
    foreach ($info in (Get-SiteBindingInfo -Site $Site)) {
        if ($info -match ":${PortNumber}(:|$)") { return $true }
    }
    return $false
}

function Get-MainSite {
    if ($SiteName) {
        $site = Get-Website -Name $SiteName -ErrorAction SilentlyContinue
        if (-not $site) { throw "找不到名为「$SiteName」的 IIS 站点，请用 -SiteName 指定正确名称。" }
        return $site
    }
    $all = @(Get-Website)
    if ($all.Count -eq 0) { throw 'IIS 里一个站点都没有，请先在 IIS 管理器里建好主站。' }

    $started = @($all | Where-Object { $_.State -eq 'Started' })
    $picked = $null
    if ($started.Count -gt 0) {
        $picked = $started | Where-Object { Test-SiteBindsPort -Site $_ -PortNumber $Port } | Select-Object -First 1
    }
    if (-not $picked -and $started.Count -eq 1) {
        $picked = $started[0]
        Write-Warn "没取到 $Port 端口绑定，但只有一个已启动站点，按其部署；请复核它确实是主站。"
    }
    if (-not $picked) {
        $picked = $all | Where-Object { Test-SiteBindsPort -Site $_ -PortNumber $Port } | Select-Object -First 1
        if ($picked) { Write-Warn "站点 $($picked.Name) 当前不是 Started 状态，部署完请确认它已启动。" }
    }
    if (-not $picked) {
        $list = (($all | ForEach-Object { "$($_.Name)[$(@(Get-SiteBindingInfo -Site $_) -join ' ')]" }) -join '; ')
        throw "没有找到绑定 $Port 端口的站点。现有站点：$list。请用 -SiteName 手动指定。"
    }
    return $picked
}

function Get-SiteRootPath {
    param($Site)
    $path = $Site.PhysicalPath
    if (-not $path) {
        try { $path = (Get-Item "IIS:\Sites\$($Site.Name)" -ErrorAction Stop).PhysicalPath } catch { }
    }
    if (-not $path) { throw "取不到站点 $($Site.Name) 的物理路径。" }
    return [Environment]::ExpandEnvironmentVariables($path).TrimEnd('\')
}

function Get-WebAppPath {
    param($App)
    foreach ($prop in @('Path', 'Name', 'PSChildName')) {
        if ($App.PSObject.Properties[$prop]) {
            $value = [string]$App.$prop
            if ($value) { return '/' + $value.Trim('/') }
        }
    }
    return ''
}

function Test-WebApplicationExists {
    param($Site, [string]$Name)
    try {
        foreach ($app in @(Get-WebApplication -Site $Site.Name -ErrorAction Stop)) {
            if ((Get-WebAppPath -App $app) -eq "/$Name") { return $true }
        }
    } catch { }
    try {
        $app = Get-WebConfiguration -PSPath "IIS:\Sites\$($Site.Name)" `
            -Filter "system.applicationHost/sites/site/application[@path='/$Name']" -ErrorAction Stop
        if ($app) { return $true }
    } catch { }
    return $false
}

# ---------------- 主流程 ----------------

function Invoke-Deploy {
    Write-Host ''
    Write-Host '========================================' -ForegroundColor White
    Write-Host " 云海工作台 · Agent 后端部署（IIS 子应用 /$AppName）" -ForegroundColor White
    Write-Host '========================================' -ForegroundColor White

    Write-Step '0/8 环境侦察'
    Assert-Administrator
    Write-Ok '管理员权限'
    Write-Info "系统：$([Environment]::OSVersion.VersionString)"
    Write-Info "PowerShell：$($PSVersionTable.PSVersion)"

    $pkgRoot = $PSScriptRoot
    foreach ($need in @('app\main.py', 'requirements-server.txt', 'web.config')) {
        if (-not (Test-Path (Join-Path $pkgRoot $need))) {
            throw "包不完整：找不到 $need。请确认 zip 解压完整（app\、requirements-server.txt、web.config 与 install.ps1 同级）。"
        }
    }
    Write-Ok '部署包完整（app\ + requirements-server.txt + web.config）'

    Import-Module WebAdministration -ErrorAction Stop
    Write-Ok 'WebAdministration 模块已加载'

    Write-Step '1/8 准备 Python（3.10~3.12）'
    $python = Find-UsablePython
    if ($python) {
        Write-Ok "找到 Python $($python.Version)：$($python.Exe) $($python.Args -join ' ')"
    } else {
        Write-Warn "没找到可用的 Python 3.10~3.12，开始下载安装 $PythonVersion"
        $python = Install-Python -Version $PythonVersion -TargetDir $PythonTargetDir
        if (-not $python) { throw 'Python 安装后仍没探测到，请手动装好 3.10~3.12 再用 -PythonExe 指定路径重跑。' }
        Write-Ok "Python $($python.Version)：$($python.Exe)"
    }

    Write-Step "2/8 部署应用到 $InstallDir"
    if (-not (Test-Path $InstallDir)) { New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null }
    $appTarget = Join-Path $InstallDir 'app'
    if (Test-Path $appTarget) { Remove-Item $appTarget -Recurse -Force }
    Copy-Item -Path (Join-Path $pkgRoot 'app') -Destination $appTarget -Recurse -Force
    foreach ($name in @('requirements-server.txt', '.env.example', 'README.md')) {
        $src = Join-Path $pkgRoot $name
        if (Test-Path $src) { Copy-Item -Path $src -Destination (Join-Path $InstallDir $name) -Force }
    }
    foreach ($dir in @('data', 'logs')) {
        $full = Join-Path $InstallDir $dir
        if (-not (Test-Path $full)) { New-Item -ItemType Directory -Path $full -Force | Out-Null }
    }
    Write-Ok '应用文件已同步（data\ 与 logs\ 保留）'

    $venvPython = Install-Venv -Python $python.Exe -PythonArgs $python.Args -Dir $InstallDir -IndexUrl $PipIndexUrl -Recreate:$Force

    Write-Step '3/8 写 .env'
    $existing = Join-Path $InstallDir '.env'
    $fromEnvFile = { param($key)
        if (-not (Test-Path $existing)) { return '' }
        $m = Select-String -Path $existing -Pattern "^$key=(.*)$" | Select-Object -First 1
        if ($m) { return $m.Matches.Groups[1].Value.Trim() }
        return ''
    }
    if (-not $LlmApiKey) { $LlmApiKey = & $fromEnvFile 'LLM_API_KEY' }
    if (-not $LlmApiKey) {
        Write-Warn '没给 -LlmApiKey，也没读到旧的 .env'
        $LlmApiKey = Read-Host '    请粘贴 DeepSeek（或其它 OpenAI 兼容）API Key（留空=稍后手填 .env）'
    }
    if (-not $EmbedApiKey) { $EmbedApiKey = & $fromEnvFile 'EMBED_API_KEY' }
    if (-not $EmbedApiKey) {
        Write-Warn '没给 -EmbedApiKey，也没读到旧的 .env'
        $EmbedApiKey = Read-Host '    请粘贴 SiliconFlow API Key（bge-m3 向量用；留空=稍后手填 .env）'
    }
    $envPath = New-EnvFile -Dir $InstallDir -Recreate:$Force -Values @{
        AppName = $AppName
        LlmApiKey = $LlmApiKey
        LlmBaseUrl = $LlmBaseUrl
        LlmModel = $LlmModel
        EmbedApiKey = $EmbedApiKey
        EmbedApiBaseUrl = $EmbedApiBaseUrl
        EmbedApiModel = $EmbedApiModel
    }

    Write-Step '4/8 离线自检（挂 IIS 之前先确认服务本身能跑）'
    if ($SkipSelfCheck) {
        Write-Warn '已指定 -SkipSelfCheck，跳过'
    } else {
        Test-AgentDirectly -Dir $InstallDir -VenvPython $venvPython -Prefix "/$AppName" -AgentPort $AgentPort
    }

    Write-Step '5/8 IIS 模块：URL Rewrite + ARR'
    if (Test-UrlRewriteInstalled) {
        Write-Ok 'URL Rewrite 已安装'
    } elseif ($SkipArrInstall) {
        Write-Warn 'URL Rewrite 未安装，且指定了 -SkipArrInstall —— 反代规则写不了，页面会 500.19'
    } else {
        Install-MsiModule -Urls $UrlRewriteUrls -LocalNames $UrlRewriteMsiNames -Label 'URL Rewrite 2.0' `
            -OfflineHint '请在能上网的机器上下载 rewrite_amd64_en-US.msi，和本脚本放同一目录后重跑。'
    }
    if (Test-ArrInstalled) {
        Write-Ok 'ARR 已安装'
    } elseif ($SkipArrInstall) {
        Write-Warn 'ARR 未安装，且指定了 -SkipArrInstall —— 反代不生效'
    } else {
        Install-MsiModule -Urls $ArrUrls -LocalNames $ArrMsiNames -Label 'Application Request Routing 3.0' `
            -OfflineHint '请在能上网的机器上下载 requestRouter_amd64.msi，和本脚本放同一目录后重跑。'
        Write-Info '重启 IIS 使模块生效'
        & iisreset /noforce | Out-Null
    }
    Set-ArrMachineConfig

    Write-Step "6/8 建 IIS 子应用 /$AppName/"
    $site = Get-MainSite
    $siteRoot = Get-SiteRootPath -Site $site
    $hostDir = Join-Path $siteRoot $AppName
    Write-Ok "主站点：$($site.Name)"

    # 子应用物理目录**只放 web.config**：应用本体与 venv 都在站点外，
    # 静态文件处理器够不着 .env / .venv / app
    if (-not (Test-Path $hostDir)) { New-Item -ItemType Directory -Path $hostDir -Force | Out-Null }
    $template = Get-Content -Path (Join-Path $pkgRoot 'web.config') -Raw -Encoding UTF8
    $rendered = $template.Replace('__AGENT_PORT__', "$AgentPort")
    Write-TextFileNoBom -Path (Join-Path $hostDir 'web.config') -Content $rendered
    Write-Ok "web.config 已渲染（反代到 http://127.0.0.1:$AgentPort）"

    # 独立应用池：**刻意不碰主站那个池**（主站是二维码工具，不该被我们改回收策略）
    if (-not (Test-Path "IIS:\AppPools\$PoolName")) {
        New-WebAppPool -Name $PoolName | Out-Null
        Write-Ok "已新建独立应用池 $PoolName"
    }
    try {
        Set-ItemProperty -Path "IIS:\AppPools\$PoolName" -Name managedRuntimeVersion -Value ''
        Set-ItemProperty -Path "IIS:\AppPools\$PoolName" -Name processModel.idleTimeout -Value ([TimeSpan]::Zero)
        Write-Ok "应用池 $PoolName：No Managed Code、idleTimeout=0"
    } catch {
        Write-Warn "应用池参数设置失败：$($_.Exception.Message)"
    }

    if (Test-WebApplicationExists -Site $site -Name $AppName) {
        Set-ItemProperty -Path "IIS:\Sites\$($site.Name)\$AppName" -Name applicationPool -Value $PoolName
        Write-Ok '子应用已存在 → 复用（幂等），已指向独立应用池'
    } else {
        New-WebApplication -Site $site.Name -Name $AppName -PhysicalPath $hostDir -ApplicationPool $PoolName | Out-Null
        Write-Ok "子应用创建完成（应用池 $PoolName）"
    }
    & icacls $hostDir /grant 'IIS_IUSRS:(OI)(CI)(RX)' /T /Q /C | Out-Null

    Write-Step "7/8 常驻进程（计划任务 $TaskName，开机自启）"
    $launcher = Write-AgentLauncher -Dir $InstallDir -VenvPython $venvPython -AgentPort $AgentPort
    Register-AgentTask -LauncherCmd $launcher.Cmd -TaskLabel $TaskName -Dir $InstallDir
    Restart-AgentTaskProcess -TaskLabel $TaskName -Dir $InstallDir
    Start-ScheduledTask -TaskName $TaskName
    Write-Info '等待 uvicorn 就绪...'
    $health = Wait-AgentPort -AgentPort $AgentPort -Prefix "/$AppName"
    if ($health) {
        Write-Ok "后端已就绪：status=$($health.status) embedder=$($health.embedder)/$($health.embedder_model) docs=$($health.documents)"
    } else {
        Write-Warn "后端没在 $AgentPort 上就绪。查日志：$($launcher.Log)"
    }

    Write-Step "8/8 经 IIS 探活 /$AppName/api/health"
    $probeUrl = "http://127.0.0.1/$AppName/api/health"
    $iisOk = $false
    for ($i = 1; $i -le 10; $i++) {
        Start-Sleep -Seconds 2
        try {
            $resp = Invoke-WebRequest -Uri $probeUrl -UseBasicParsing -TimeoutSec 20
            Write-Ok "$probeUrl → HTTP $($resp.StatusCode)"
            $iisOk = $true
            break
        } catch {
            if ($i -eq 10) { Write-Warn "$probeUrl → $($_.Exception.Message)" }
        }
    }
    if (-not $iisOk) {
        Write-Warn @"
经 IIS 探活没成功。按顺序查这五样（说明.md「排障」一节有完整对照表）：
     1) 后端日志：$($launcher.Log)
     2) ARR 代理开了没：IIS 管理器 → 服务器节点 → Application Request Routing Cache → Server Proxy Settings
     3) URL Rewrite / ARR 装没装：%windir%\system32\inetsrv\ 下有 rewrite.dll、requestRouter.dll
     4) 子应用 /$AppName 的 web.config 里 <rewrite> 规则在不在（本脚本渲染的那份）
     5) 500.19 → web.config 里写了机器级节（如 proxy）；502.3 → 后端没起来
"@
    }

    Write-Host ''
    Write-Host '========================================' -ForegroundColor White
    Write-Host ' 部署完成' -ForegroundColor Green
    Write-Host '========================================' -ForegroundColor White
    Write-Host " 站点        : $($site.Name)"
    Write-Host " 子应用      : /$AppName/（同源反代，不占新端口）"
    Write-Host " 应用目录    : $InstallDir（站点外）"
    Write-Host " 托管目录    : $hostDir（只有 web.config）"
    Write-Host " 进程        : 计划任务 $TaskName（SYSTEM，开机自启；日志 $($launcher.Log)）"
    Write-Host ' 前端基地址  : http://124.220.159.58/yhai' -ForegroundColor Cyan
    Write-Host ''
    Write-Host ' 现在请在浏览器里复核：' -ForegroundColor Cyan
    Write-Host "   1) http://124.220.159.58/$AppName/api/health        ← 应返回 JSON（status=ok）"
    Write-Host "   2) http://124.220.159.58/workspace/ → AI 助手页点「检测」→ 应变成已连接"
    Write-Host "   3) 传一份文档 → 提问 → 回答应**逐字**出现（不是憋到最后一次性吐）"
    Write-Host "   4) http://124.220.159.58/                            ← 主站二维码工具应完好无损"
    Write-Host ''
    Write-Host ' 常用运维：' -ForegroundColor Cyan
    Write-Host "   重启后端 : Restart-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue; Start-ScheduledTask -TaskName $TaskName"
    Write-Host "   看日志   : Get-Content '$($launcher.Log)' -Tail 50"
    Write-Host "   改配置   : 编辑 $InstallDir\.env 后重启后端"
    Write-Host ''
}

$exitCode = 0
try {
    Invoke-Deploy
} catch {
    Write-Host ''
    Write-Host "部署失败：$($_.Exception.Message)" -ForegroundColor Red
    $exitCode = 1
}

if (-not $NoPause) {
    Write-Host ''
    Read-Host '按回车键关闭窗口' | Out-Null
}
exit $exitCode
