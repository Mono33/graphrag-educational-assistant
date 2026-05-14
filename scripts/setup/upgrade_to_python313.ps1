<#
═══════════════════════════════════════════════════════════════════════════════
  Python 3.13 upgrade for graphaixlearning — Plan A automation
═══════════════════════════════════════════════════════════════════════════════

  AUTHORS' NOTE — HOW TO RUN THIS SCRIPT (READ FIRST)
  ───────────────────────────────────────────────────

  This script is the automated version of "Plan A" from the SSL-fix
  conversation (May 10, 2026). It exists because Python 3.11.0's bundled
  OpenSSL 1.1.1q (EOL since Sep 2023) can't validate certs from
  intermediate CAs issued after that date — every TLS call to OpenRouter,
  PyPI, GlitchTip, etc. blew up with "Connection error.". Python 3.13
  ships OpenSSL 3.0.x+ which reads the Windows certificate store
  natively, eliminating the entire class of issue.

  ───── WHEN TO RUN ───────────────────────────────────────────────────────

  RUN THIS:
    * On a calm afternoon. NOT before a demo, NOT mid-feature.
    * After Plan B (`pip-system-certs`) has stabilised the current venv
      so you have a known-good baseline to compare against.
    * After running the dep pre-flight check (see below) so you know
      whether MSVC Build Tools will be needed.

  DON'T RUN THIS YET if:
    * You haven't installed Python 3.13 from python.org. The script
      checks for it but cannot install Python itself.
    * The pre-flight reports any package without wheels for 3.13 + Win64
      AND you don't have Microsoft C++ Build Tools installed.

  ───── PREREQUISITES ─────────────────────────────────────────────────────

  1. Install Python 3.13 from https://www.python.org/downloads/
     During the installer:
       ✓ Add python.exe to PATH
       ✓ Install py launcher
     After install, verify:
       py -3.13 --version       # should print Python 3.13.x
       py -3.13 -c "import ssl; print(ssl.OPENSSL_VERSION)"
                                # should print OpenSSL 3.0.x or 3.4.x

  2. Run the dep pre-flight check from your CURRENT (3.11) venv to know
     up-front whether the install will hit any source-build dead ends:

       $pkgs = @('gensim','node2vec','numpy','scipy','pandas',
                 'scikit-learn','lingua-language-detector','reportlab')
       foreach ($p in $pkgs) {
           Write-Host "`n--- $p ---" -ForegroundColor Cyan
           python -c "import urllib.request, json, re; d=json.loads(urllib.request.urlopen('https://pypi.org/pypi/$p/json').read()); v=d['info']['version']; print('latest:', v); files=[f['filename'] for f in d['releases'][v] if f['filename'].endswith('.whl')];
       def ok(fn):
           if 'py3-none-any' in fn or 'py2.py3-none-any' in fn: return True
           if 'win_amd64' not in fn: return False
           if 'cp313' in fn: return True
           m=re.search(r'cp3(\d+)-abi3-win_amd64', fn)
           return bool(m and int(m.group(1))<=13)
       hits=[fn for fn in files if ok(fn)];
       print('Compatible with 3.13 + Windows:', 'YES' if hits else 'NO')"
       }

     A wheel is 3.13-Windows-compatible if it's:
       * py3-none-any         → pure Python (works everywhere)
       * cp313-cp313-win_amd64 → built specifically for 3.13
       * cp3<N>-abi3-win_amd64 (N<=13) → stable-ABI wheel (forward-compat)

     If every package shows YES, no build tools needed. If any shows NO,
     either install MS C++ Build Tools first
       https://visualstudio.microsoft.com/visual-cpp-build-tools/
     or pin a slightly older version that has a 3.13 wheel.

  ───── HOW TO INVOKE ─────────────────────────────────────────────────────

  Open a fresh PowerShell terminal (so it doesn't inherit the active
  3.11 venv), then:

      cd C:\Users\louis\KBRAGold\graphaixlearning
      .\scripts\setup\upgrade_to_python313.ps1

  Optional flags (see param block):
      -Force        rebuild a failed venv-313 from scratch
      -SkipTests    skip pytest tests\unit at the end
      -SkipProbe    skip the LLM connectivity probe at the end
      -PythonExe    use a non-standard Python 3.13 path

  Examples:
      .\scripts\setup\upgrade_to_python313.ps1                    # standard
      .\scripts\setup\upgrade_to_python313.ps1 -Force             # rebuild
      .\scripts\setup\upgrade_to_python313.ps1 -SkipProbe         # offline
      .\scripts\setup\upgrade_to_python313.ps1 -PythonExe `
          "D:\Tools\Python313\python.exe"                         # custom path

  Expected runtime: 3–10 minutes, dominated by `pip install -e .[dev]`
  downloading the ~50 transitive deps. The script prints clearly-tagged
  step headers ([1/8]…[8/8]) so you can see where time is going.

  ───── WHAT THE SCRIPT DOES ──────────────────────────────────────────────

  Creates a fresh venv at  ..\..\venv-313  (sibling of the existing
  ..\..\venv), installs the project in editable mode with dev extras,
  smoke-imports the agent module tree, runs the unit-test suite, and
  runs the LLM connectivity probe to verify SSL works end-to-end.

  The existing  ..\..\venv  (Python 3.11) is NEVER touched — kept as a
  rollback if anything goes wrong. After several days of stable use on
  the new venv you can manually retire the old one:

      Remove-Item -Recurse -Force C:\Users\louis\KBRAGold\venv
      Rename-Item C:\Users\louis\KBRAGold\venv-313 venv

  IDE follow-up (Cursor / VS Code):
      Ctrl+Shift+P → "Python: Select Interpreter"
        → C:\Users\louis\KBRAGold\venv-313\Scripts\python.exe

  ───── ON FAILURE ────────────────────────────────────────────────────────

  The script aborts with a non-zero exit code and leaves venv-313 in a
  partially-installed state. To recover:

      Remove-Item -Recurse -Force C:\Users\louis\KBRAGold\venv-313
      # Fix the underlying cause (network, missing wheels, MSVC tools…)
      .\scripts\setup\upgrade_to_python313.ps1 -Force

  Your existing 3.11 venv at  C:\Users\louis\KBRAGold\venv  is untouched
  by any failure path, so the app keeps running on 3.11 in the meantime.

═══════════════════════════════════════════════════════════════════════════════

.SYNOPSIS
    Automate Plan A — migrate the graphaixlearning venv from Python 3.11 to 3.13.

.DESCRIPTION
    Creates a fresh venv at ``..\..\venv-313`` (sibling of the existing
    ``..\..\venv``), installs the project in editable mode with dev extras,
    runs the unit-test suite, and runs the LLM connectivity probe.

    The existing ``..\..\venv`` (Python 3.11) is **NOT touched** — kept as a
    rollback if anything goes wrong on 3.13. After several days of stable use
    on the new venv you can swap them manually:

        Remove-Item -Recurse -Force C:\Users\louis\KBRAGold\venv
        Rename-Item C:\Users\louis\KBRAGold\venv-313 venv

    This script is idempotent: re-running it deletes the partial ``venv-313``
    (only if explicitly requested with -Force) and rebuilds, never mutates
    the active 3.11 venv.

.PARAMETER Force
    Delete an existing ``venv-313`` directory before creating a fresh one.
    Without -Force, the script aborts if ``venv-313`` already exists, to
    protect against accidentally clobbering a half-good environment.

.PARAMETER SkipTests
    Skip the pytest run at the end. Useful when running the script for the
    first time without your full dev-dep tree (still runs the connectivity
    probe).

.PARAMETER SkipProbe
    Skip the LLM connectivity probe at the end. Useful when offline.

.PARAMETER PythonExe
    Path to the Python 3.13 interpreter. Defaults to ``py -3.13`` via the
    Windows ``py`` launcher (recommended). Pass an absolute path like
    ``C:\Python313\python.exe`` if your installation is non-standard.

.EXAMPLE
    # Standard run from the project root
    .\scripts\setup\upgrade_to_python313.ps1

.EXAMPLE
    # Force-rebuild a previously failed venv-313
    .\scripts\setup\upgrade_to_python313.ps1 -Force

.EXAMPLE
    # Use a non-default Python 3.13 location
    .\scripts\setup\upgrade_to_python313.ps1 -PythonExe "D:\Tools\Python313\python.exe"

.NOTES
    Prerequisites checked by the script:
      * Python 3.13.x interpreter reachable via ``py -3.13`` or -PythonExe.
      * Network access to PyPI (probe attempted before downloading deps).
      * Project root contains pyproject.toml + requirements.txt.

    On failure the script aborts with a non-zero exit code and leaves
    venv-313 in a broken state — re-run with -Force after fixing the
    underlying issue (network, missing wheels, etc.).
#>

[CmdletBinding()]
param(
    [switch]$Force,
    [switch]$SkipTests,
    [switch]$SkipProbe,
    [string]$PythonExe = ""
)

$ErrorActionPreference = "Stop"

# ---------------------------------------------------------------------------
# 0. Resolve project root regardless of where the script is invoked from.
# ---------------------------------------------------------------------------
# Script lives at <project>/scripts/setup/upgrade_to_python313.ps1
# So <project> = parent.parent of $PSScriptRoot.
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$NewVenvPath = Join-Path (Split-Path $ProjectRoot -Parent) "venv-313"
$OldVenvPath = Join-Path (Split-Path $ProjectRoot -Parent) "venv"

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Python 3.13 upgrade — graphaixlearning" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Project root      : $ProjectRoot"
Write-Host "  Existing venv     : $OldVenvPath  (preserved)"
Write-Host "  New venv (3.13)   : $NewVenvPath"
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Sanity-check the project layout.
# ---------------------------------------------------------------------------
$pyproject = Join-Path $ProjectRoot "pyproject.toml"
$requirements = Join-Path $ProjectRoot "requirements.txt"
if (-not (Test-Path $pyproject)) {
    throw "pyproject.toml not found at $pyproject — is the script in the right place?"
}
if (-not (Test-Path $requirements)) {
    throw "requirements.txt not found at $requirements — is the script in the right place?"
}

# ---------------------------------------------------------------------------
# 2. Locate Python 3.13.
# ---------------------------------------------------------------------------
function Resolve-Python313 {
    param([string]$Override)

    if ($Override) {
        if (-not (Test-Path $Override)) {
            throw "Provided -PythonExe path does not exist: $Override"
        }
        return $Override
    }

    # Try the Windows ``py`` launcher (installed by every python.org installer).
    $py = Get-Command py -ErrorAction SilentlyContinue
    if ($py) {
        # ``py -3.13 -c "import sys; print(sys.executable)"`` returns the path.
        $resolved = & py -3.13 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return $resolved.Trim()
        }
    }

    # Fallback: a bare ``python3.13`` on PATH (rare on Windows but possible).
    $cmd = Get-Command python3.13 -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }

    throw @"
Python 3.13 not found.

Install it from https://www.python.org/downloads/ (any 3.13.x). During
install, make sure ``Add python.exe to PATH`` and ``Install py launcher``
are both checked. Then re-run this script.

If 3.13 IS installed at a non-standard path, pass it explicitly:
    .\scripts\setup\upgrade_to_python313.ps1 -PythonExe "C:\path\to\python.exe"
"@
}

$Py313 = Resolve-Python313 -Override $PythonExe
Write-Host "[1/8] Python 3.13 interpreter: $Py313" -ForegroundColor Green

$pyVersion = & $Py313 --version 2>&1
$pyOpenSSL = & $Py313 -c "import ssl; print(ssl.OPENSSL_VERSION)"
Write-Host "       version  : $pyVersion"
Write-Host "       openssl  : $pyOpenSSL"
if ($pyVersion -notmatch "Python 3\.13\.") {
    throw "Expected Python 3.13.x but got '$pyVersion' — refusing to continue."
}
Write-Host ""

# ---------------------------------------------------------------------------
# 3. Handle existing venv-313.
# ---------------------------------------------------------------------------
if (Test-Path $NewVenvPath) {
    if ($Force) {
        Write-Host "[2/8] -Force passed — deleting existing $NewVenvPath" -ForegroundColor Yellow
        Remove-Item -Recurse -Force $NewVenvPath
    } else {
        throw @"
$NewVenvPath already exists.

Re-run with -Force to delete and rebuild it, or remove it manually:
    Remove-Item -Recurse -Force "$NewVenvPath"
"@
    }
} else {
    Write-Host "[2/8] No existing $NewVenvPath — clean install path." -ForegroundColor Green
}
Write-Host ""

# ---------------------------------------------------------------------------
# 4. Create the venv.
# ---------------------------------------------------------------------------
Write-Host "[3/8] Creating venv with $Py313 ..." -ForegroundColor Green
& $Py313 -m venv $NewVenvPath
if ($LASTEXITCODE -ne 0) {
    throw "venv creation failed (exit $LASTEXITCODE)."
}

$VenvPython = Join-Path $NewVenvPath "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "venv was created but $VenvPython does not exist — installer broken?"
}
Write-Host "       venv created: $VenvPython"
Write-Host ""

# ---------------------------------------------------------------------------
# 5. Upgrade pip + setuptools + wheel inside the new venv.
# ---------------------------------------------------------------------------
Write-Host "[4/8] Upgrading pip + setuptools + wheel..." -ForegroundColor Green
& $VenvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "pip/setuptools/wheel upgrade failed (exit $LASTEXITCODE). " +
          "If this is the SSL issue again, the OS cert store may not be " +
          "reachable for 3.13 either. See pip-system-certs notes in README."
}
Write-Host ""

# ---------------------------------------------------------------------------
# 6. Install pip-system-certs FIRST so the rest of the install uses the
#    Windows certificate store (mirrors what we did for the 3.11 venv).
#    On 3.13 this is usually unnecessary because OpenSSL 3.x reads the
#    Windows store natively, but it's a cheap safety net for corp networks.
# ---------------------------------------------------------------------------
Write-Host "[5/8] Installing pip-system-certs (defensive — OS cert store fallback)..." -ForegroundColor Green
& $VenvPython -m pip install --upgrade pip-system-certs
if ($LASTEXITCODE -ne 0) {
    Write-Host "       (non-fatal) pip-system-certs install failed; continuing." -ForegroundColor Yellow
}
Write-Host ""

# ---------------------------------------------------------------------------
# 7. Install the project itself with dev extras.
# ---------------------------------------------------------------------------
Write-Host "[6/8] Installing graphaixlearning + dev extras (this may take 3–10 min)..." -ForegroundColor Green
Push-Location $ProjectRoot
try {
    & $VenvPython -m pip install -e ".[dev]"
    if ($LASTEXITCODE -ne 0) {
        throw @"
Project install failed (exit $LASTEXITCODE).

Common causes on Python 3.13:
  * A C-extension dependency (most likely 'gensim' via 'node2vec') has no
    cp313 wheel for win_amd64 and pip is trying to compile from source.
    Fix: install Microsoft C++ Build Tools
      https://visualstudio.microsoft.com/visual-cpp-build-tools/
    OR pin a slightly older version known to ship a 3.13 wheel.

  * Pre-flight check (run from your CURRENT 3.11 venv to know in advance):
      python -c "import urllib.request, json; d=json.loads(urllib.request.urlopen('https://pypi.org/pypi/gensim/json').read()); v=d['info']['version']; print(v); print(any('cp313' in f['filename'] and 'win_amd64' in f['filename'] for f in d['releases'][v]))"

The new venv at $NewVenvPath is left in a broken state — delete it with:
    Remove-Item -Recurse -Force "$NewVenvPath"
or re-run this script with -Force.
"@
    }
} finally {
    Pop-Location
}
Write-Host ""

# ---------------------------------------------------------------------------
# 8. Sanity import — load the agent graph to surface any 3.13 incompat early.
# ---------------------------------------------------------------------------
Write-Host "[7/8] Smoke-importing aix.* modules..." -ForegroundColor Green
$smokeScript = @"
import sys
print('python:', sys.version.split()[0])

# Touch every major package so import errors surface here, not at runtime.
import aix.core.config
import aix.agent.graph.lesson_planner_graph
import aix.agent.agents.planner_agent
import aix.agent.agents.critic_agent
import aix.agent.agents.retrieval_grader_agent  # CORE 2 #9
import aix.api.main  # FastAPI app + lifespan

print('OK')
"@
$smokeOutput = & $VenvPython -c $smokeScript 2>&1
Write-Host $smokeOutput
if ($LASTEXITCODE -ne 0) {
    throw "Smoke import failed under Python 3.13 — see traceback above."
}
Write-Host ""

# ---------------------------------------------------------------------------
# 9. Run unit tests (optional).
# ---------------------------------------------------------------------------
if (-not $SkipTests) {
    Write-Host "[8/8] Running unit tests..." -ForegroundColor Green
    Push-Location $ProjectRoot
    try {
        & $VenvPython -m pytest tests\unit -v
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            Write-Host "WARNING: pytest failed under Python 3.13 (exit $LASTEXITCODE)." -ForegroundColor Yellow
            Write-Host "         The new venv is installed but has test regressions." -ForegroundColor Yellow
            Write-Host "         Investigate before swapping venvs." -ForegroundColor Yellow
        }
    } finally {
        Pop-Location
    }
} else {
    Write-Host "[8/8] -SkipTests passed — skipping pytest." -ForegroundColor Yellow
}
Write-Host ""

# ---------------------------------------------------------------------------
# 10. Run LLM connectivity probe (optional).
# ---------------------------------------------------------------------------
if (-not $SkipProbe) {
    Write-Host "Running LLM connectivity probe (verifies SSL on 3.13)..." -ForegroundColor Green
    $probeScript = @"
import asyncio, logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
from dotenv import load_dotenv
load_dotenv('.env')
from aix.core.connectivity_probe import _probe_once
asyncio.run(_probe_once())
"@
    Push-Location $ProjectRoot
    try {
        & $VenvPython -c $probeScript
    } finally {
        Pop-Location
    }
} else {
    Write-Host "-SkipProbe passed — skipping connectivity probe." -ForegroundColor Yellow
}

# ---------------------------------------------------------------------------
# Final guidance.
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Python 3.13 venv READY at: $NewVenvPath" -ForegroundColor Green
Write-Host "═══════════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host ""
Write-Host "  1. Activate the new venv in a fresh terminal:"
Write-Host "       & '$NewVenvPath\Scripts\Activate.ps1'"
Write-Host ""
Write-Host "  2. Point your IDE (Cursor/VS Code) at the new interpreter:"
Write-Host "       Ctrl+Shift+P → 'Python: Select Interpreter' → $VenvPython"
Write-Host ""
Write-Host "  3. Try a real lesson run end-to-end through uvicorn:"
Write-Host "       cd '$ProjectRoot'"
Write-Host "       python -m uvicorn aix.api.main:app --host 127.0.0.1 --port 8765 --log-level info"
Write-Host ""
Write-Host "  4. After several days of stable use, retire the old 3.11 venv:"
Write-Host "       Remove-Item -Recurse -Force '$OldVenvPath'"
Write-Host "       Rename-Item '$NewVenvPath' venv"
Write-Host ""
Write-Host "  5. Optional: bump pyproject.toml classifiers + ruff target-version"
Write-Host "     to mention py313 once you're committed (cosmetic, non-blocking)."
Write-Host ""
