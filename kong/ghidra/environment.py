"""Ghidra environment discovery — find Ghidra install and JDK."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def _java_version(java_home: str) -> int | None:
    """Return the major version of the JDK at *java_home*, or None on failure."""
    java_bin = Path(java_home) / "bin" / "java"
    if not java_bin.exists():
        return None
    try:
        result = subprocess.run(
            [str(java_bin), "-version"],
            capture_output=True, text=True, timeout=10,
        )
        # `java -version` prints to stderr, e.g. 'openjdk version "21.0.2"'
        output = result.stderr or result.stdout
        import re
        match = re.search(r'"(\d+)', output)
        return int(match.group(1)) if match else None
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


def find_java_home(min_version: int = 21) -> str | None:
    """Auto-detect a JDK 21+ installation for Ghidra.

    Checks in order:
      1. ``JAVA_HOME`` environment variable (if it points to JDK 21+)
      2. macOS: ``/usr/libexec/java_home -v 21+``
      3. Homebrew: ``brew --prefix openjdk@21``
      4. Homebrew: ``brew --prefix openjdk``
    """
    # 1. Existing JAVA_HOME — only if version is sufficient
    env_java = os.environ.get("JAVA_HOME")
    if env_java and Path(env_java).is_dir():
        ver = _java_version(env_java)
        if ver is not None and ver >= min_version:
            return env_java
        logger.debug("JAVA_HOME=%s is JDK %s, need %d+", env_java, ver, min_version)

    # 2. macOS java_home utility
    try:
        result = subprocess.run(
            ["/usr/libexec/java_home", "-v", "21+"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            java_home = result.stdout.strip()
            if java_home and Path(java_home).is_dir():
                ver = _java_version(java_home)
                if ver is not None and ver >= min_version:
                    return java_home
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3-4. Homebrew openjdk
    for pkg in ["openjdk@21", "openjdk"]:
        try:
            result = subprocess.run(
                ["brew", "--prefix", pkg],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                prefix = Path(result.stdout.strip())
                # Homebrew JDK layout: <prefix>/libexec/openjdk.jdk/Contents/Home
                jdk_home = prefix / "libexec" / "openjdk.jdk" / "Contents" / "Home"
                if jdk_home.is_dir():
                    return str(jdk_home)
                # Fallback: prefix itself might be JAVA_HOME
                if (prefix / "bin" / "java").exists():
                    return str(prefix)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    return None


def find_ghidra_install() -> str | None:
    """Auto-detect Ghidra installation directory.

    Checks in order:
      1. ``GHIDRA_INSTALL_DIR`` environment variable
      2. Homebrew: ``brew --prefix ghidra`` → ``<prefix>/libexec``
      3. Common paths: ``/opt/ghidra*``, ``/Applications/ghidra*``

    Returns the path as a string, or ``None`` if not found.
    """
    # 1. Environment variable
    env_dir = os.environ.get("GHIDRA_INSTALL_DIR")
    if env_dir and Path(env_dir).is_dir():
        return env_dir

    # 2. Homebrew
    try:
        result = subprocess.run(
            ["brew", "--prefix", "ghidra"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            prefix = Path(result.stdout.strip())
            libexec = prefix / "libexec"
            if libexec.is_dir():
                return str(libexec)
            if prefix.is_dir():
                return str(prefix)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # 3. Common paths
    for pattern in ["/opt/ghidra*", "/Applications/ghidra*", "/Applications/Ghidra*"]:
        import glob as _glob
        for candidate in sorted(_glob.glob(pattern), reverse=True):
            p = Path(candidate)
            if p.is_dir() and (p / "support" / "analyzeHeadless").exists():
                return str(p)

    return None
