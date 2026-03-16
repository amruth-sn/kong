"""Tests for Ghidra environment discovery (install + JDK detection)."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

from kong.ghidra.environment import (
    find_ghidra_install,
    find_java_home,
)


class TestFindGhidraInstall:
    def test_env_var(self, tmp_path):
        ghidra_dir = tmp_path / "ghidra"
        ghidra_dir.mkdir()
        with patch.dict(os.environ, {"GHIDRA_INSTALL_DIR": str(ghidra_dir)}):
            assert find_ghidra_install() == str(ghidra_dir)

    def test_env_var_nonexistent(self):
        with patch.dict(os.environ, {"GHIDRA_INSTALL_DIR": "/no/such/dir"}, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                assert find_ghidra_install() is None

    def test_homebrew(self, tmp_path):
        libexec = tmp_path / "libexec"
        libexec.mkdir()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = str(tmp_path) + "\n"
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", return_value=mock_result):
                assert find_ghidra_install() == str(libexec)

    def test_homebrew_not_installed(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                assert find_ghidra_install() is None

    def test_common_path(self, tmp_path):
        ghidra = tmp_path / "ghidra_11"
        (ghidra / "support").mkdir(parents=True)
        (ghidra / "support" / "analyzeHeadless").touch()
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                with patch("glob.glob", return_value=[str(ghidra)]):
                    assert find_ghidra_install() == str(ghidra)


class TestFindJavaHome:
    def test_existing_java_home_sufficient(self, tmp_path):
        java_home = tmp_path / "jdk21"
        (java_home / "bin").mkdir(parents=True)
        (java_home / "bin" / "java").touch()
        with patch.dict(os.environ, {"JAVA_HOME": str(java_home)}):
            with patch("kong.ghidra.environment._java_version", return_value=21):
                assert find_java_home() == str(java_home)

    def test_existing_java_home_too_old(self, tmp_path):
        """JAVA_HOME pointing to JDK 17 should be skipped, find 21 via brew."""
        java_home_17 = tmp_path / "jdk17"
        (java_home_17 / "bin").mkdir(parents=True)
        (java_home_17 / "bin" / "java").touch()

        jdk21_home = tmp_path / "libexec" / "openjdk.jdk" / "Contents" / "Home"
        jdk21_home.mkdir(parents=True)

        def mock_version(path):
            return 17 if "jdk17" in path else 21

        def mock_run(cmd, **kwargs):
            if "/usr/libexec/java_home" in cmd:
                r = MagicMock()
                r.returncode = 0
                r.stdout = str(java_home_17) + "\n"
                return r
            r = MagicMock()
            r.returncode = 0
            r.stdout = str(tmp_path) + "\n"
            return r

        with patch.dict(os.environ, {"JAVA_HOME": str(java_home_17)}):
            with patch(
                "kong.ghidra.environment._java_version", side_effect=mock_version
            ):
                with patch("subprocess.run", side_effect=mock_run):
                    assert find_java_home() == str(jdk21_home)

    def test_java_home_invalid_dir(self):
        with patch.dict(os.environ, {"JAVA_HOME": "/no/such/jdk"}, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                assert find_java_home() is None

    def test_macos_java_home_utility(self, tmp_path):
        java_home = tmp_path / "jdk21"
        java_home.mkdir()
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = str(java_home) + "\n"
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", return_value=mock_result):
                with patch("kong.ghidra.environment._java_version", return_value=21):
                    assert find_java_home() == str(java_home)

    def test_homebrew_openjdk(self, tmp_path):
        jdk_home = tmp_path / "libexec" / "openjdk.jdk" / "Contents" / "Home"
        jdk_home.mkdir(parents=True)

        def mock_run(cmd, **kwargs):
            if "/usr/libexec/java_home" in cmd:
                raise FileNotFoundError
            mock = MagicMock()
            mock.returncode = 0
            mock.stdout = str(tmp_path) + "\n"
            return mock

        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", side_effect=mock_run):
                assert find_java_home() == str(jdk_home)

    def test_nothing_found(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("subprocess.run", side_effect=FileNotFoundError):
                assert find_java_home() is None
