"""
System Commands Module

Provides safe, asynchronous execution of system commands with proper
error handling, logging, and result capture.
"""

import asyncio
import logging
import shlex
import subprocess
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CommandResult:
    """Result of a command execution."""
    command: str
    returncode: int
    stdout: str
    stderr: str
    success: bool

    def __bool__(self) -> bool:
        return self.success


class SystemCommandExecutor:
    """
    Asynchronous system command executor with safety features.
    
    Features:
    - Async execution with timeout
    - Output capture (stdout/stderr)
    - Command validation
    - Structured logging
    - Error handling
    """
    
    # Commands that are explicitly allowed (empty = allow all with caution)
    ALLOWED_COMMANDS: set[str] = {
        "ls", "pwd", "echo", "cat", "mkdir", "touch", "cp", "mv", "rm",
        "python", "python3", "pip", "git", "which", "whoami", "date",
        "ps", "top", "df", "du", "free", "uname", "hostname", "uptime",
        "curl", "wget", "ping", "ssh", "scp", "rsync", "tar", "zip", "unzip",
        "systemctl", "service", "journalctl", "dmesg", "lsblk", "mount",
        "find", "grep", "awk", "sed", "sort", "uniq", "wc", "head", "tail",
        "chmod", "chown", "chgrp", "stat", "file", "which", "type",
    }
    
    # Commands that are explicitly blocked for safety
    BLOCKED_COMMANDS: set[str] = {
        "rm -rf /", "mkfs", "dd", "fdisk", "parted", "shutdown", "reboot",
        "halt", "poweroff", "init 0", "init 6", ":(){ :|:& };:", "fork",
    }
    
    def __init__(
        self,
        timeout: float = 30.0,
        allowed_commands: Optional[set[str]] = None,
        blocked_commands: Optional[set[str]] = None,
        shell: bool = False,
    ):
        """
        Initialize the command executor.
        
        Args:
            timeout: Default timeout in seconds for command execution
            allowed_commands: Set of allowed command names (None = use default)
            blocked_commands: Set of blocked command patterns (None = use default)
            shell: Whether to use shell=True (security risk, use with caution)
        """
        self.timeout = timeout
        self.shell = shell
        self.allowed_commands = allowed_commands or self.ALLOWED_COMMANDS.copy()
        self.blocked_commands = blocked_commands or self.BLOCKED_COMMANDS.copy()
    
    def _validate_command(self, command: str) -> tuple[bool, str]:
        """
        Validate a command for safety.
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check blocked patterns
        for blocked in self.blocked_commands:
            if blocked in command:
                return False, f"Command contains blocked pattern: {blocked}"
        
        # Extract base command
        try:
            parts = shlex.split(command)
            if not parts:
                return False, "Empty command"
            base_cmd = parts[0].split("/")[-1]  # Handle paths like /usr/bin/ls
        except ValueError as e:
            return False, f"Invalid command syntax: {e}"
        
        # Check if allowed (if allowlist is configured)
        if self.allowed_commands and base_cmd not in self.allowed_commands:
            return False, f"Command not in allowlist: {base_cmd}"
        
        return True, ""
    
    async def execute(
        self,
        command: str,
        timeout: Optional[float] = None,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
    ) -> CommandResult:
        """
        Execute a command asynchronously.
        
        Args:
            command: The command to execute
            timeout: Override default timeout
            cwd: Working directory
            env: Environment variables
            
        Returns:
            CommandResult with execution details
        """
        # Validate command
        valid, error = self._validate_command(command)
        if not valid:
            logger.warning("Command validation failed: %s - %s", command, error)
            return CommandResult(
                command=command,
                returncode=-1,
                stdout="",
                stderr=error,
                success=False,
            )
        
        exec_timeout = timeout or self.timeout
        logger.info("Executing command: %s (timeout=%.1fs)", command, exec_timeout)
        
        try:
            if self.shell:
                # Shell mode - pass as string
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                )
            else:
                # Non-shell mode - split command
                args = shlex.split(command)
                process = await asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=cwd,
                    env=env,
                )
            
            # Wait with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=exec_timeout,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.error("Command timed out after %.1fs: %s", exec_timeout, command)
                return CommandResult(
                    command=command,
                    returncode=-1,
                    stdout="",
                    stderr=f"Command timed out after {exec_timeout} seconds",
                    success=False,
                )
            
            stdout_str = stdout.decode("utf-8", errors="replace").strip()
            stderr_str = stderr.decode("utf-8", errors="replace").strip()
            success = process.returncode == 0
            
            logger.info(
                "Command completed: %s (returncode=%d, success=%s)",
                command, process.returncode, success
            )
            
            if stdout_str:
                logger.debug("stdout: %s", stdout_str[:500])
            if stderr_str:
                logger.debug("stderr: %s", stderr_str[:500])
            
            return CommandResult(
                command=command,
                returncode=process.returncode,
                stdout=stdout_str,
                stderr=stderr_str,
                success=success,
            )
            
        except Exception as e:
            logger.exception("Command execution failed: %s", command)
            return CommandResult(
                command=command,
                returncode=-1,
                stdout="",
                stderr=str(e),
                success=False,
            )
    
    async def execute_safe(
        self,
        command: str,
        timeout: Optional[float] = None,
    ) -> CommandResult:
        """
        Execute a command with additional safety checks.
        
        This is a convenience method that uses shell=False by default
        and validates against the allowlist.
        """
        return await self.execute(command, timeout=timeout, shell=False)
    
    def add_allowed_command(self, command: str) -> None:
        """Add a command to the allowlist."""
        self.allowed_commands.add(command)
        logger.debug("Added to allowlist: %s", command)
    
    def remove_allowed_command(self, command: str) -> None:
        """Remove a command from the allowlist."""
        self.allowed_commands.discard(command)
        logger.debug("Removed from allowlist: %s", command)
    
    def add_blocked_command(self, pattern: str) -> None:
        """Add a pattern to the blocklist."""
        self.blocked_commands.add(pattern)
        logger.debug("Added to blocklist: %s", pattern)


# Convenience function for simple use cases
async def run_command(
    command: str,
    timeout: float = 30.0,
    shell: bool = False,
) -> CommandResult:
    """
    Run a single command with default executor.
    
    Args:
        command: Command to execute
        timeout: Timeout in seconds
        shell: Use shell execution
        
    Returns:
        CommandResult
    """
    executor = SystemCommandExecutor(timeout=timeout, shell=shell)
    return await executor.execute(command, timeout=timeout)


# Synchronous wrapper for non-async contexts
def run_command_sync(
    command: str,
    timeout: float = 30.0,
    shell: bool = False,
) -> CommandResult:
    """
    Run a command synchronously (blocks until complete).
    
    Args:
        command: Command to execute
        timeout: Timeout in seconds
        shell: Use shell execution
        
    Returns:
        CommandResult
    """
    return asyncio.run(run_command(command, timeout=timeout, shell=shell))
