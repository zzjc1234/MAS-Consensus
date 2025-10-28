"""
Logging configuration for multi-agent consensus system.

Provides thread-safe logging with separate log files for:
- Each agent
- Each auditor  
- System-wide events
- Experiment-specific logs
"""

import logging
import logging.handlers
import os
from pathlib import Path
import threading


# Global lock for logger creation
_logger_lock = threading.Lock()
_loggers = {}


def setup_experiment_logging(model, dataset, sample_id, graph_type, num_agents, attacker_num, suffix=""):
    """
    Set up logging for an experiment run.
    
    Creates log directory structure:
    logs/{model}/{dataset}/{sample_id}/{graph_type}_{num_agents}_{attacker_num}{suffix}/
        ├── system.log           (system-wide events)
        ├── agents/
        │   ├── agent_0.log
        │   ├── agent_1.log
        │   └── ...
        └── auditors/
            ├── auditor_0.log
            ├── auditor_1.log
            └── ...
    """
    # Create log directory structure
    exp_name = f"{graph_type}_{num_agents}_{attacker_num}{suffix}"
    log_dir = Path(f"logs/{model}/{dataset}/{sample_id}/{exp_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (log_dir / "agents").mkdir(exist_ok=True)
    (log_dir / "auditors").mkdir(exist_ok=True)
    
    return log_dir


def get_agent_logger(agent_idx, log_dir, is_auditor=False, task_id=None):
    """
    Get a thread-safe logger for a specific agent or auditor in a specific task.
    
    Args:
        agent_idx: Index of the agent/auditor
        log_dir: Task-specific log directory
        is_auditor: Whether this is an auditor (True) or agent (False)
        task_id: Task ID for unique logger naming
    
    Returns:
        Logger instance configured for this agent/auditor in this task
    """
    with _logger_lock:
        # Create unique logger name including task_id to avoid mixing across tasks
        prefix = "auditor" if is_auditor else "agent"
        task_suffix = f"_task_{task_id}" if task_id else ""
        logger_name = f"{prefix}_{agent_idx}{task_suffix}"
        
        # Return existing logger if already created
        if logger_name in _loggers:
            return _loggers[logger_name]
        
        # Create new logger
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # Don't propagate to root logger
        
        # Remove any existing handlers
        logger.handlers.clear()
        
        # File handler - separate file for each agent/auditor
        subdir = "auditors" if is_auditor else "agents"
        log_file = Path(log_dir) / subdir / f"{prefix}_{agent_idx}.log"
        
        # Ensure subdirectory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        
        # Detailed format for file logs
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler - disabled for agents/auditors (too verbose with many tasks)
        # Only file logging for individual agents/auditors
        # Console will only show system-level messages
        
        # Cache logger
        _loggers[logger_name] = logger
        
        return logger


def get_system_logger(log_dir, task_id=None):
    """
    Get a logger for system-wide events for a specific task.
    
    Args:
        log_dir: Task-specific log directory
        task_id: Task ID for unique logger naming
    
    Returns:
        Logger instance for system events
    """
    with _logger_lock:
        task_suffix = f"_task_{task_id}" if task_id else ""
        logger_name = f"system{task_suffix}"
        
        if logger_name in _loggers:
            return _loggers[logger_name]
        
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()
        
        # File handler for system logs
        log_file = Path(log_dir) / "system.log"
        
        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler for system logs - disabled to reduce noise
        # With many parallel tasks, console output becomes overwhelming
        # All logs go to files; use tqdm progress bar for high-level progress
        
        _loggers[logger_name] = logger
        
        return logger


def get_console_logger():
    """
    Get a high-level console logger for experiment progress.
    This logger only outputs to console (not files) for overall progress tracking.
    
    Returns:
        Logger instance for console-only output
    """
    with _logger_lock:
        logger_name = "progress"
        
        if logger_name in _loggers:
            return _loggers[logger_name]
        
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False
        logger.handlers.clear()
        
        # Console handler only - for high-level progress
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(message)s')  # Simple format for progress
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        _loggers[logger_name] = logger
        
        return logger


def clear_loggers():
    """Clear all cached loggers (useful for testing or new experiments)"""
    with _logger_lock:
        for logger in _loggers.values():
            handlers = logger.handlers[:]
            for handler in handlers:
                handler.close()
                logger.removeHandler(handler)
        _loggers.clear()


def get_experiment_logger(name, log_dir):
    """
    Get a general experiment logger.
    
    Args:
        name: Logger name
        log_dir: Base log directory
    
    Returns:
        Logger instance
    """
    with _logger_lock:
        if name in _loggers:
            return _loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        logger.handlers.clear()
        
        # File handler
        log_file = Path(log_dir) / f"{name}.log"
        
        # Ensure directory exists
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=3,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        _loggers[name] = logger
        
        return logger

