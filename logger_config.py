"""
Structured logging configuration with correlation IDs.
Logs are written in JSON format for easy parsing and monitoring.
"""
import structlog
import logging
import sys
from datetime import datetime
from typing import Optional

# Configure structlog to output JSON
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

# Configure standard logging to work with structlog
logging.basicConfig(
    format="%(message)s",
    stream=sys.stdout,
    level=logging.INFO,
)

def get_logger(correlation_id: Optional[str] = None):
    """
    Get a logger instance with optional correlation ID.
    
    Args:
        correlation_id: Unique identifier for tracking a request through the system
        
    Returns:
        A structured logger instance
    """
    logger = structlog.get_logger()
    if correlation_id:
        logger = logger.bind(correlation_id=correlation_id)
    return logger

def log_query(logger, query: str, correlation_id: str):
    """Log a user query."""
    logger.info(
        "user_query_received",
        query=query,
        correlation_id=correlation_id,
        timestamp=datetime.utcnow().isoformat()
    )

def log_tool_usage(logger, tool_name: str, query: str, correlation_id: str):
    """Log when a tool is being used."""
    logger.info(
        "tool_usage",
        tool_name=tool_name,
        query=query,
        correlation_id=correlation_id,
        timestamp=datetime.utcnow().isoformat()
    )

def log_tool_result(logger, tool_name: str, result: str, correlation_id: str, success: bool = True):
    """Log tool execution result."""
    log_level = "info" if success else "error"
    logger.log(
        log_level.upper(),
        "tool_result",
        tool_name=tool_name,
        result_length=len(result) if result else 0,
        success=success,
        correlation_id=correlation_id,
        timestamp=datetime.utcnow().isoformat()
    )

def log_agent_response(logger, response: str, correlation_id: str, latency_ms: Optional[float] = None):
    """Log the final agent response."""
    log_data = {
        "event": "agent_response",
        "response_length": len(response) if response else 0,
        "correlation_id": correlation_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    if latency_ms:
        log_data["latency_ms"] = latency_ms
    
    logger.info(**log_data)

def log_error(logger, error: Exception, correlation_id: str, context: Optional[dict] = None):
    """Log an error with context."""
    log_data = {
        "event": "error",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "correlation_id": correlation_id,
        "timestamp": datetime.utcnow().isoformat()
    }
    if context:
        log_data.update(context)
    
    logger.error(**log_data)

def log_agent_step(logger, step: dict, correlation_id: str):
    """Log an agent reasoning step."""
    logger.info(
        "agent_step",
        step_type=step.get("type", "unknown"),
        tool=step.get("tool", None),
        input=step.get("input", None),
        correlation_id=correlation_id,
        timestamp=datetime.utcnow().isoformat()
    )
