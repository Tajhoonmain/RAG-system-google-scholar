"""
Simple script to view and analyze structured logs.
"""
import json
import sys
from datetime import datetime
from collections import defaultdict

def view_logs(log_file="logs/agent_logs.json", limit=20):
    """View recent logs from the JSON log file."""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        print(f"\n{'='*80}")
        print(f"Viewing last {min(limit, len(lines))} log entries from {log_file}")
        print(f"{'='*80}\n")
        
        # Parse and display logs
        for i, line in enumerate(lines[-limit:], 1):
            try:
                log_entry = json.loads(line.strip())
                print(f"\n[{i}] {log_entry.get('event', 'unknown_event')}")
                print(f"    Timestamp: {log_entry.get('timestamp', 'N/A')}")
                print(f"    Correlation ID: {log_entry.get('correlation_id', 'N/A')}")
                print(f"    Level: {log_entry.get('level', 'N/A')}")
                
                # Show relevant fields
                if 'query' in log_entry:
                    print(f"    Query: {log_entry['query'][:100]}...")
                if 'tool_name' in log_entry:
                    print(f"    Tool: {log_entry['tool_name']}")
                if 'latency_ms' in log_entry:
                    print(f"    Latency: {log_entry['latency_ms']:.2f}ms")
                if 'error_message' in log_entry:
                    print(f"    Error: {log_entry['error_message']}")
                    
            except json.JSONDecodeError:
                print(f"[{i}] Invalid JSON line: {line[:100]}...")
        
        print(f"\n{'='*80}\n")
        
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
        print("Run the Streamlit app first to generate logs.")
    except Exception as e:
        print(f"Error reading logs: {e}")

def analyze_logs(log_file="logs/agent_logs.json"):
    """Analyze logs and show statistics."""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        print(f"\n{'='*80}")
        print(f"Log Analysis: {log_file}")
        print(f"{'='*80}\n")
        
        total_logs = len(lines)
        events = defaultdict(int)
        tools_used = defaultdict(int)
        errors = []
        correlation_ids = set()
        latencies = []
        
        for line in lines:
            try:
                log_entry = json.loads(line.strip())
                event = log_entry.get('event', 'unknown')
                events[event] += 1
                
                if 'correlation_id' in log_entry:
                    correlation_ids.add(log_entry['correlation_id'])
                
                if 'tool_name' in log_entry:
                    tools_used[log_entry['tool_name']] += 1
                
                if 'latency_ms' in log_entry:
                    latencies.append(log_entry['latency_ms'])
                
                if log_entry.get('level') == 'ERROR' or 'error' in event.lower():
                    errors.append(log_entry)
                    
            except json.JSONDecodeError:
                continue
        
        print(f"Total Log Entries: {total_logs}")
        print(f"Unique Queries: {len(correlation_ids)}")
        print(f"Errors: {len(errors)}")
        
        if latencies:
            print(f"\nLatency Statistics:")
            print(f"  Average: {sum(latencies)/len(latencies):.2f}ms")
            print(f"  Min: {min(latencies):.2f}ms")
            print(f"  Max: {max(latencies):.2f}ms")
        
        print(f"\nEvent Types:")
        for event, count in sorted(events.items(), key=lambda x: x[1], reverse=True):
            print(f"  {event}: {count}")
        
        if tools_used:
            print(f"\nTools Used:")
            for tool, count in sorted(tools_used.items(), key=lambda x: x[1], reverse=True):
                print(f"  {tool}: {count}")
        
        if errors:
            print(f"\nRecent Errors:")
            for error in errors[-5:]:
                print(f"  {error.get('timestamp', 'N/A')}: {error.get('error_message', 'N/A')}")
        
        print(f"\n{'='*80}\n")
        
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")
    except Exception as e:
        print(f"Error analyzing logs: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "analyze":
            analyze_logs()
        else:
            limit = int(sys.argv[1]) if sys.argv[1].isdigit() else 20
            view_logs(limit=limit)
    else:
        view_logs()
