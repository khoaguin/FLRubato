import subprocess
import json

def run_go_function(input_data):
    """
    Run a Go program and pass data to it via command line arguments.
    The Go program should be compiled first.
    """
    # Convert Python data to JSON string
    input_json = json.dumps(input_data)
    
    try:
        # Run the compiled Go program with input as bytes
        result = subprocess.run(
            ['./calculator'],  # Path to your compiled Go executable
            input=input_json.encode('utf-8'),  # Properly encode the string to bytes
            capture_output=True,
            check=True  # Raise CalledProcessError if the program returns non-zero
        )
        
        # Parse the JSON output from Go
        return json.loads(result.stdout.decode('utf-8'))
    except subprocess.CalledProcessError as e:
        print(f"Go program failed with error code {e.returncode}")
        print(f"Error output: {e.stderr.decode('utf-8')}")
        return None
    except Exception as e:
        print(f"Error running Go code: {str(e)}")
        return None

# Example usage
if __name__ == "__main__":
    data = {
        "operation": "multiply",
        "numbers": [5, 3]
    }

    result = run_go_function(data)
    print(f"Result from Go: {result}")