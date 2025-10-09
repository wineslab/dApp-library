# E3 Service Model Generator Tools

This directory contains utility scripts for developing E3 Service Models.

## generate_sm.sh

Automated skeleton code generator for E3 Service Models. This script creates boilerplate code for both agent-side (C implementation in OAI) and dApp-side (Python implementation) service models.

### Quick Start

**Interactive mode (recommended):**
```bash
./generate_sm.sh --interactive
```

**Command-line mode:**
```bash
./generate_sm.sh -n my_sm -a my_sm.asn -r 2 --agent-root /path/to/spear-openairinterface5g
```

### Features

- **Dual-target generation**: Creates both agent and dApp code, or either separately
- **ASN.1 integration**: Automatically sets up ASN.1 compilation and integration
- **CMake configuration**: Generates complete build configuration files
- **Format flexibility**: Supports both ASN.1 (PER) and JSON encoding formats
- **Interactive mode**: User-friendly prompts for all options
- **Validation**: Checks paths, naming conventions, and dependencies

### Prerequisites

- Bash shell (Linux/macOS)
- For agent code generation:
  - Access to spear-openairinterface5g repository
  - ASN.1 compiler (asn1c)
- For dApp code generation:
  - Python 3.10+
  - asn1tools package

### Generated Code Structure

The generator creates a complete skeleton with:

**Agent-side (C):**
- Service Model main implementation (init, destroy, thread)
- Encoding/decoding functions (ASN.1 or JSON)
- CMake build integration
- ASN.1 message compilation setup
- SM registry integration

**dApp-side (Python):**
- DApp class implementation
- E3 interface integration
- Encoding/decoding methods
- Callback handling
- ASN.1 schema loading

### Customization After Generation

After running the generator, you'll need to:

1. **Review generated TODOs**: Each file contains TODO comments marking areas requiring implementation
2. **Implement SM logic**: Add your specific service model data collection and processing
3. **Update ASN.1 files**: Add generated ASN.1 source files to cmake files
4. **Test the implementation**: Build and test both agent and dApp sides

### Environment Variables

- `AGENT_ROOT`: Path to spear-openairinterface5g repository (can also use `--agent-root`)

### Examples

**Generate positioning SM with both sides:**
```bash
./generate_sm.sh -n positioning -a positioning.asn -r 2 --agent-root ../spear-openairinterface5g
```

**Generate KPM SM with multiple RAN functions (dApp only):**
```bash
./generate_sm.sh -n kpm -a kpm.asn -r 3,4,5 -t dapp
```

**Generate with JSON encoding:**
```bash
./generate_sm.sh -n my_sm -a my_sm.asn -f json
```

### Troubleshooting

**Error: "AGENT_ROOT not found or invalid"**
- Ensure the OAI repository path is correct
- Use `--agent-root` flag or set `AGENT_ROOT` environment variable
- Verify the path contains `openair2/E3AP` directory

**Error: "ASN.1 file not found"**
- Check the ASN.1 file path is correct
- Use absolute path or path relative to current directory

**Error: "Invalid SM name"**
- SM names must start with a letter
- Only alphanumeric characters and underscores allowed
- Use lowercase for consistency (e.g., `positioning_sm`)

### Reference Implementation

See the Spectrum Service Model (`src/spectrum/`) for a complete reference implementation that demonstrates:
- ASN.1 encoding/decoding
- Real-time data processing
- Control message handling
- Visualization integration

### Support

For questions or issues:
- Check the main [README](../README.md)
- Review the [CONTRIBUTING](../CONTRIBUTING.md) guidelines
- See the OpenRAN Gym tutorials: https://openrangym.com/tutorials/dapps-oai
