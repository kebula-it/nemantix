# CLI Reference

The Nemantix CLI (`nemantix`) exposes five subcommands that map to the
[Script Lifecycle](./04%20-%20Script%20Lifecycle.md):

| Subcommand | Purpose                                         |
|------------|-------------------------------------------------|
| `run`      | Execute NXS / NXC / NXV scripts via an Agent    |
| `code`     | Compile NXS scripts to NXC without executing    |
| `sign`     | Sign NXC files to produce verifiable NXV files  |
| `verify`   | Verify the cryptographic signature of NXV files |
| `keygen`   | Generate an ECDSA key pair for signing          |

---

## Shorthand invocation

When the first argument is not a subcommand, the CLI implicitly prepends `run`:

```bash
# Equivalent to: nemantix run agent.nxs --user-request "Summarise the report"
nemantix agent.nxs --user-request "Summarise the report"
```

---

## `nemantix run`

Executes one or more NXS / NXC / NXV scripts through a Semantic Agent.

```bash
nemantix run [paths ...] [options]
```

### General options

| Flag                   | Default            | Description                                                 |
|------------------------|--------------------|-------------------------------------------------------------|
| `paths`                | —                  | Scripts to execute (positional, repeatable)                 |
| `-u`, `--user-request` | stdin              | User request string                                         |
| `--vendor`             | `openai`           | LLM vendor (env: `NEMANTIX_VENDOR`)                         |
| `--model`              | `gpt-5-mini`       | LLM model name (env: `NEMANTIX_MODEL`)                      |
| `--credentials`        | `credentials.json` | Path to credentials JSON file                               |
| `--export-location`    | `coding_output`    | Directory for compiled script output                        |
| `--no-build`           | `false`            | Skip build-on-start                                         |
| `--use-embedder`       | `false`            | Enable sentence-transformer embedder                        |
| `--use-knowledge-base` | `false`            | Enable the Knowledge Base                                   |
| `--log-level`          | —                  | Agent log level: `DEBUG`, `INFO`, `WARNING`, `ERROR`        |
| `--verify`             | —                  | Path to public key PEM for NXV signature verification       |
| `--debug`              | `false`            | Enable the interactive CLI debugger (ndb)                   |
| `--profile`            | `false`            | Enable the Profiler and print stats on exit                 |
| `--toolset`            | —                  | Load extra toolsets (repeatable, see [Toolsets](#toolsets)) |

### Knowledge Base options

These flags are active only when `--use-knowledge-base` is set.
Sensitive credentials (`db_username`, `db_password`) are read exclusively from
environment variables to avoid exposing secrets in shell history.

| Flag                     | Default       | Description                                       |
|--------------------------|---------------|---------------------------------------------------|
| `--kb-view-id VIEW_ID`   | —             | Knowledge Base view ID. Repeatable — **required** |
| `--kb-db-engine`         | `postgresql`  | Database engine                                   |
| `--kb-db-host`           | `localhost`   | Database host                                     |
| `--kb-db-port`           | `5432`        | Database port                                     |
| `--kb-db-database`       | `nemantix_db` | Database name                                     |
| `--kb-base-storage-path` | `kb_storage`  | Base storage path for the Knowledge Base          |
| `--kb-vector-subdir`     | `vector_db`   | Vector store subdirectory                         |
| `--kb-vector-store-type` | `qdrant`      | Vector store type (`qdrant`, `faiss`, `milvus`)   |

**Environment variables for KB credentials (obbligatorie con `--use-knowledge-base`):**

| Variable               | Description                      |
|------------------------|----------------------------------|
| `NEMANTIX_KB_USERNAME` | Database username — **required** |
| `NEMANTIX_KB_PASSWORD` | Database password — **required** |

> Se `--use-knowledge-base` è attivo ma `--kb-view-ids` è assente o una delle due
> variabili non è impostata, il CLI termina con exit code `1` e stampa un messaggio
> descrittivo su stderr.

**Example:**

```bash
export NEMANTIX_KB_USERNAME=admin
export NEMANTIX_KB_PASSWORD=secret

nemantix run agent.nxs \
  --use-knowledge-base \
  --kb-view-id prod_kb \
  --kb-view-id staging_kb \
  --kb-db-host db.internal \
  --kb-db-database mydb \
  --user-request "Find all reports from last quarter"
```

### Toolsets

The `--toolset` flag registers extra toolsets before execution.
It can be repeated and accepts two forms:

| Form                    | Effect                                          |
|-------------------------|-------------------------------------------------|
| `module.path`           | Adds the package to the wildcard lookup list    |
| `ClassName=module.path` | Maps the class name directly to its import path |

```bash
# Add a whole package to the lookup list
nemantix run agent.nxs --toolset myapp.toolsets

# Map a specific class
nemantix run agent.nxs --toolset PizzaToolset=myapp.pizza

# Combine both forms
nemantix run agent.nxs \
  --toolset myapp.toolsets \
  --toolset PizzaToolset=myapp.pizza
```

For details on writing custom toolsets see
[Toolsets](./05%20-%20Toolsets.md).

---

## `nemantix code`

Compiles one or more NXS scripts to NXC without executing them. Useful for
iterating on the coding step independently.

```bash
nemantix code [paths ...] [options]
```

| Flag            | Default            | Description                            |
|-----------------|--------------------|----------------------------------------|
| `paths`         | —                  | NXS scripts to compile (positional)    |
| `--output`      | same dir as source | Output directory for NXC files         |
| `--vendor`      | `openai`           | LLM vendor (env: `NEMANTIX_VENDOR`)    |
| `--model`       | `gpt-5-mini`       | LLM model name (env: `NEMANTIX_MODEL`) |
| `--credentials` | `credentials.json` | Path to credentials JSON file          |

**Example:**

```bash
nemantix code scripts/agent.nxs --output build/
```

---

## `nemantix sign`

Signs one or more NXC files with an ECDSA private key to produce `.nxv` files.

> **Note:** `.nxv` files must never be edited manually — the embedded signature
> would be invalidated. Always re-sign after any change.

```bash
nemantix sign [paths ...] --key PRIVATE_KEY_PEM [options]
```

| Flag       | Default            | Description                            |
|------------|--------------------|----------------------------------------|
| `paths`    | —                  | NXC files to sign (positional)         |
| `--key`    | **required**       | Path to the ECDSA private key PEM file |
| `--output` | same dir as source | Output directory for signed NXV files  |

**Example:**

```bash
nemantix sign build/agent.nxc --key keys/nmx_ecdsa_private.pem --output dist/
```

---

## `nemantix verify`

Verifies the cryptographic signature of one or more `.nxv` files against a
public key.

```bash
nemantix verify [paths ...] --key PUBLIC_KEY_PEM
```

| Flag    | Default      | Description                           |
|---------|--------------|---------------------------------------|
| `paths` | —            | NXV files to verify (positional)      |
| `--key` | **required** | Path to the ECDSA public key PEM file |

**Example:**

```bash
nemantix verify dist/agent.nxv --key keys/nmx_ecdsa_public.pem
```

Exit code is `0` if all files pass, `1` if any verification fails.

---

## `nemantix keygen`

Generates an ECDSA key pair (`SECP256R1`) for use with `sign` and `verify`.
The output directory must already exist.

```bash
nemantix keygen [--output DIR]
```

| Flag       | Default | Description                                   |
|------------|---------|-----------------------------------------------|
| `--output` | `.`     | Directory where the key files will be written |

The command produces two files in the output directory:

| File                    | Use with          |
|-------------------------|-------------------|
| `nmx_ecdsa_private.pem` | `nemantix sign`   |
| `nmx_ecdsa_public.pem`  | `nemantix verify` |

**Example:**

```bash
mkdir keys/
nemantix keygen --output keys/
# Keys generated in 'keys/':
#   Private key: keys/nmx_ecdsa_private.pem
#   Public key:  keys/nmx_ecdsa_public.pem
```

---

## Plugin system

Subcommands are discovered via Python entry points under the
`nemantix.cli` group. Third-party packages can add new subcommands
without modifying the core package:

```toml
# pyproject.toml of a plugin package
[project.entry-points."nemantix.cli"]
my-command = "mypackage.cli.my_command:register"
```

The `register` function must accept an `argparse._SubParsersAction` and
return the registered `ArgumentParser`:

```python
# mypackage/cli/my_command.py
import argparse


def register(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    p = subparsers.add_parser("my-command", help="...")
    p.add_argument("--foo")
    p.set_defaults(handler=handle)
    return p


def handle(args: argparse.Namespace) -> int:
    ...
    return 0
```

When two plugins register the same subcommand name, the last entry point
(in installation order) wins.
