import json
import os
import sys
import click


def get_ollama_directories():
    """Get the standard Ollama directory paths."""
    ollama_base = os.path.join(os.path.expanduser('~'), '.ollama', 'models')
    return {
        'manifest': os.path.join(ollama_base, 'manifests', 'registry.ollama.ai'),
        'blob': os.path.join(ollama_base, 'blobs'),
        'output': os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Output')
    }


def find_manifest_files(manifest_dir):
    """
    Recursively find all manifest files in the Ollama manifest directory.

    :param manifest_dir: Path to the manifest directory
    :return: List of manifest file paths
    """
    files = []
    if not os.path.exists(manifest_dir):
        return files

    for dirpath, _, filenames in os.walk(manifest_dir):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))
    return files


def get_model_size(layers, blob_directory):
    """
    Calculate total size of a model by summing all layer blob sizes.

    :param layers: List of layer dictionaries from manifest
    :param blob_directory: Path to blob storage directory
    :return: Total size in bytes
    """
    total_size = 0
    for layer_info in layers:
        digest = layer_info.get('digest', '')
        if ':' not in digest:
            continue
        sha = digest.split(':')[1]
        full_sha = f'sha256-{sha}'
        source_blob = os.path.join(blob_directory, full_sha)
        if os.path.exists(source_blob):
            total_size += os.path.getsize(source_blob)
    return total_size


def get_model_metadata(manifest_path, blob_directory):
    """
    Extract metadata from a manifest file.

    :param manifest_path: Path to the manifest JSON file
    :param blob_directory: Path to blob storage directory
    :return: Dictionary with model metadata or None if invalid
    """
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)

        config = manifest.get('config')
        if not config:
            return None

        digest = config.get('digest')
        if not digest:
            return None

        sha_value = digest.split(':')[-1]
        sha256_value = f'sha256-{sha_value}'
        sha_file = os.path.join(blob_directory, sha256_value)

        # Load model configuration
        try:
            with open(sha_file) as f:
                config_data = json.load(f)
            quantization = config_data.get('file_type', 'Unknown')
            model_type = config_data.get('model_type', 'unknown')
        except Exception:
            quantization = 'Unknown'
            model_type = 'unknown'

        layers = manifest.get('layers', [])
        size = get_model_size(layers, blob_directory)

        return {
            'name': os.path.basename(os.path.dirname(manifest_path)),
            'manifest_filename': os.path.basename(manifest_path),
            'quantization': quantization,
            'model_type': model_type,
            'size': size,
            'size_str': f"{size / (1024 * 1024):.2f} MB" if size > 0 else "Unknown",
            'layers': len(layers)
        }
    except Exception as e:
        return None


def find_manifest_by_name(manifest_locations, blob_directory, model_name):
    """
    Find a manifest file by model name with optional tag specification.

    Supports both simple names (e.g., 'llama3.2') and tagged names (e.g., 'gemma2:27b').

    :param manifest_locations: List of all manifest file paths
    :param blob_directory: Path to blob storage directory
    :param model_name: Name of the model to find, optionally with tag (e.g., 'model:tag')
    :return: Tuple of (manifest_path, matches_list) where matches_list contains all matching models
    """
    # Parse model name and optional tag
    if ':' in model_name:
        base_name, tag = model_name.split(':', 1)
    else:
        base_name = model_name
        tag = None

    matches = []

    for manifest_path in manifest_locations:
        # Extract model name from path structure
        # Path structure: .../manifests/registry.ollama.ai/library/modelname/tag
        path_parts = manifest_path.split(os.sep)

        # Find the model name (second to last directory)
        if len(path_parts) >= 2:
            model_dir = path_parts[-2]
            manifest_tag = path_parts[-1]

            # Check if model name matches
            if model_dir.lower() == base_name.lower():
                metadata = get_model_metadata(manifest_path, blob_directory)
                if metadata:
                    matches.append({
                        'path': manifest_path,
                        'model': model_dir,
                        'tag': manifest_tag,
                        'metadata': metadata
                    })

    # If tag specified, filter by tag
    if tag:
        tagged_matches = [m for m in matches if m['tag'].lower() == tag.lower()]
        if tagged_matches:
            return tagged_matches[0]['path'], matches
        return None, matches

    # If only one match, return it
    if len(matches) == 1:
        return matches[0]['path'], matches

    # Multiple matches without tag specification
    if len(matches) > 1:
        return None, matches

    # No matches
    return None, []


def recombine_model(manifest_path, blob_directory, output_directory, verbose=True):
    """
    Recombine model parts from Ollama's split format into a single GGUF equivalent file.

    :param manifest_path: Path to the JSON manifest file describing the model.
    :param blob_directory: Path where all blob parts are stored.
    :param output_directory: Target directory where combined gguf will be saved.
    :param verbose: Whether to print detailed progress information.
    :return: Path to the output file or None if failed
    """
    # Load and parse JSON data from manifest
    with open(manifest_path) as f_obj:
        obj = json.load(f_obj)

    config = obj.get('config')
    if not config:
        raise ValueError('Config section missing from JSON.')

    digest = config.get('digest')
    if not digest:
        raise ValueError('Digest missing from config section.')

    sha_value = digest.split(':')[-1]
    sha256_value = f'sha256-{sha_value}'
    sha_file = os.path.join(blob_directory, sha256_value)

    # Load configuration data about this specific SHA value
    with open(sha_file) as f_model_config_obj:
        config_data = json.load(f_model_config_obj)

    try:
        modelQuant = config_data['file_type']
        assert len(modelQuant) > 0, "Model quantization type cannot be empty."
        assert isinstance(modelQuant, str), "Model quantization type must be string."
    except Exception as e:
        raise ValueError("Invalid or missing `file_type` parameter.") from e

    try:
        trained_on = str(config_data['model_type'])
    except KeyError:
        trained_on = 'unknown'

    layers = obj.get('layers')
    if not layers:
        raise ValueError("Layers section is required but missing.")

    modelName = os.path.basename(os.path.dirname(manifest_path))
    target_subdir = os.path.join(output_directory, modelName)
    combined_filename = f"{modelName}-{trained_on}-{modelQuant}.gguf"
    final_output_filepath = os.path.join(target_subdir, combined_filename)

    if not os.path.exists(target_subdir):
        os.makedirs(target_subdir)

    # Stream data directly from source blobs to output file
    # Use a reasonable buffer size (e.g., 64MB) to balance memory and I/O efficiency
    BUFFER_SIZE = 64 * 1024 * 1024  # 64 MB buffer

    try:
        if verbose:
            click.echo(f"\nStreaming layers to: {combined_filename}")

        with open(final_output_filepath, 'wb') as final_fobj:
            for layer_index, layer_info in enumerate(layers):
                mediaType = layer_info.get('mediaType', 'unknown')
                digest = layer_info['digest']
                sha = digest.split(':')[1]
                full_sha = f'sha256-{sha}'
                source_blob = os.path.join(blob_directory, full_sha)

                if verbose:
                    click.echo(f"  Layer {layer_index + 1}/{len(layers)}: [{mediaType}]")

                # Stream data in chunks instead of loading entire file
                with open(source_blob, 'rb') as layer_fobj:
                    while True:
                        chunk = layer_fobj.read(BUFFER_SIZE)
                        if not chunk:
                            break
                        final_fobj.write(chunk)

        if verbose:
            click.secho(f"✓ Successfully exported to: {final_output_filepath}\n", fg='green')

        return final_output_filepath

    except Exception as excp:
        # Clean up partial file on error
        if os.path.exists(final_output_filepath):
            os.remove(final_output_filepath)
        if verbose:
            click.secho(f"✗ Failed: {excp}\n", fg='red')
        raise


@click.group()
@click.version_option(version='1.0.0', prog_name='ollama-to-gguf')
def cli():
    """
    Ollama to GGUF Converter

    Convert Ollama models to standalone GGUF files.
    """
    pass


@cli.command()
@click.option('--manifest-dir', type=click.Path(exists=True),
              help='Custom path to Ollama manifest directory')
@click.option('--blob-dir', type=click.Path(exists=True),
              help='Custom path to Ollama blob directory')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed information')
def list(manifest_dir, blob_dir, verbose):
    """List all available Ollama models."""

    directories = get_ollama_directories()
    manifest_dir = manifest_dir or directories['manifest']
    blob_dir = blob_dir or directories['blob']

    if not os.path.exists(manifest_dir):
        click.secho(f"✗ Manifest directory not found: {manifest_dir}", fg='red')
        click.echo("  Make sure Ollama is installed and you have downloaded models.")
        return

    manifest_locations = find_manifest_files(manifest_dir)

    if not manifest_locations:
        click.secho("✗ No models found.", fg='yellow')
        click.echo("  Run 'ollama pull <model>' to download a model first.")
        return

    if verbose:
        # Verbose output - detailed format
        click.echo("\n" + "=" * 80)
        click.secho(f" Found {len(manifest_locations)} Ollama Model(s)", fg='cyan', bold=True)
        click.echo("=" * 80 + "\n")

        for manifest_path in manifest_locations:
            metadata = get_model_metadata(manifest_path, blob_dir)

            if metadata:
                # Extract model:tag format
                path_parts = manifest_path.split(os.sep)
                if len(path_parts) >= 2:
                    full_name = f"{path_parts[-2]}:{path_parts[-1]}"
                else:
                    full_name = metadata['name']

                click.secho(f"Model: {full_name}", fg='green', bold=True)
                click.echo(f"  Manifest: {metadata['manifest_filename']}")
                click.echo(f"  Model Type: {metadata['model_type']}")
                click.echo(f"  Quantization: {metadata['quantization']}")
                click.echo(f"  Size: {metadata['size_str']}")
                click.echo(f"  Layers: {metadata['layers']}")
                click.echo()
            else:
                modelName = os.path.basename(os.path.dirname(manifest_path))
                click.secho(f"Model: {modelName}", fg='yellow')
                click.echo(f"  Status: Unable to read metadata")
                click.echo()
    else:
        # Concise output - table format
        click.echo()
        click.secho(f"Found {len(manifest_locations)} Ollama model(s):", fg='cyan', bold=True)
        click.echo()

        # Prepare table data
        rows = []
        for manifest_path in manifest_locations:
            metadata = get_model_metadata(manifest_path, blob_dir)

            # Extract model:tag format
            path_parts = manifest_path.split(os.sep)
            if len(path_parts) >= 2:
                full_name = f"{path_parts[-2]}:{path_parts[-1]}"
            else:
                full_name = metadata['name'] if metadata else os.path.basename(os.path.dirname(manifest_path))

            if metadata:
                rows.append({
                    'name': full_name,
                    'type': metadata['model_type'],
                    'quant': metadata['quantization'],
                    'size': metadata['size_str'],
                    'layers': str(metadata['layers'])
                })
            else:
                rows.append({
                    'name': full_name,
                    'type': 'N/A',
                    'quant': 'N/A',
                    'size': 'N/A',
                    'layers': 'N/A'
                })

        # Calculate column widths
        col_widths = {
            'name': max(len(row['name']) for row in rows) if rows else 10,
            'type': max(len(row['type']) for row in rows) if rows else 10,
            'quant': max(len(row['quant']) for row in rows) if rows else 12,
            'size': max(len(row['size']) for row in rows) if rows else 10,
            'layers': max(len(row['layers']) for row in rows) if rows else 6
        }

        # Ensure minimum widths for headers
        col_widths['name'] = max(col_widths['name'], len('Model Name'))
        col_widths['type'] = max(col_widths['type'], len('Type'))
        col_widths['quant'] = max(col_widths['quant'], len('Quantization'))
        col_widths['size'] = max(col_widths['size'], len('Size'))
        col_widths['layers'] = max(col_widths['layers'], len('Layers'))

        # Print header
        header = (
            f"{'Model Name':<{col_widths['name']}}  "
            f"{'Type':<{col_widths['type']}}  "
            f"{'Quantization':<{col_widths['quant']}}  "
            f"{'Size':>{col_widths['size']}}  "
            f"{'Layers':>{col_widths['layers']}}"
        )
        click.secho(header, fg='cyan', bold=True)

        # Print separator
        separator = (
            f"{'-' * col_widths['name']}  "
            f"{'-' * col_widths['type']}  "
            f"{'-' * col_widths['quant']}  "
            f"{'-' * col_widths['size']}  "
            f"{'-' * col_widths['layers']}"
        )
        click.echo(separator)

        # Print rows
        for row in rows:
            row_str = (
                f"{row['name']:<{col_widths['name']}}  "
                f"{row['type']:<{col_widths['type']}}  "
                f"{row['quant']:<{col_widths['quant']}}  "
                f"{row['size']:>{col_widths['size']}}  "
                f"{row['layers']:>{col_widths['layers']}}"
            )

            # Color code based on availability
            if row['type'] == 'N/A':
                click.secho(row_str, fg='yellow')
            else:
                click.echo(row_str)

        click.echo()


@cli.command()
@click.argument('model_name')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Custom output directory (default: ./Output)')
@click.option('--manifest-dir', type=click.Path(exists=True),
              help='Custom path to Ollama manifest directory')
@click.option('--blob-dir', type=click.Path(exists=True),
              help='Custom path to Ollama blob directory')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output')
def export(model_name, output_dir, manifest_dir, blob_dir, quiet):
    """
    Export a specific Ollama model to GGUF format.

    MODEL_NAME: The name of the model to export (e.g., llama3.2, gemma2:27b)

    If multiple versions of a model exist, specify the tag using 'model:tag' format.
    """

    directories = get_ollama_directories()
    manifest_dir = manifest_dir or directories['manifest']
    blob_dir = blob_dir or directories['blob']
    output_dir = output_dir or directories['output']

    if not os.path.exists(manifest_dir):
        click.secho(f"✗ Manifest directory not found: {manifest_dir}", fg='red')
        return

    if not os.path.exists(blob_dir):
        click.secho(f"✗ Blob directory not found: {blob_dir}", fg='red')
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        if not quiet:
            click.secho(f"✓ Created output directory: {output_dir}", fg='green')

    # Find all manifests
    manifest_locations = find_manifest_files(manifest_dir)

    if not manifest_locations:
        click.secho("✗ No models found.", fg='red')
        return

    # Find the specific model
    manifest_path, matches = find_manifest_by_name(manifest_locations, blob_dir, model_name)

    if not manifest_path:
        if len(matches) > 1:
            # Multiple matches found - ambiguous
            click.secho(f"✗ Multiple versions of '{model_name.split(':')[0]}' found. Please specify a tag:",
                        fg='yellow')
            click.echo()
            for match in matches:
                metadata = match['metadata']
                full_name = f"{match['model']}:{match['tag']}"
                click.echo(f"  {full_name:<30} ({metadata['quantization']}, {metadata['size_str']})")
            click.echo()
            click.secho(f"Example: python main.py export {matches[0]['model']}:{matches[0]['tag']}", fg='cyan')
            return
        elif len(matches) == 0:
            # No matches found
            click.secho(f"✗ Model '{model_name}' not found.", fg='red')
            click.echo("\nAvailable models:")

            # Group models by name
            model_groups = {}
            for mp in manifest_locations:
                path_parts = mp.split(os.sep)
                if len(path_parts) >= 2:
                    model_name_part = path_parts[-2]
                    tag_part = path_parts[-1]
                    if model_name_part not in model_groups:
                        model_groups[model_name_part] = []
                    metadata = get_model_metadata(mp, blob_dir)
                    if metadata:
                        model_groups[model_name_part].append((tag_part, metadata))

            for model, variants in sorted(model_groups.items()):
                if len(variants) == 1:
                    click.echo(f"  - {model}")
                else:
                    for tag, metadata in variants:
                        click.echo(f"  - {model}:{tag} ({metadata['size_str']})")
            return

    # Get metadata for confirmation
    metadata = get_model_metadata(manifest_path, blob_dir)

    if not quiet:
        click.echo("\n" + "=" * 80)
        # Show full model:tag name if there were multiple variants
        display_name = metadata['name']
        if len(matches) > 1:
            path_parts = manifest_path.split(os.sep)
            if len(path_parts) >= 2:
                display_name = f"{path_parts[-2]}:{path_parts[-1]}"

        click.secho(f" Exporting: {display_name}", fg='cyan', bold=True)
        click.echo("=" * 80)
        click.echo(f"Quantization: {metadata['quantization']}")
        click.echo(f"Size: {metadata['size_str']}")
        click.echo(f"Layers: {metadata['layers']}")

    # Perform the export
    try:
        output_path = recombine_model(manifest_path, blob_dir, output_dir, verbose=not quiet)
        if not quiet:
            click.secho(f"\n✓ Export complete: {output_path}", fg='green', bold=True)
    except Exception as e:
        click.secho(f"\n✗ Export failed: {e}", fg='red', bold=True)
        sys.exit(1)


@cli.command()
@click.option('--output-dir', '-o', type=click.Path(),
              help='Custom output directory (default: ./Output)')
@click.option('--manifest-dir', type=click.Path(exists=True),
              help='Custom path to Ollama manifest directory')
@click.option('--blob-dir', type=click.Path(exists=True),
              help='Custom path to Ollama blob directory')
def export_all(output_dir, manifest_dir, blob_dir):
    """Export all available Ollama models to GGUF format."""

    directories = get_ollama_directories()
    manifest_dir = manifest_dir or directories['manifest']
    blob_dir = blob_dir or directories['blob']
    output_dir = output_dir or directories['output']

    if not os.path.exists(manifest_dir):
        click.secho(f"✗ Manifest directory not found: {manifest_dir}", fg='red')
        return

    if not os.path.exists(blob_dir):
        click.secho(f"✗ Blob directory not found: {blob_dir}", fg='red')
        return

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        click.secho(f"✓ Created output directory: {output_dir}", fg='green')

    # Find all manifests
    manifest_locations = find_manifest_files(manifest_dir)

    if not manifest_locations:
        click.secho("✗ No models found.", fg='red')
        return

    click.echo("\n" + "=" * 80)
    click.secho(f" Exporting {len(manifest_locations)} Model(s)", fg='cyan', bold=True)
    click.echo("=" * 80 + "\n")

    successful = 0
    failed = 0

    for idx, manifest_path in enumerate(manifest_locations, 1):
        metadata = get_model_metadata(manifest_path, blob_dir)
        if not metadata:
            click.secho(f"[{idx}/{len(manifest_locations)}] Skipping invalid manifest", fg='yellow')
            failed += 1
            continue

        click.secho(f"\n[{idx}/{len(manifest_locations)}] Exporting: {metadata['name']}", fg='cyan', bold=True)

        try:
            output_path = recombine_model(manifest_path, blob_dir, output_dir, verbose=True)
            successful += 1
        except Exception as e:
            click.secho(f"✗ Failed to export {metadata['name']}: {e}", fg='red')
            failed += 1

    click.echo("\n" + "=" * 80)
    click.secho(f" Export Summary", fg='cyan', bold=True)
    click.echo("=" * 80)
    click.secho(f"✓ Successful: {successful}", fg='green')
    if failed > 0:
        click.secho(f"✗ Failed: {failed}", fg='red')
    click.echo()


if __name__ == "__main__":
    cli()