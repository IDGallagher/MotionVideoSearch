import os
import shutil
import typer
from huggingface_hub import Repository

app = typer.Typer()

def copy_contents(src: str, dst: str):
    """
    Recursively copy contents from src to dst,
    skipping any .git or repo_backup folder to avoid loops.
    """
    for item in os.listdir(src):
        # Skip .git to avoid nested Git repos
        if item == ".git":
            continue
        # Skip the repo_backup folder to avoid infinite recursion
        if item == "repo_backup":
            continue

        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

@app.command()
def backup(
    repo_id: str = typer.Argument(..., help="Hugging Face repository id (e.g. username/my-backup-repo)."),
    token: str = typer.Option(..., help="Your Hugging Face access token."),
    workspace_dir: str = typer.Option("/workspace", help="Path to the workspace directory to backup.")
):
    """
    Backup the local workspace to a Hugging Face repository.
    """
    temp_repo_dir = "repo_backup"

    # Remove temp directory if it already exists
    if os.path.exists(temp_repo_dir):
        shutil.rmtree(temp_repo_dir)

    # Clone the HF repo locally
    typer.echo(f"Cloning repo {repo_id} into {temp_repo_dir}...")
    repo = Repository(local_dir=temp_repo_dir, clone_from=repo_id, use_auth_token=token)

    # Copy the workspace into the cloned repo
    typer.echo(f"Copying contents from {workspace_dir} to {temp_repo_dir}...")
    copy_contents(workspace_dir, temp_repo_dir)

    # Commit and push changes
    typer.echo("Committing and pushing to Hugging Face...")
    repo.git_add(all=True)
    repo.git_commit("Backup workspace")
    repo.git_push()

    typer.echo("Backup complete!")

@app.command()
def restore(
    repo_id: str = typer.Argument(..., help="Hugging Face repository id (e.g. username/my-backup-repo)."),
    token: str = typer.Option(..., help="Your Hugging Face access token."),
    workspace_dir: str = typer.Option("/workspace", help="Path to the workspace directory to restore.")
):
    """
    Restore the workspace from the Hugging Face repository.
    """
    temp_repo_dir = "repo_backup"

    # Remove temp directory if it already exists
    if os.path.exists(temp_repo_dir):
        shutil.rmtree(temp_repo_dir)

    # Clone the HF repo locally
    typer.echo(f"Cloning repo {repo_id} into {temp_repo_dir}...")
    repo = Repository(local_dir=temp_repo_dir, clone_from=repo_id, use_auth_token=token)

    # Copy from the cloned repo back into the workspace
    typer.echo(f"Copying contents from {temp_repo_dir} to {workspace_dir}...")
    copy_contents(temp_repo_dir, workspace_dir)

    typer.echo("Restore complete!")

if __name__ == "__main__":
    app()
