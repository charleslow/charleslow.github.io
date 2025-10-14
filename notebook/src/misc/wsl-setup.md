# Setting up WSL

We may need to uninstall and re-install WSL (Windows Subsystem for Linux) from time to time. Here is the step-by-step.

1. Tear down and delete all files. `wsl --unregister Ubuntu-22.04`
2. Re-install wsl. `wsl --install Ubuntu-22.04`
3. Set up keychain to auto-find ssh-agent (from Windows) and add keys
    - Copy `.ssh` folder from Windows to wsl `~/.ssh`
    - Add to `~/.bashrc` the following: `eval $(keychain --eval --agents ssh id_rsa)`
4. Install Docker
    - Add users to docker to allow vscode access 
    - https://docs.docker.com/engine/install/linux-postinstall/
5. Essentials (get C linker)
	- `sudo apt install build-essential`
6. Install rust
	- https://www.rust-lang.org/tools/install
7. Install mdbook
	- `cargo install mdbook mdbook-katex`

# ssh-agent forwarding

One challenge in WSL is to forward the ssh-agent from WSL into the devcontainer. Make sure to include the following in `devcontainer.json` for this to happen.

```json
{
  "customizations": {
    "vscode": {
      "settings": {
        "remote.containers.sshForwardAgent": true
      }
    }
  }
}
```

